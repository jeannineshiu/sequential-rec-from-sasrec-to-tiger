"""TIGER-style generative recommender over semantic ID tokens (Week 5 Day 5-7).

Deliberately *not* a T5 encoder-decoder. This reuses SASRec's backbone --
the same causal blocks, the same pre-LN ordering, the same tied output head --
and changes exactly one thing: an item is no longer one atomic embedding but a
sequence of L semantic ID tokens, so the model predicts the next item by
autoregressively emitting its codes.

That is the whole point. Week 4 spent itself discovering that a comparison
between two architectures had quietly become a comparison between two framework
defaults; swapping in a different transformer stack alongside the different item
representation would repeat the mistake in a new place. With a shared backbone,
"atomic vs semantic ID" is the only variable that moves.

Token layout: sequences live on a grid of L tokens per item, left-padded to a
whole number of items, so a token at position p always belongs to level p % L.
Logits are restricted to the target position's level slice -- at decode time
only that slice is legal, and training against the same restricted softmax
keeps the two consistent.
"""

import torch
import torch.nn as nn

from src.models.sasrec import PointWiseFeedForward, build_masks


class GenRec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_levels: int,
        level_offsets: list[int],
        level_sizes: list[int],
        maxlen_items: int = 50,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
        pos_emb_type: str = "learnable",
    ):
        super().__init__()
        if pos_emb_type not in ("learnable", "none"):
            raise ValueError(f"unknown pos_emb_type: {pos_emb_type}")
        self.vocab_size = vocab_size
        self.n_levels = n_levels
        self.maxlen_items = maxlen_items
        self.maxlen_tokens = maxlen_items * n_levels
        self.hidden_dim = hidden_dim
        self.pos_emb_type = pos_emb_type

        self.register_buffer(
            "level_offsets", torch.tensor(level_offsets, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "level_sizes", torch.tensor(level_sizes, dtype=torch.long), persistent=False
        )
        # [L, V] mask of which vocabulary entries are legal at each level.
        legal = torch.zeros(n_levels, vocab_size, dtype=torch.bool)
        for level, (start, size) in enumerate(zip(level_offsets, level_sizes)):
            legal[level, start : start + size] = True
        self.register_buffer("level_legal", legal, persistent=False)

        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        if pos_emb_type == "learnable":
            self.pos_emb = nn.Embedding(self.maxlen_tokens + 1, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.attn_layernorms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ffn_layernorms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attn_layernorms.append(nn.LayerNorm(hidden_dim, eps=1e-8))
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_layernorms.append(nn.LayerNorm(hidden_dim, eps=1e-8))
            self.ffn_layers.append(PointWiseFeedForward(hidden_dim, dropout))

        self.last_layernorm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def encode(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """input_tokens: LongTensor [B, T] (0 = padding) -> hidden states [B, T, D]."""
        B, T = input_tokens.shape
        device = input_tokens.device
        causal_mask, key_padding_mask = build_masks(input_tokens)

        seqs = self.token_emb(input_tokens) * (self.hidden_dim**0.5)
        if self.pos_emb_type == "learnable":
            positions = torch.arange(1, T + 1, device=device).unsqueeze(0).expand(B, T)
            positions = positions * (~key_padding_mask).long()
            seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        seqs = seqs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        for i in range(len(self.attn_layers)):
            q = self.attn_layernorms[i](seqs)
            attn_out, _ = self.attn_layers[i](
                q,
                seqs,
                seqs,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            seqs = q + attn_out
            seqs = self.ffn_layernorms[i](seqs)
            seqs = self.ffn_layers[i](seqs)
            # A query at an all-padding position attends to nothing and softmaxes
            # to NaN; overwriting with 0 here (rather than multiplying) clears it.
            seqs = seqs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return self.last_layernorm(seqs)

    def logits(self, input_tokens: torch.Tensor, target_levels: torch.Tensor) -> torch.Tensor:
        """-> [B, T, V] logits with non-target-level vocabulary masked to -inf.

        target_levels: [T] or [B, T], the level of the token each position predicts.
        """
        hidden = self.encode(input_tokens)
        logits = hidden @ self.token_emb.weight.T  # tied head, as in SASRec
        legal = self.level_legal[target_levels]
        if legal.dim() == 2:  # [T, V] -> broadcast over batch
            legal = legal.unsqueeze(0)
        return logits.masked_fill(~legal, float("-inf"))

    def target_levels_for(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Level predicted at each input position of a length-`seq_len` window.

        Position p of the input predicts the token at grid position p + 1, hence
        level (p + 1) % L. Valid because left padding is a whole number of items,
        which keeps the grid aligned.
        """
        return (torch.arange(seq_len, device=device) + 1) % self.n_levels

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Training forward: [B, T] -> level-masked logits [B, T, V]."""
        levels = self.target_levels_for(input_tokens.shape[1], input_tokens.device)
        return self.logits(input_tokens, levels)

    # -- cached decoding ------------------------------------------------
    #
    # Scoring a candidate re-runs the whole history through the encoder, once
    # per candidate: 101 candidates x 196 history tokens to read 4 numbers. The
    # history is identical across candidates, so its keys and values are cached
    # here and only the candidate's own tokens are pushed through attention.
    # Measured on Beauty, this is the difference between a ~9-minute sampled
    # evaluation and a ~15-second one, which is what makes per-epoch validation
    # affordable at all.

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """[..., T, D] -> [..., H, T, dh]."""
        *lead, T, D = x.shape
        n_heads = self.attn_layers[0].num_heads
        return x.reshape(*lead, T, n_heads, D // n_heads).transpose(-3, -2)

    def _qkv(self, layer: nn.MultiheadAttention, x: torch.Tensor, part: str) -> torch.Tensor:
        """Apply one third of MultiheadAttention's packed input projection."""
        D = self.hidden_dim
        index = {"q": 0, "k": 1, "v": 2}[part]
        weight = layer.in_proj_weight[index * D : (index + 1) * D]
        bias = layer.in_proj_bias[index * D : (index + 1) * D] if layer.in_proj_bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)

    @torch.no_grad()
    def build_cache(self, history_tokens: torch.Tensor) -> dict:
        """Encode a history once, keeping each layer's keys/values.

        Returns the per-layer K/V, the padding mask, the final hidden state of
        the last history position (which predicts level 0 and is therefore
        shared by every candidate), and the history length.
        """
        history_tokens = self.trim_history(history_tokens)
        B, T = history_tokens.shape
        causal_mask, key_padding_mask = build_masks(history_tokens)

        seqs = self.token_emb(history_tokens) * (self.hidden_dim**0.5)
        if self.pos_emb_type == "learnable":
            positions = torch.arange(1, T + 1, device=history_tokens.device).unsqueeze(0).expand(B, T)
            positions = positions * (~key_padding_mask).long()
            seqs = seqs + self.pos_emb(positions)
        seqs = seqs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        layers = []
        for i in range(len(self.attn_layers)):
            # Keys/values are projected from the *unnormalized* stream, as in
            # SASRec's block (query normalized, key/value not).
            layers.append(
                (
                    self._heads(self._qkv(self.attn_layers[i], seqs, "k")),
                    self._heads(self._qkv(self.attn_layers[i], seqs, "v")),
                )
            )
            q = self.attn_layernorms[i](seqs)
            attn_out, _ = self.attn_layers[i](
                q, seqs, seqs, attn_mask=causal_mask,
                key_padding_mask=key_padding_mask, need_weights=False,
            )
            seqs = q + attn_out
            seqs = self.ffn_layernorms[i](seqs)
            seqs = self.ffn_layers[i](seqs)
            seqs = seqs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return {
            "layers": layers,
            "key_padding_mask": key_padding_mask,
            "last_hidden": self.last_layernorm(seqs)[:, -1, :],
            "length": T,
        }

    @torch.no_grad()
    def _decode_with_cache(self, cache: dict, tokens: torch.Tensor) -> torch.Tensor:
        """Hidden states for `tokens` [B, C, S] appended after the cached history.

        -> [B, C, S, D], the states that predict the tokens at levels 1..S.
        """
        B, C, S = tokens.shape
        T = cache["length"]
        device = tokens.device

        positions = torch.arange(T + 1, T + S + 1, device=device).clamp(max=self.maxlen_tokens)
        seqs = self.token_emb(tokens) * (self.hidden_dim**0.5)
        if self.pos_emb_type == "learnable":
            seqs = seqs + self.pos_emb(positions).view(1, 1, S, -1)

        # Candidate tokens see the whole history plus their own past.
        history_mask = cache["key_padding_mask"].view(B, 1, 1, 1, T)
        self_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        n_heads = self.attn_layers[0].num_heads
        scale = (self.hidden_dim // n_heads) ** -0.5

        for i in range(len(self.attn_layers)):
            layer = self.attn_layers[i]
            k_hist, v_hist = cache["layers"][i]  # [B, H, T, dh]
            k_new = self._heads(self._qkv(layer, seqs, "k"))  # [B, C, H, S, dh]
            v_new = self._heads(self._qkv(layer, seqs, "v"))

            q = self._heads(self._qkv(layer, self.attn_layernorms[i](seqs), "q"))

            scores_hist = torch.einsum("bchsd,bhtd->bchst", q, k_hist) * scale
            scores_hist = scores_hist.masked_fill(history_mask, float("-inf"))
            scores_new = torch.einsum("bchsd,bchtd->bchst", q, k_new) * scale
            scores_new = scores_new.masked_fill(self_mask.view(1, 1, 1, S, S), float("-inf"))

            weights = torch.softmax(torch.cat([scores_hist, scores_new], dim=-1), dim=-1)
            attn = torch.einsum("bchst,bhtd->bchsd", weights[..., :T], v_hist)
            attn = attn + torch.einsum("bchst,bchtd->bchsd", weights[..., T:], v_new)

            attn = attn.transpose(-3, -2).reshape(B, C, S, self.hidden_dim)
            attn = layer.out_proj(attn)

            seqs = self.attn_layernorms[i](seqs) + attn
            # The feed-forward block is a Conv1d over positions, so it wants a
            # 3-D batch; candidates are independent here and flatten cleanly.
            flat = self.ffn_layernorms[i](seqs).reshape(B * C, S, self.hidden_dim)
            seqs = self.ffn_layers[i](flat).reshape(B, C, S, self.hidden_dim)

        return self.last_layernorm(seqs)

    def trim_history(self, history_tokens: torch.Tensor) -> torch.Tensor:
        """Reserve the last item slot of the window for the item being generated.

        Scoring and decoding both append tokens to the history, which would push
        the sequence past `maxlen_tokens` -- positions the positional embedding
        was never trained on. Reserving one item slot up front means neither path
        ever overflows, and (this is the part worth being careful about) *both
        paths see the identical context*, so a beam's score and the same item's
        candidate score are the same number rather than two nearby ones.

        Trimming whole items keeps position % L, and therefore the level grid,
        intact. Sequences shorter than the window are left-padded, so this
        usually discards padding; a full window loses its oldest item.
        """
        keep = self.maxlen_tokens - self.n_levels
        if history_tokens.shape[1] <= keep:
            return history_tokens
        return history_tokens[:, history_tokens.shape[1] - keep :]

    @torch.no_grad()
    def next_token_logprobs(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        """Log-probs of the next token given `tokens` [B, T] -> [B, V].

        `level` is the level of the token being predicted. `tokens` must already
        be a trimmed history plus the tokens generated so far -- see
        `trim_history`. Used by both the candidate scorer and beam search.
        """
        hidden = self.encode(tokens)[:, -1, :]
        logits = hidden @ self.token_emb.weight.T
        logits = logits.masked_fill(~self.level_legal[level].unsqueeze(0), float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def score_item_tokens(
        self, history_tokens: torch.Tensor, item_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Log-likelihood of each candidate item's full code sequence.

        history_tokens: [B, T]  (left-padded, grid-aligned)
        item_tokens:    [B, C, L]
        -> [B, C] summed log p(code_l | history, code_<l)

        This is what makes the generative model scorable by the *same*
        evaluators as SASRec: a candidate's score is the log-probability the
        model assigns to generating it. One forward pass per (row, candidate)
        with the candidate's own tokens in context, teacher-forced, so all L
        levels are read off a single pass rather than L sequential ones.
        """
        B, C, L = item_tokens.shape
        history_tokens = self.trim_history(history_tokens)
        T = history_tokens.shape[1]

        # [B*C, T + L - 1]: history followed by the candidate's first L-1 tokens;
        # position T-1+l then predicts the candidate's level-l token.
        history = history_tokens.unsqueeze(1).expand(B, C, T).reshape(B * C, T)
        candidates = item_tokens.reshape(B * C, L)
        context = torch.cat([history, candidates[:, : L - 1]], dim=1)

        hidden = self.encode(context)[:, -L:, :]  # [B*C, L, D]
        logits = hidden @ self.token_emb.weight.T  # [B*C, L, V]
        levels = torch.arange(L, device=logits.device)
        logits = logits.masked_fill(~self.level_legal[levels].unsqueeze(0), float("-inf"))
        logprobs = torch.log_softmax(logits, dim=-1)

        token_logprobs = logprobs.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)  # [B*C, L]
        return token_logprobs.sum(-1).reshape(B, C)

    @torch.no_grad()
    def score_item_tokens_cached(
        self, history_tokens: torch.Tensor, item_tokens: torch.Tensor
    ) -> torch.Tensor:
        """`score_item_tokens`, with the history encoded once instead of C times.

        Numerically identical to the uncached path (asserted in the tests) --
        this is purely the same computation with the shared prefix shared.
        """
        B, C, L = item_tokens.shape
        cache = self.build_cache(history_tokens)

        # The level-0 prediction depends only on the history, so all C
        # candidates read it off the one state the cache already holds.
        head = self.token_emb.weight.T
        level0 = cache["last_hidden"] @ head
        level0 = level0.masked_fill(~self.level_legal[0].unsqueeze(0), float("-inf"))
        level0 = torch.log_softmax(level0, dim=-1)  # [B, V]
        total = level0.gather(-1, item_tokens[:, :, 0])  # [B, C]

        if L > 1:
            hidden = self._decode_with_cache(cache, item_tokens[:, :, : L - 1])
            logits = hidden @ head  # [B, C, L-1, V]
            levels = torch.arange(1, L, device=logits.device)
            logits = logits.masked_fill(~self.level_legal[levels].view(1, 1, L - 1, -1), float("-inf"))
            logprobs = torch.log_softmax(logits, dim=-1)
            total = total + logprobs.gather(-1, item_tokens[:, :, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)

        return total

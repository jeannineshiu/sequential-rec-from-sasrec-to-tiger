"""SASRec (Kang & McAuley, 2018): self-attentive sequential recommendation.

Follows the original architecture closely (see the widely-used pmixer/SASRec.pytorch
port for the reference PyTorch idioms this mirrors):
- item embedding scaled by sqrt(hidden_dim), + learnable positional embedding
  indexed by *absolute slot position in the padded window* (not relative-to-content),
  zeroed at padding slots via padding_idx=0.
- N causal self-attention blocks (pre-LN: LayerNorm -> MHA -> residual), each
  followed by a point-wise (kernel-size-1 conv) feed-forward block.
- Prediction head shares weights with the input item embedding (no separate
  output projection).
"""

import torch
import torch.nn as nn


def sinusoidal_position_table(maxlen: int, hidden_dim: int) -> torch.Tensor:
    """Fixed (non-learnable) sinusoidal position encoding, shape [maxlen + 1, D].

    Row 0 is the padding slot (zeros); rows 1..maxlen follow the standard
    Vaswani et al. (2017) formula for positions 0..maxlen-1, used here for the
    A1 positional-embedding ablation (learnable vs none vs sinusoidal).
    """
    table = torch.zeros(maxlen + 1, hidden_dim)
    position = torch.arange(0, maxlen, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
    )
    table[1:, 0::2] = torch.sin(position * div_term)
    table[1:, 1::2] = torch.cos(position * div_term[: table[1:, 1::2].shape[1]])
    return table


def build_masks(input_seqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    input_seqs: LongTensor [B, T], 0 = padding.

    Returns:
        causal_mask: BoolTensor [T, T], True = position i may NOT attend to
                     position j (upper triangle, j > i -- future positions).
        key_padding_mask: BoolTensor [B, T], True = this key position is
                          padding and must be excluded from attention.
    """
    T = input_seqs.shape[1]
    causal_mask = torch.triu(
        torch.ones(T, T, device=input_seqs.device, dtype=torch.bool), diagonal=1
    )
    key_padding_mask = input_seqs == 0
    return causal_mask, key_padding_mask


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> conv over the D dim needs [B, D, T]
        out = x.transpose(-1, -2)
        out = self.dropout1(self.relu(self.conv1(out)))
        out = self.dropout2(self.conv2(out))
        out = out.transpose(-1, -2)
        return out + x


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        maxlen: int = 200,
        hidden_dim: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
        pos_emb_type: str = "learnable",
    ):
        super().__init__()
        if pos_emb_type not in ("learnable", "none", "sinusoidal"):
            raise ValueError(f"unknown pos_emb_type: {pos_emb_type}")
        self.n_items = n_items
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim
        self.pos_emb_type = pos_emb_type

        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        # positions are 1-indexed absolute slots in the window; slot 0 (padding_idx)
        # is reserved so padded positions contribute a zero positional embedding.
        if pos_emb_type == "learnable":
            self.pos_emb = nn.Embedding(maxlen + 1, hidden_dim, padding_idx=0)
        elif pos_emb_type == "sinusoidal":
            self.register_buffer("pos_emb_table", sinusoidal_position_table(maxlen, hidden_dim))
        # "none": no positional embedding parameter/buffer at all
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

    def encode(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """input_seqs: LongTensor [B, T] (0 = padding) -> hidden states [B, T, D]."""
        B, T = input_seqs.shape
        device = input_seqs.device
        causal_mask, key_padding_mask = build_masks(input_seqs)

        seqs = self.item_emb(input_seqs) * (self.hidden_dim**0.5)
        if self.pos_emb_type != "none":
            positions = torch.arange(1, T + 1, device=device).unsqueeze(0).expand(B, T)
            positions = positions * (~key_padding_mask).long()
            if self.pos_emb_type == "learnable":
                seqs = seqs + self.pos_emb(positions)
            else:  # sinusoidal
                seqs = seqs + self.pos_emb_table[positions]
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
            seqs = seqs.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return self.last_layernorm(seqs)

    def forward(
        self, input_seqs: torch.Tensor, pos_seqs: torch.Tensor, neg_seqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward: per-position pos/neg logits (BCE-with-logits target)."""
        seq_out = self.encode(input_seqs)
        pos_logits = (seq_out * self.item_emb(pos_seqs)).sum(-1)
        neg_logits = (seq_out * self.item_emb(neg_seqs)).sum(-1)
        return pos_logits, neg_logits

    def score(self, input_seqs: torch.Tensor, candidate_items: torch.Tensor) -> torch.Tensor:
        """input_seqs: [B, T]; candidate_items: [B, C] -> scores [B, C]."""
        final_hidden = self.encode(input_seqs)[:, -1, :]  # [B, D]
        candidate_emb = self.item_emb(candidate_items)  # [B, C, D]
        return torch.bmm(candidate_emb, final_hidden.unsqueeze(-1)).squeeze(-1)

    def score_full_catalog(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """input_seqs: [B, T] -> scores [B, n_items + 1] (column 0 unused)."""
        final_hidden = self.encode(input_seqs)[:, -1, :]  # [B, D]
        return final_hidden @ self.item_emb.weight.T

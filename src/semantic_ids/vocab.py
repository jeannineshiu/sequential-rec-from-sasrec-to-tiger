"""Semantic ID vocabulary: codes <-> token ids, plus the Trie of legal sequences.

An item's semantic ID is L codes (3 quantizer levels + 1 disambiguation token).
Each level gets its own disjoint slice of one shared token vocabulary, so a
token id says both *which level* it belongs to and *which code* it is:

    0                 PAD
    1                 BOS (decoder start / sequence start)
    2 .. 2+K1         level-1 codes
    ...               level-l codes
                      level-L codes (disambiguation)

Level-disjoint ids make the level-validity constraint free -- at decode step l
only that slice is legal -- so the Trie only has to enforce the harder
constraint: that the emitted prefix corresponds to at least one real item.
"""

from pathlib import Path

import numpy as np

PAD = 0
BOS = 1
N_SPECIAL = 2


class SemanticIdVocab:
    """Maps between internal item ids, code tuples, and flat token ids."""

    def __init__(self, item_ids: np.ndarray, codes: np.ndarray):
        if len(item_ids) != len(codes):
            raise ValueError("item_ids and codes must have the same length")

        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.codes = np.asarray(codes, dtype=np.int64)
        self.n_levels = int(self.codes.shape[1])
        # Level sizes come from the data rather than the config: the
        # disambiguation level's size is whatever the largest collision group
        # turned out to be, and hard-coding it would silently truncate.
        self.level_sizes = [int(self.codes[:, level].max()) + 1 for level in range(self.n_levels)]

        offsets, running = [], N_SPECIAL
        for size in self.level_sizes:
            offsets.append(running)
            running += size
        self.level_offsets = offsets
        self.vocab_size = running

        n_items = int(self.item_ids.max())
        # Row 0 is the padding item, encoded as all-PAD so a padded item slot
        # contributes n_levels padding tokens and stays aligned to the grid.
        self.item_tokens = np.zeros((n_items + 1, self.n_levels), dtype=np.int64)
        for row, item in enumerate(self.item_ids):
            self.item_tokens[item] = self.codes[row] + np.array(offsets, dtype=np.int64)

        self.token_to_item = {tuple(int(t) for t in tok): int(item)
                              for item, tok in enumerate(self.item_tokens) if item > 0}
        if len(self.token_to_item) != len(self.item_ids):
            raise ValueError("semantic IDs are not unique -- disambiguation token missing?")

        self._trie = self._build_trie()

    # -- encoding -------------------------------------------------------

    def encode_items(self, items: np.ndarray) -> np.ndarray:
        """items [..., N] -> tokens [..., N * n_levels], item order preserved."""
        items = np.asarray(items, dtype=np.int64)
        tokens = self.item_tokens[items]
        return tokens.reshape(*items.shape[:-1], items.shape[-1] * self.n_levels)

    def decode_tokens(self, tokens) -> int | None:
        """One item's token tuple -> internal item id, or None if illegal."""
        return self.token_to_item.get(tuple(int(t) for t in tokens))

    def level_of_position(self, position: int) -> int:
        """Which level the token at a token-grid position belongs to."""
        return position % self.n_levels

    def level_slice(self, level: int) -> tuple[int, int]:
        """[start, end) token ids belonging to `level`."""
        start = self.level_offsets[level]
        return start, start + self.level_sizes[level]

    # -- constrained decoding -------------------------------------------

    def _build_trie(self) -> dict[tuple, np.ndarray]:
        """prefix tuple -> sorted array of tokens that continue it to a real item."""
        children: dict[tuple, set[int]] = {}
        for tokens in self.item_tokens[1:]:
            for level in range(self.n_levels):
                prefix = tuple(int(t) for t in tokens[:level])
                children.setdefault(prefix, set()).add(int(tokens[level]))
        return {prefix: np.array(sorted(tokens), dtype=np.int64) for prefix, tokens in children.items()}

    def allowed_next(self, prefix) -> np.ndarray:
        """Tokens that extend `prefix` toward at least one real item (may be empty)."""
        key = tuple(int(t) for t in prefix)
        return self._trie.get(key, np.empty(0, dtype=np.int64))

    def build_mask_table(self) -> tuple[dict[tuple, int], np.ndarray]:
        """(prefix -> row index, [n_prefixes, vocab_size] bool legality masks).

        Lets beam search apply the Trie constraint as one tensor op per level
        instead of a Python loop over (row, beam) pairs -- which, with the
        history cache in place, would otherwise become the slowest part of
        decoding. Built on first use; ~14 MB on Beauty.
        """
        if getattr(self, "_mask_table", None) is None:
            prefixes = sorted(self._trie)
            index = {prefix: row for row, prefix in enumerate(prefixes)}
            masks = np.zeros((len(prefixes), self.vocab_size), dtype=bool)
            for prefix, row in index.items():
                masks[row, self._trie[prefix]] = True
            self._mask_table = (index, masks)
        return self._mask_table

    def is_complete_item(self, tokens) -> bool:
        return len(tokens) == self.n_levels and self.decode_tokens(tokens) is not None

    # -- io --------------------------------------------------------------

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "SemanticIdVocab":
        data = np.load(Path(data_dir) / "semantic_ids" / "semantic_ids.npz")
        return cls(data["item_ids"], data["codes"])

    def summary(self) -> str:
        sizes = " + ".join(str(s) for s in self.level_sizes)
        return (
            f"SemanticIdVocab: {len(self.item_ids)} items, {self.n_levels} levels "
            f"({sizes} codes), vocab {self.vocab_size} tokens"
        )

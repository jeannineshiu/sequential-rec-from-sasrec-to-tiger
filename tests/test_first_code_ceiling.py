"""The oracle ladder's grouping is what makes the whole decomposition mean
anything: if `prefix_group_ids` grouped wrongly, every depth would still produce
a plausible-looking monotone table.
"""

import numpy as np

from scripts.first_code_ceiling import prefix_group_ids


def _tokens():
    # Row 0 is the padding item, as SemanticIdVocab builds it. Items 1 and 2
    # share their first two codes; item 3 shares only the first; item 4 shares
    # nothing.
    return np.array(
        [
            [0, 0, 0],
            [2, 10, 20],
            [2, 10, 21],
            [2, 11, 22],
            [3, 12, 23],
        ]
    )


def test_depth_zero_is_one_group():
    groups = prefix_group_ids(_tokens(), 0)
    assert len(set(groups.tolist())) == 1


def test_depth_one_groups_by_first_code():
    groups = prefix_group_ids(_tokens(), 1)
    assert groups[1] == groups[2] == groups[3]
    assert groups[4] != groups[1]
    assert groups[0] != groups[1]  # padding row must not join a real group


def test_depth_two_splits_what_depth_one_joined():
    groups = prefix_group_ids(_tokens(), 2)
    assert groups[1] == groups[2]
    assert groups[3] != groups[1]


def test_full_depth_separates_every_item():
    tokens = _tokens()
    groups = prefix_group_ids(tokens, tokens.shape[1])
    assert len(set(groups.tolist())) == len(tokens)


def test_groups_get_finer_monotonically():
    """Each extra oracle code may only split groups, never merge them."""
    tokens = _tokens()
    counts = [len(set(prefix_group_ids(tokens, d).tolist())) for d in range(tokens.shape[1] + 1)]
    assert counts == sorted(counts)
    assert counts[0] == 1

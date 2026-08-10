"""Semantic ID construction: text join, residual quantization, collision handling."""

import numpy as np
import pytest

from src.semantic_ids.rq_kmeans import add_disambiguation, codebook_usage, fit_rq_kmeans
from src.semantic_ids.text import _beauty_text, ml1m_texts


def _clustered_embeddings(n_clusters=8, per_cluster=12, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 5
    points = np.repeat(centers, per_cluster, axis=0) + rng.normal(
        size=(n_clusters * per_cluster, dim)
    )
    return points.astype(np.float32)


def test_rq_kmeans_shapes_and_monotonic_residual():
    embeddings = _clustered_embeddings()
    codes, codebooks, residual_norms = fit_rq_kmeans(
        embeddings, n_levels=3, n_codes=8, seed=42, n_init=3
    )

    assert codes.shape == (len(embeddings), 3)
    assert codebooks.shape == (3, 8, embeddings.shape[1])
    assert codes.min() >= 0 and codes.max() < 8
    # Each extra level explains some of what the previous ones left over.
    assert all(a > b for a, b in zip(residual_norms, residual_norms[1:]))


def test_rq_kmeans_is_deterministic_given_seed():
    embeddings = _clustered_embeddings()
    codes_a, _, _ = fit_rq_kmeans(embeddings, n_levels=2, n_codes=8, seed=42, n_init=3)
    codes_b, _, _ = fit_rq_kmeans(embeddings, n_levels=2, n_codes=8, seed=42, n_init=3)
    assert np.array_equal(codes_a, codes_b)


def test_rq_kmeans_recovers_planted_clusters():
    """With as many codes as planted clusters, level 1 should recover them."""
    embeddings = _clustered_embeddings(n_clusters=8, per_cluster=12)
    codes, _, _ = fit_rq_kmeans(embeddings, n_levels=1, n_codes=8, seed=42, n_init=10)
    for cluster in range(8):
        block = codes[cluster * 12 : (cluster + 1) * 12, 0]
        assert len(np.unique(block)) == 1, "a planted cluster was split across codes"


def test_disambiguation_token_makes_every_id_unique():
    codes = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [5, 6]], dtype=np.int32)
    full, stats = add_disambiguation(codes)

    ids = {tuple(row) for row in full}
    assert len(ids) == len(codes), "semantic IDs are not unique after disambiguation"
    assert full[:, -1].tolist() == [0, 1, 2, 0, 1, 0]
    assert stats["n_colliding_items"] == 3  # the 2nd/3rd [1,2] and the 2nd [3,4]
    assert stats["max_group_size"] == 3
    assert stats["n_unique_prefixes"] == 3


def test_disambiguation_is_noop_without_collisions():
    codes = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    full, stats = add_disambiguation(codes)
    assert full[:, -1].tolist() == [0, 0, 0]
    assert stats["collision_rate"] == 0.0


def test_codebook_usage_counts_dead_codes():
    codes = np.array([[0], [0], [2]], dtype=np.int32)
    usage = codebook_usage(codes, n_codes=4)[0]
    assert usage["codes_used"] == 2
    assert usage["dead_codes"] == 2
    assert usage["max_items_per_code"] == 2


def test_ml1m_text_join_skips_filtered_items(tmp_path):
    (tmp_path / "movies.dat").write_text(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "2::Jumanji (1995)::Adventure|Fantasy\n"
        "3::Dropped Movie (1999)::Drama\n",
        encoding="latin-1",
    )
    # movie 3 was removed by 5-core filtering, so it has no internal id.
    texts = ml1m_texts(tmp_path, {"1": 1, "2": 2})

    assert texts[1] == "Toy Story (1995). Genres: Animation, Children's, Comedy"
    assert set(texts) == {1, 2}


def test_beauty_text_uses_longest_category_path_and_drops_constant_root():
    record = {
        "title": "Some Shampoo",
        "categories": [["Beauty", "Hair Care"], ["Beauty", "Hair Care", "Shampoos"]],
        "brand": "Acme",
    }
    assert _beauty_text(record) == "Some Shampoo. Category: Hair Care > Shampoos. Brand: Acme"


def test_beauty_text_tolerates_missing_fields():
    assert _beauty_text({"categories": [["Beauty", "Makeup"]]}) == "Category: Makeup"
    assert _beauty_text({"title": "  ", "brand": None, "categories": []}) == ""


@pytest.mark.parametrize("n_levels", [1, 3])
def test_full_semantic_ids_are_unique_on_real_shaped_data(n_levels):
    embeddings = _clustered_embeddings(n_clusters=4, per_cluster=25, dim=8, seed=7)
    codes, _, _ = fit_rq_kmeans(embeddings, n_levels=n_levels, n_codes=4, seed=42, n_init=3)
    full, _ = add_disambiguation(codes)
    assert len({tuple(row) for row in full}) == len(embeddings)

import pytest
from src.dsrqs.benchmark import generate_benchmark, DATASET_META
from src.dsrqs.metrics import path_coherence_score


@pytest.mark.parametrize("key", DATASET_META.keys())
def test_generate_sizes(key):
    data = generate_benchmark(key, n_queries=5, seed=0)
    assert len(data) == 5
    assert all("gold_paths" in q and "relations" in q for q in data)


def test_omim_hop3_three_hops():
    data = generate_benchmark("omim_hop3", n_queries=10, seed=1)
    for q in data:
        for path in q["gold_paths"]:
            assert len(path) == 3


def test_relations_have_hrt():
    q = generate_benchmark("orphanet_fq274", n_queries=1, seed=0)[0]
    assert all("h" in r and "t" in r for r in q["relations"])


def test_pcs_from_gold_labels():
    q = generate_benchmark("disgenet_rd411", n_queries=1, seed=0)[0]
    retained = {(r["r"], r["hop"]) for r in q["relations"] if r["label"] == 1}
    assert path_coherence_score(q["gold_paths"], retained) == 1.0

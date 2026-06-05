# =============================================================================
# Author: Marwan Dhifallah
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology
#
# Description:
#   Paper-faithful benchmark construction (Orphanet-FQ274, DisGeNET-RD411,
#   OMIM-Hop3) with depth-conditional labels and full (h,r,t) edges.
#
# Copyright (c) 2026
# =============================================================================
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

Edge = Tuple[str, str, str]

# Relation inventories (paper §2.1 / appendix)
RELATIONS_ORPHANET = [
    "causal_gene", "associated_with", "expressed_in", "participates_in",
    "targets", "phenotype", "part_of", "regulates",
]
RELATIONS_DISGENET = [
    "associated_gene", "gene_role", "pathway_member", "biomarker",
    "therapeutic_target", "epistatic", "expressed_in", "causal_mut",
    "has_phenotype", "treats", "gene_disease", "pathway",
]
RELATIONS_OMIM = [
    "causal_gene", "allelic_variant_of", "has_phenotype", "encodes_protein",
    "functions_in_pathway", "phenotypic_series_component", "expressed_in",
    "associated_with", "gene_disease",
]

INTENTS = ["Etiology", "Treatment", "Phenotype", "Gene-Function"]

# Depth-conditional relevance templates driving PCE (paper §3.2, Fig. 6)
PCE_RULES: Dict[str, Dict[int, Optional[int]]] = {
    "expressed_in":       {1: 0, 2: 1, 3: 0},
    "causal_gene":        {1: 1, 2: 0, 3: 0},
    "allelic_variant_of": {1: 0, 2: 1, 3: 0},
    "has_phenotype":      {1: 0, 2: 0, 3: 1},
    "causal_mut":         {1: 0, 2: 0, 3: 1},
    "phenotypic_series_component": {1: 0, 2: 1, 3: 0},
    "targets":            {1: 0, 2: 1, 3: 0},
    "treats":             {1: 1, 2: 0, 3: 0},
    "pathway":            {1: 0, 2: 1, 3: 0},
}

QUERY_TEMPLATES = {
    "Etiology": "What is the genetic basis of {disease}?",
    "Treatment": "What targeted therapy exploits the molecular mechanism of {disease}?",
    "Phenotype": "What facial dysmorphisms are associated with the allelic variant underlying {disease}?",
    "Gene-Function": (
        "Through which cellular pathway does the protein product of the causal "
        "gene of {disease} exert its pathogenic effect?"
    ),
}

DATASET_META = {
    "orphanet_fq274": {
        "dataset": "orphanet",
        "n_queries": 274,
        "max_hops": 2,
        "relations": RELATIONS_ORPHANET,
    },
    "disgenet_rd411": {
        "dataset": "disgenet",
        "n_queries": 411,
        "max_hops": 2,
        "relations": RELATIONS_DISGENET,
    },
    "omim_hop3": {
        "dataset": "omim",
        "n_queries": 183,
        "max_hops": 3,
        "relations": RELATIONS_OMIM,
    },
}


def _label_for_relation(r: str, hop: int, on_gold: bool, rng: random.Random) -> int:
    if on_gold:
        return 1
    if r in PCE_RULES and hop in PCE_RULES[r]:
        fixed = PCE_RULES[r][hop]
        if fixed is not None:
            return fixed
    return rng.randint(0, 1)


def _gold_path_for_intent(
    intent: str,
    disease: str,
    gene: str,
    variant: str,
    pathway: str,
    phenotype: str,
    max_hops: int,
) -> List[List[Edge]]:
    if max_hops >= 3:
        if intent == "Phenotype":
            path = [
                (disease, "causal_gene", gene),
                (gene, "allelic_variant_of", variant),
                (variant, "has_phenotype", phenotype),
            ]
        else:
            path = [
                (disease, "causal_gene", gene),
                (gene, "encodes_protein", f"{gene} protein"),
                (f"{gene} protein", "functions_in_pathway", pathway),
            ]
        return [path]

    if intent == "Treatment":
        path = [
            (disease, "causal_gene", gene),
            (gene, "targets", f"Therapy-{gene[:3]}"),
        ]
    elif intent == "Phenotype":
        path = [
            (disease, "has_phenotype", phenotype),
        ]
    else:
        path = [
            (disease, "causal_gene", gene),
            (gene, "participates_in", pathway),
        ]
    return [path]


def _distractor_edges(
    gold_edges: Set[Tuple[Edge, int]],
    rel_pool: List[str],
    max_hops: int,
    rng: random.Random,
    n_per_hop: int = 4,
) -> List[Dict[str, Any]]:
    extras: List[Dict[str, Any]] = []
    for hop in range(1, max_hops + 1):
        used_r = {e[1] for e, h in gold_edges if h == hop}
        candidates = [r for r in rel_pool if r not in used_r]
        rng.shuffle(candidates)
        for r in candidates[:n_per_hop]:
            extras.append({
                "h": f"Entity-{hop}-{r[:4]}",
                "r": r,
                "t": f"Tail-{hop}-{r[:4]}",
                "hop": hop,
                "label": _label_for_relation(r, hop, on_gold=False, rng=rng),
            })
    return extras


def build_query_item(
    qid: int,
    dataset_key: str,
    disease: str,
    gene: str,
    intent: str,
    rng: random.Random,
) -> Dict[str, Any]:
    meta = DATASET_META[dataset_key]
    max_hops = meta["max_hops"]
    rel_pool = meta["relations"]

    variant = f"{gene}-{rng.choice(['R352W', 'L444P', 'N370S', 'del'])}"
    pathway = rng.choice([
        "DNA Repair", "Lipid Metabolism", "Glycosylation",
        "HR repair", "Lysosomal degradation",
    ])
    phenotype = rng.choice([
        "Microcephaly", "Ptosis", "Anteverted nares", "Hepatosplenomegaly",
    ])

    gold_paths = _gold_path_for_intent(
        intent, disease, gene, variant, pathway, phenotype, max_hops
    )

    relations: List[Dict[str, Any]] = []
    gold_edge_set: Set[Tuple[Edge, int]] = set()

    for path in gold_paths:
        for hop_idx, (h, r, t) in enumerate(path, start=1):
            gold_edge_set.add(((h, r, t), hop_idx))
            relations.append({
                "h": h, "r": r, "t": t,
                "hop": hop_idx,
                "label": 1,
            })

    relations.extend(
        _distractor_edges(gold_edge_set, rel_pool, max_hops, rng)
    )

    seen = set()
    unique: List[Dict[str, Any]] = []
    for rel in relations:
        key = (rel["h"], rel["r"], rel["t"], rel["hop"])
        if key not in seen:
            seen.add(key)
            unique.append(rel)

    query = QUERY_TEMPLATES.get(intent, "Query about {disease}").format(disease=disease)

    return {
        "qid": qid,
        "query": query,
        "intent": intent,
        "dataset": meta["dataset"],
        "seed_entities": [disease],
        "gold_paths": gold_paths,
        "relations": unique,
    }


def _entity_pools(dataset_key: str, rng: random.Random) -> Tuple[List[str], List[str]]:
    diseases = [
        "Gaucher Disease", "Fanconi Anaemia D1", "Smith-Lemli-Opitz Syndrome",
        "Cystic Fibrosis", "Duchenne Muscular Dystrophy", "Fabry Disease",
        "Pompe Disease", "Huntington Disease", "Beta Thalassemia",
        "Marfan Syndrome", "Krabbe Disease", "Niemann-Pick Disease",
    ]
    genes = [
        "GBA", "FANCD1", "BRCA2", "DHCR7", "CFTR", "DMD", "GLA", "GAA",
        "HTT", "HBB", "FBN1", "GALC", "SMPD1", "APOE", "SNCA",
    ]
    if dataset_key == "omim_hop3":
        diseases = diseases[:12] + ["Mitochondrial Cytopathy", "Mucopolysaccharidosis"]
    rng.shuffle(diseases)
    rng.shuffle(genes)
    return diseases, genes


def generate_benchmark(
    dataset_key: str,
    n_queries: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build full benchmark JSON list for one dataset."""
    if dataset_key not in DATASET_META:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    meta = DATASET_META[dataset_key]
    n = n_queries or meta["n_queries"]
    rng = random.Random(seed)
    diseases, genes = _entity_pools(dataset_key, rng)

    intent_cycle = INTENTS * ((n // len(INTENTS)) + 1)
    data: List[Dict[str, Any]] = []

    for qid in range(n):
        disease = diseases[qid % len(diseases)]
        gene = genes[qid % len(genes)]
        intent = intent_cycle[qid]
        data.append(
            build_query_item(qid, dataset_key, disease, gene, intent, rng)
        )

    return data


def generate_all_benchmarks(
    seed: int = 42,
    sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for key in DATASET_META:
        n = (sizes or {}).get(key, DATASET_META[key]["n_queries"])
        out[key] = generate_benchmark(key, n_queries=n, seed=seed + hash(key) % 1000)
    return out

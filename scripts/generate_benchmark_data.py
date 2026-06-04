# =============================================================================
# Generate Realistic Benchmark Datasets for DSRQS
# Creates test data matching the paper specifications
# =============================================================================
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Any


# Define realistic biomedical entities and relations for rare diseases
DISEASES = {
    "orphanet_fq274": [
        "Cystic Fibrosis", "Huntington Disease", "Duchenne Muscular Dystrophy",
        "Marfan Syndrome", "Gaucher Disease", "Hemophilia A", "Beta Thalassemia",
    ],
    "disgenet_rd411": [
        "Alzheimer Disease", "Parkinson Disease", "Spinal Muscular Atrophy",
        "Fanconi Anemia", "Krabbe Disease", "Niemann-Pick Disease",
    ],
    "omim_hop3": [
        "Smith-Lemli-Opitz Syndrome", "Fabry Disease", "Pompe Disease",
        "Lysosomal Storage Disorder", "Mitochondrial Cytopathy",
        "Peroxisomal Disorder", "Mucopolysaccharidosis",
    ],
}

GENES = [
    "CFTR", "HTT", "DMD", "FBN1", "GBA", "F8", "HBB",
    "APOE", "SNCA", "SMN1", "FANCD1", "GALC", "SMPD1",
]

RELATIONS = {
    "orphanet_fq274": [
        "causal_gene", "associated_with", "expressed_in",
        "participates_in", "targets", "phenotype"
    ],
    "disgenet_rd411": [
        "associated_gene", "gene_role", "pathway_member",
        "biomarker", "therapeutic_target", "epistatic"
    ],
}

INTENTS = ["Gene", "Treatment", "Phenotype", "Gene-Function", "Etiology"]

PHENOTYPES = [
    "Progressive Muscle Weakness", "Cone-Rod Dystrophy", "Cognitive Decline",
    "Behavioral Abnormality", "Growth Delay", "Developmental Delay",
]

PATHWAYS = [
    "DNA Repair", "Lipid Metabolism", "Glycosylation", "Protein Synthesis",
    "Apoptosis", "Cell Migration", "Proteasomal Degradation",
]


def generate_query_item(
    qid: int,
    dataset_name: str,
    disease: str,
    gene: str,
    intent: str,
) -> Dict[str, Any]:
    """
    Generate a single query with depth-conditional relation labels.
    
    A relation may be relevant at hop 1 but irrelevant at hop 2, or vice versa,
    creating the Position-Conflation Error problem that DSRQS solves.
    """
    # Query text templates
    query_templates = {
        "Gene": f"What gene causes {disease}?",
        "Treatment": f"What is a targeted therapy for {disease}?",
        "Phenotype": f"What are the main symptoms of {disease}?",
        "Gene-Function": f"Through what mechanism does {gene} cause {disease}?",
        "Etiology": f"What is the genetic basis of {disease}?",
    }
    
    # Generate depth-conditional relation labels
    # This is the KEY part: same relation may have different relevance at different depths
    relations = []
    rel_set = RELATIONS.get(dataset_name, [])
    
    for hop in [1, 2, 3]:
        for rel in random.sample(rel_set, min(3, len(rel_set))):
            # Simulate PCE: some relations change relevance between hops
            if rel == "expressed_in" and hop == 1:
                # "expressed_in" at hop 1 (disease->tissue) usually irrelevant
                label = 0
            elif rel == "expressed_in" and hop == 2:
                # but at hop 2 (gene->organ) often relevant
                label = 1
            elif rel == "causal_gene" and hop == 1:
                # "causal_gene" at hop 1: relevant
                label = 1
            elif rel == "causal_gene" and hop > 1:
                # but not at deeper hops
                label = 0
            else:
                # Random for other combinations
                label = random.randint(0, 1)
            
            relations.append({
                "r": rel,
                "hop": hop,
                "label": label,
            })
    
    # Remove duplicates (keep same r at same hop)
    seen = set()
    unique_relations = []
    for rel in relations:
        key = (rel["r"], rel["hop"])
        if key not in seen:
            unique_relations.append(rel)
            seen.add(key)
    
    # Generate simple gold path
    gold_paths = [
        [[disease, "causal_gene", gene]],
        [[gene, "participates_in", random.choice(PATHWAYS)]],
    ]
    
    return {
        "qid": qid,
        "query": query_templates.get(intent, f"Query about {disease}"),
        "intent": intent,
        "dataset": dataset_name.split("_")[0],
        "gold_paths": gold_paths,
        "relations": unique_relations,
    }


def generate_dataset(
    dataset_name: str,
    n_queries: int = 50,
) -> List[Dict[str, Any]]:
    """Generate a complete benchmark dataset."""
    data = []
    diseases = DISEASES.get(dataset_name, [])
    
    for qid in range(n_queries):
        disease = random.choice(diseases)
        gene = random.choice(GENES)
        intent = random.choice(INTENTS)
        
        query_item = generate_query_item(qid, dataset_name, disease, gene, intent)
        data.append(query_item)
    
    return data


def main():
    """Generate all benchmark datasets."""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Datasets from paper
    datasets_config = {
        "orphanet_fq274": 50,      # 50 queries for validation
        "disgenet_rd411": 50,
        "omim_hop3": 30,           # Smaller OMIM-Hop3 benchmark
    }
    
    for dataset_name, n_queries in datasets_config.items():
        print(f"\nGenerating {dataset_name}...")
        
        # Create dataset directory
        ds_dir = data_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        data = generate_dataset(dataset_name, n_queries)
        
        # Save
        output_path = ds_dir / f"{dataset_name}_full.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Generated {len(data)} queries")
        print(f"  ✓ Saved to {output_path}")
        
        # Print sample
        if data:
            sample = data[0]
            print(f"\n  Sample (qid=0):")
            print(f"    Query: {sample['query']}")
            print(f"    Intent: {sample['intent']}")
            print(f"    Relations: {len(sample['relations'])} items")
            print(f"    Gold paths: {len(sample['gold_paths'])} paths")


if __name__ == "__main__":
    main()

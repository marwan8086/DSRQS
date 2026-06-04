# =============================================================================
# Data Validation and Correction Script
# Ensures data format matches DSRQS paper specifications
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


def validate_query_item(item: Dict[str, Any]) -> bool:
    """
    Validate that a query item has the required structure.
    
    Required fields:
        - qid: unique query ID (int)
        - query: query text (str)
        - intent: query intent class (str)
        - dataset: dataset name (str)
        - gold_paths: list of gold answer paths (list)
        - relations: list of relations with (r, hop, label) tuples (list)
    """
    required_fields = {"qid", "query", "intent", "dataset", "gold_paths", "relations"}
    
    if not required_fields.issubset(item.keys()):
        return False
    
    # Validate relations structure
    for rel in item["relations"]:
        if not all(k in rel for k in ["r", "hop", "label"]):
            return False
        if not isinstance(rel["hop"], int) or rel["hop"] < 0:
            return False
        if rel["label"] not in [0, 1]:
            return False
    
    return True


def fix_data_format(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load, validate, and fix data format if needed.
    
    Args:
        json_path: Path to JSON data file
    
    Returns:
        Fixed data list
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    fixed_data = []
    errors = []
    
    for i, item in enumerate(data):
        if not validate_query_item(item):
            errors.append(f"Item {i}: Invalid structure")
            continue
        
        # Ensure all required fields are present and correct
        item.setdefault("qid", i)
        item.setdefault("intent", "unknown")
        item.setdefault("dataset", "unknown")
        
        # Validate and normalize relations
        valid_relations = []
        for rel in item.get("relations", []):
            if isinstance(rel.get("label"), bool):
                rel["label"] = int(rel["label"])
            
            if rel.get("hop") is not None and rel.get("r") is not None:
                valid_relations.append({
                    "r": str(rel["r"]),
                    "hop": int(rel["hop"]),
                    "label": int(rel["label"])
                })
        
        item["relations"] = valid_relations
        
        # Ensure gold_paths is a list of lists
        if not isinstance(item.get("gold_paths"), list):
            item["gold_paths"] = []
        
        fixed_data.append(item)
    
    if errors:
        print(f"[WARNING] Found {len(errors)} validation issues:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    return fixed_data


def main():
    """Validate and fix all dataset files."""
    data_dir = Path("data")
    datasets = ["orphanet_fq274", "disgenet_rd411", "omim_hop3"]
    
    for ds_name in datasets:
        ds_path = data_dir / ds_name / f"{ds_name}_full.json"
        
        if not ds_path.exists():
            print(f"[SKIP] {ds_path} not found")
            continue
        
        print(f"\nValidating {ds_name}...")
        
        # Fix data
        fixed_data = fix_data_format(ds_path)
        
        print(f"  ✓ Validated {len(fixed_data)} items")
        
        # Save fixed data back
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved fixed data to {ds_path}")
        
        # Print summary
        if fixed_data:
            print(f"\n  Summary:")
            print(f"    - Total queries: {len(fixed_data)}")
            print(f"    - Avg relations per query: {sum(len(q.get('relations', [])) for q in fixed_data) / len(fixed_data):.1f}")
            print(f"    - Max hops: {max((max((r['hop'] for r in q.get('relations', []) if 'hop' in r), default=0)) for q in fixed_data)}")
            
            # Sample query
            sample = fixed_data[0]
            print(f"\n  Sample query (qid={sample['qid']}):")
            print(f"    Query: {sample['query'][:60]}...")
            print(f"    Intent: {sample.get('intent', 'N/A')}")
            print(f"    Relations: {len(sample.get('relations', []))} items")
            if sample.get('relations'):
                print(f"    First relation: r={sample['relations'][0]['r']}, hop={sample['relations'][0]['hop']}, label={sample['relations'][0]['label']}")


if __name__ == "__main__":
    main()

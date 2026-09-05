"""
Debug script to trace why normalize_labels.py produces so many UNKNOWNs.
Run from cropguard_phase2/ directory after data has been downloaded.

Usage:
    python datasets/debug_normalization.py
"""
import json
import yaml
import sys
from pathlib import Path
from collections import defaultdict

# ── replicate normalize_label exactly as-is ──────────────────────────────────

def load_mapping(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def normalize_label(crop, disease, mapping):
    crop_lower = crop.lower().strip()
    crop_cap = crop_lower.capitalize()
    disease_clean = disease.lower().replace("-", "_").replace(" ", "_").strip(" _")
    if disease_clean == "healthy" or "healthy" in disease_clean:
        return crop_cap, "healthy"
    if crop_lower in mapping:
        crop_mapping = mapping[crop_lower]
        for canonical_disease, info in crop_mapping.items():
            canonical_clean = canonical_disease.lower().replace("-", "_").replace(" ", "_").strip(" _")
            aliases = [canonical_clean]
            if isinstance(info, dict) and 'aliases' in info:
                aliases += [a.lower().replace("-", "_").replace(" ", "_").strip(" _") for a in info['aliases']]
            if disease_clean in aliases:
                return crop_cap, canonical_disease
            # Substring check: alias contained in disease_clean (always safe)
            # or disease_clean contained in alias (only when disease_clean is long
            # enough to not be a generic word like 'leaf', 'rust', etc.)
            for a in aliases:
                if len(a) > 4 and a in disease_clean:
                    return crop_cap, canonical_disease
                if len(disease_clean) > 4 and disease_clean in a:
                    return crop_cap, canonical_disease
    cleaned = disease_clean.replace("leaf_", "").replace("_leaf", "").strip(" _")
    if cleaned and cleaned != "unknown":
        return crop_cap, cleaned
    return crop_cap, "UNKNOWN"

# ─────────────────────────────────────────────────────────────────────────────

def main():
    mapping_path = Path("configs/label_mapping.yaml")
    if not mapping_path.exists():
        print(f"ERROR: {mapping_path} not found. Run from cropguard_phase2/ directory.")
        sys.exit(1)

    mapping = load_mapping(mapping_path)

    # ── 1. Scan actual raw directories ───────────────────────────────────────
    datasets_to_check = {
        "plantvillage": Path("data/metadata/plantvillage.jsonl"),
        "plantdoc":     Path("data/metadata/plantdoc.jsonl"),
    }

    for ds_name, jsonl_path in datasets_to_check.items():
        if not jsonl_path.exists():
            print(f"\n[{ds_name}] metadata not found at {jsonl_path}; skipping.")
            continue

        print(f"\n{'='*60}")
        print(f" Dataset: {ds_name}")
        print(f"{'='*60}")

        unknown_entries = []
        mapped_entries  = defaultdict(list)

        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                raw_crop    = r['crop']
                raw_disease = r['disease']
                orig_label  = r.get('original_label', '')

                norm_crop, norm_disease = normalize_label(raw_crop, raw_disease, mapping)

                if norm_disease == "UNKNOWN":
                    unknown_entries.append({
                        "crop": raw_crop,
                        "disease": raw_disease,
                        "original_label": orig_label,
                    })
                else:
                    mapped_entries[norm_disease].append(raw_disease)

        # Unique UNKNOWN inputs
        unique_unknowns = {}
        for e in unknown_entries:
            key = (e['crop'], e['disease'])
            if key not in unique_unknowns:
                unique_unknowns[key] = e['original_label']

        print(f"\n[UNKNOWN] {len(unknown_entries)} records ({len(unique_unknowns)} unique combos):")
        for (crop, disease), orig in sorted(unique_unknowns.items())[:40]:
            crop_lower = crop.lower().strip()
            in_mapping = crop_lower in mapping
            disease_clean = disease.lower().replace("-","_").replace(" ","_").strip(" _")
            print(f"  crop={repr(crop):<20} disease={repr(disease):<35} orig={repr(orig):<50} "
                  f"crop_in_yaml={in_mapping}  disease_clean={repr(disease_clean)}")

        print(f"\n[MAPPED]  {sum(len(v) for v in mapped_entries.values())} records across {len(mapped_entries)} diseases:")
        for disease, raws in sorted(mapped_entries.items()):
            unique_raws = sorted(set(raws))[:3]
            print(f"  {disease:<35} <- {unique_raws}")

    # ── 2. Quick unit-test of normalize_label with known inputs ──────────────
    print(f"\n{'='*60}")
    print(" normalize_label unit tests")
    print(f"{'='*60}")
    test_cases = [
        # (crop_in, disease_in, expected_disease)
        ("Tomato",       "Early_blight",        "early_blight"),
        ("Tomato",       "early blight",         "early_blight"),
        ("Tomato",       "leaf_early_blight",    "early_blight"),
        ("Tomato",       "Tomato_Early_blight",  "early_blight"),
        ("Pepper",       "Bacterial_spot",       "bacterial_spot"),
        ("Pepper,_bell", "Bacterial_spot",       "bacterial_spot"),
        ("Corn",         "Common_rust_",         "common_rust"),
        ("Corn_(maize)", "Common_rust",          "common_rust"),
        ("Corn",         "Northern_Leaf_Blight", "northern_leaf_blight"),
        ("Apple",        "Apple_scab",           "apple_scab"),
        ("Tomato",       "healthy",              "healthy"),
        # NOTE: 'leaf' alone is handled by prepare_*.py before calling normalize_label.
        # When it reaches here without filtering, the fallback returns 'leaf' (not 'healthy').
        ("Tomato",       "leaf",                 "leaf"),
    ]
    all_pass = True
    for crop, disease, expected in test_cases:
        _, got = normalize_label(crop, disease, mapping)
        status = "PASS" if got == expected else "FAIL"
        if got != expected:
            all_pass = False
        print(f"  [{status}] normalize_label({repr(crop):<20}, {repr(disease):<30}) -> {repr(got)}  (expected {repr(expected)})")

    print()
    if all_pass:
        print("All unit tests PASSED.")
    else:
        print("Some unit tests FAILED -- bugs found in normalize_label or label_mapping.yaml.")

if __name__ == "__main__":
    main()

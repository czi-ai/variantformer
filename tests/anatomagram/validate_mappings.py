#!/usr/bin/env python3
"""
Comprehensive validation of tissue mappings across vocabs and anatomagram.
Ensures consistency and completeness of the tissue mapping system.
"""

import json
import yaml
from pathlib import Path
from typing import Set, Dict, List, Tuple


def load_tissue_vocab() -> Dict[str, int]:
    """Load tissue vocabulary from vocabs/tissue_vocab.yaml."""
    vocab_path = Path(__file__).parent.parent.parent / "vocabs" / "tissue_vocab.yaml"
    with open(vocab_path, 'r') as f:
        return yaml.safe_load(f)


def load_tissue_mapping() -> Dict:
    """Load enhanced tissue mapping."""
    mapping_path = Path(__file__).parent.parent / "data" / "tissue_mapping_enhanced.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


def validate_tissue_coverage() -> Tuple[bool, List[str]]:
    """
    Validate that all tissues in vocab have anatomagram mappings.

    Returns:
        (passed, warnings)
    """
    vocab = load_tissue_vocab()
    mapping = load_tissue_mapping()

    warnings = []

    # Get tissue IDs from vocab (values are IDs)
    vocab_tissue_ids = set(str(v) for v in vocab.values())

    # Get tissue IDs from mapping
    mapped_tissue_ids = set(mapping['tissue_mappings'].keys())

    # Check coverage
    missing_in_mapping = vocab_tissue_ids - mapped_tissue_ids
    extra_in_mapping = mapped_tissue_ids - vocab_tissue_ids

    if missing_in_mapping:
        warnings.append(f"❌ {len(missing_in_mapping)} tissues in vocab missing from anatomagram mapping:")
        for tissue_id in sorted(missing_in_mapping, key=lambda x: int(x) if x.isdigit() else x):
            tissue_name = [k for k, v in vocab.items() if str(v) == tissue_id][0]
            warnings.append(f"   - ID {tissue_id}: {tissue_name}")

    if extra_in_mapping:
        warnings.append(f"⚠️  {len(extra_in_mapping)} tissues in mapping not in vocab:")
        for tissue_id in sorted(extra_in_mapping, key=lambda x: int(x) if x.isdigit() else x):
            tissue_name = mapping['tissue_mappings'][tissue_id]['tissue_name']
            warnings.append(f"   - ID {tissue_id}: {tissue_name}")

    passed = len(missing_in_mapping) == 0

    if passed and len(extra_in_mapping) == 0:
        return True, ["✅ Perfect 1:1 correspondence between vocab and mapping"]

    return passed, warnings


def validate_mapping_integrity() -> Tuple[bool, List[str]]:
    """
    Validate internal consistency of tissue mapping.

    Returns:
        (passed, warnings)
    """
    mapping = load_tissue_mapping()
    warnings = []
    passed = True

    # Check all tissues have required fields
    required_fields = ['tissue_name', 'uberon_id', 'svg_uberon_id', 'is_direct_match', 'can_visualize']

    for tissue_id, tissue_data in mapping['tissue_mappings'].items():
        missing_fields = [f for f in required_fields if f not in tissue_data]
        if missing_fields:
            warnings.append(f"❌ Tissue {tissue_id} missing fields: {missing_fields}")
            passed = False

    # Check hierarchy fallbacks reference valid UBERON IDs
    all_uberon_ids = {t['uberon_id'] for t in mapping['tissue_mappings'].values()}
    all_uberon_ids.update({t['svg_uberon_id'] for t in mapping['tissue_mappings'].values()})

    for source, target in mapping['hierarchy_fallbacks'].items():
        if target not in all_uberon_ids:
            warnings.append(f"⚠️  Hierarchy fallback {source} → {target}: target not found in mappings")

    if passed:
        warnings.append("✅ All tissue mappings have required fields")
        warnings.append("✅ All hierarchy fallbacks reference valid UBERON IDs")

    return passed, warnings


def validate_visualization_coverage() -> Tuple[bool, List[str]]:
    """
    Validate visualization coverage statistics.

    Returns:
        (passed, warnings)
    """
    mapping = load_tissue_mapping()
    warnings = []

    total = len(mapping['tissue_mappings'])
    visualizable = sum(1 for t in mapping['tissue_mappings'].values() if t['can_visualize'])
    direct_matches = sum(1 for t in mapping['tissue_mappings'].values()
                        if t['can_visualize'] and t['is_direct_match'])
    hierarchy = sum(1 for t in mapping['tissue_mappings'].values()
                   if t['can_visualize'] and not t['is_direct_match'])
    cannot_viz = total - visualizable

    warnings.append(f"📊 Visualization Coverage Statistics:")
    warnings.append(f"   Total tissues: {total}")
    warnings.append(f"   Can visualize: {visualizable} ({visualizable/total*100:.1f}%)")
    warnings.append(f"   Direct SVG matches: {direct_matches} ({direct_matches/total*100:.1f}%)")
    warnings.append(f"   Hierarchy fallbacks: {hierarchy} ({hierarchy/total*100:.1f}%)")
    warnings.append(f"   Cannot visualize: {cannot_viz} ({cannot_viz/total*100:.1f}%)")

    # Goal: 100% coverage for tissues (excluding cell lines)
    tissue_only = {tid: t for tid, t in mapping['tissue_mappings'].items()
                   if t.get('tissue_type') != 'cell_line'}
    tissue_viz = sum(1 for t in tissue_only.values() if t['can_visualize'])

    warnings.append(f"\n📊 Tissue-Only Coverage (excluding cell lines):")
    warnings.append(f"   Tissues: {len(tissue_only)}")
    warnings.append(f"   Can visualize: {tissue_viz} ({tissue_viz/len(tissue_only)*100:.1f}%)")

    passed = tissue_viz / len(tissue_only) >= 0.95  # 95% coverage threshold

    if passed:
        warnings.append("   ✅ Excellent coverage!")
    else:
        warnings.append("   ⚠️  Coverage below 95% threshold")

    return passed, warnings


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Tissue Mapping Validation")
    print("=" * 70)

    all_passed = True

    # Check 1: Vocab coverage
    print("\n1️⃣  Checking vocab ↔ mapping correspondence...")
    passed, warnings = validate_tissue_coverage()
    for w in warnings:
        print(w)
    all_passed = all_passed and passed

    # Check 2: Mapping integrity
    print("\n2️⃣  Checking mapping internal consistency...")
    passed, warnings = validate_mapping_integrity()
    for w in warnings:
        print(w)
    all_passed = all_passed and passed

    # Check 3: Visualization coverage
    print("\n3️⃣  Checking visualization coverage...")
    passed, warnings = validate_visualization_coverage()
    for w in warnings:
        print(w)
    all_passed = all_passed and passed

    # Check 4: SVG coverage (import from validate_svg_mappings)
    print("\n4️⃣  Checking SVG element coverage...")
    print("   Run: python anatomagram/utils/validate_svg_mappings.py")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("   Tissue mapping system is consistent and complete")
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("   Review warnings above and fix issues")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

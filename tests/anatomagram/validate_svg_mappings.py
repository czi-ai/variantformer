#!/usr/bin/env python3
"""
Validate SVG UBERON ID coverage in tissue mappings.
"""

import json
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple


def extract_uberon_ids_from_svg(svg_path: Path) -> Set[str]:
    """Extract all UBERON IDs from an SVG file."""
    content = svg_path.read_text()
    # Match both id="UBERON_XXXXXXX" and id="UBERON:XXXXXXX"
    pattern = r'id="(UBERON[_:]?\d+)"'
    matches = re.findall(pattern, content)
    # Normalize to UBERON_XXXXXXX format
    return {m.replace(':', '_') for m in matches if 'UBERON' in m}


def load_tissue_mapping(mapping_path: Path) -> Dict:
    """Load tissue mapping enhanced JSON."""
    with open(mapping_path, 'r') as f:
        return json.load(f)


def get_tissue_info(uberon_id: str, mapping: Dict) -> Dict[str, str]:
    """Get tissue information for a UBERON ID."""
    # Check direct mappings
    for tissue_data in mapping['tissue_mappings'].values():
        if tissue_data.get('svg_uberon_id') == uberon_id:
            return {
                'tissue_name': tissue_data.get('tissue_name', 'Unknown'),
                'display_name': tissue_data.get('display_name', 'Unknown'),
                'match_type': 'direct' if tissue_data.get('is_direct_match') else 'hierarchy'
            }

    # Check if it's a hierarchy fallback target
    for source, target in mapping['hierarchy_fallbacks'].items():
        if target == uberon_id:
            # Found as a target, look up the target's info
            for tissue_data in mapping['tissue_mappings'].values():
                if tissue_data.get('svg_uberon_id') == uberon_id:
                    return {
                        'tissue_name': tissue_data.get('tissue_name', 'Unknown'),
                        'display_name': tissue_data.get('display_name', 'Unknown'),
                        'match_type': 'fallback_target'
                    }

    # Check if it's a hierarchy fallback source
    if uberon_id in mapping['hierarchy_fallbacks']:
        target_id = mapping['hierarchy_fallbacks'][uberon_id]
        # Get the target's info
        for tissue_data in mapping['tissue_mappings'].values():
            if tissue_data.get('svg_uberon_id') == target_id:
                return {
                    'tissue_name': f"→ {tissue_data.get('tissue_name', 'Unknown')}",
                    'display_name': f"→ {tissue_data.get('display_name', 'Unknown')}",
                    'match_type': 'fallback_source'
                }

    return {
        'tissue_name': 'UNMAPPED',
        'display_name': 'UNMAPPED',
        'match_type': 'none'
    }


def validate_svg_coverage(
    svg_path: Path,
    mapping_path: Path,
    svg_type: str
) -> Tuple[Set[str], Set[str], List[str], Dict]:
    """
    Validate that all SVG UBERON IDs have mappings.

    Returns:
        (covered_ids, uncovered_ids, warnings, mapping_details)
    """
    svg_ids = extract_uberon_ids_from_svg(svg_path)
    mapping = load_tissue_mapping(mapping_path)

    # Get all mapped UBERON IDs (both direct and via hierarchy)
    mapped_ids = set()

    # Add direct svg_uberon_id mappings
    for tissue_data in mapping['tissue_mappings'].values():
        if tissue_data.get('svg_uberon_id'):
            mapped_ids.add(tissue_data['svg_uberon_id'])

    # Add hierarchy fallback targets (these are SVG elements that data maps to)
    for fallback_target in mapping['hierarchy_fallbacks'].values():
        mapped_ids.add(fallback_target)

    # Add hierarchy fallback sources (these are SVG elements that have fallback mappings)
    for fallback_source in mapping['hierarchy_fallbacks'].keys():
        mapped_ids.add(fallback_source)

    # Check coverage
    covered = svg_ids & mapped_ids
    uncovered = svg_ids - mapped_ids

    # Build detailed mapping info
    mapping_details = {}
    for uberon_id in svg_ids:
        mapping_details[uberon_id] = get_tissue_info(uberon_id, mapping)

    warnings = []
    if uncovered:
        warnings.append(f"❌ {len(uncovered)} unmapped UBERON IDs in {svg_type} SVG:")
        for uberon_id in sorted(uncovered):
            info = mapping_details[uberon_id]
            warnings.append(f"   - {uberon_id} ({info['tissue_name']})")

    return covered, uncovered, warnings, mapping_details


def main():
    """Run validation for all three anatomagrams."""
    base_dir = Path(__file__).parent.parent
    svg_dir = base_dir / "assets" / "svg"
    mapping_path = base_dir / "data" / "tissue_mapping_enhanced.json"

    print("=" * 90)
    print("SVG UBERON ID Coverage Validation")
    print("=" * 90)

    svgs = [
        ("male", svg_dir / "homo_sapiens.male.svg"),
        ("female", svg_dir / "homo_sapiens.female.svg"),
        ("brain", svg_dir / "homo_sapiens.brain.svg")
    ]

    all_passed = True

    for svg_type, svg_path in svgs:
        if not svg_path.exists():
            print(f"\n⚠️  {svg_type.upper()} SVG not found: {svg_path}")
            continue

        covered, uncovered, warnings, mapping_details = validate_svg_coverage(
            svg_path, mapping_path, svg_type
        )

        print(f"\n{svg_type.upper()} Anatomagram:")
        print(f"  Total SVG elements: {len(covered) + len(uncovered)}")
        print(f"  Mapped: {len(covered)} ({len(covered)/(len(covered)+len(uncovered))*100:.1f}%)")
        print(f"  Unmapped: {len(uncovered)}")

        if warnings:
            print()
            for warning in warnings:
                print(f"  {warning}")
            all_passed = False
        else:
            print("  ✅ All UBERON IDs mapped!")

        # Show sample mappings for verification (first 10)
        if covered and len(covered) > 0:
            print(f"\n  Sample Mappings (first 10 of {len(covered)}):")
            print(f"  {'UBERON ID':<18} | {'Tissue Name':<35} | {'Display Name':<25} | {'Type':<15}")
            print(f"  {'-'*18}-+-{'-'*35}-+-{'-'*25}-+-{'-'*15}")

            for uberon_id in sorted(list(covered))[:10]:
                info = mapping_details[uberon_id]
                tissue_name = info['tissue_name'][:35]
                display_name = info['display_name'][:25]
                match_type = info['match_type']
                print(f"  {uberon_id:<18} | {tissue_name:<35} | {display_name:<25} | {match_type:<15}")

    print("\n" + "=" * 90)
    if all_passed:
        print("✅ VALIDATION PASSED - All SVG elements have mappings")
        print("\nYou can now:")
        print("1. Test with real data: marimo edit notebooks/test_real_vcf2exp_predictions.py")
        print("2. Test female/brain: marimo edit notebooks/test_{female,brain}_anatomagram.py")
    else:
        print("❌ VALIDATION FAILED - Some SVG elements need mappings")
        print("\nNext steps:")
        print("1. Add missing tissues to tissue_mapping_enhanced.json")
        print("2. Add hierarchy fallbacks for unmapped UBERON IDs")
        print("3. Re-run this validation script")
    print("=" * 90)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

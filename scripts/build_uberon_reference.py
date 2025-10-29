#!/usr/bin/env python3
"""
Build comprehensive UBERON reference dictionary from SVG files.

Extracts all UBERON IDs from anatomagram SVG files, queries EBI OLS API
for official labels and descriptions, and generates:
1. uberon_id_map.json - Simple label mappings (backwards compatible)
2. uberon_descriptions.json - Rich metadata with descriptions

Usage:
    python scripts/build_uberon_reference.py
    python scripts/build_uberon_reference.py --dry-run
"""

import json
import time
import sys
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib import request, error
from urllib.parse import quote


class UBERONReferenceBuilder:
    """Builds comprehensive UBERON reference from SVG files and OLS API."""

    API_BASE = "https://www.ebi.ac.uk/ols4/api/search"
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0
    REQUEST_DELAY = 0.5

    def __init__(self, anatomagram_dir: Path, current_map_path: Path):
        """Initialize builder."""
        self.anatomagram_dir = anatomagram_dir
        self.svg_dir = anatomagram_dir / "assets" / "svg"
        self.data_dir = anatomagram_dir / "data"
        self.current_map_path = current_map_path

        # Load current map for comparison
        with open(self.current_map_path, 'r') as f:
            self.current_map = json.load(f)

        # Results storage
        self.svg_uberons = {}  # uberon_id -> list of svg sources
        self.ols_data = {}  # uberon_id -> {label, description, iri, etc}
        self.label_map = {}  # uberon_id -> simplified label
        self.metadata = {}  # uberon_id -> full metadata

    def extract_svg_uberons(self) -> None:
        """Extract all UBERON IDs from SVG files."""
        print("Extracting UBERON IDs from SVG files...")

        for svg_type in ['male', 'female', 'brain']:
            svg_path = self.svg_dir / f"homo_sapiens.{svg_type}.svg"
            if not svg_path.exists():
                print(f"  ⚠️  {svg_type}.svg not found, skipping")
                continue

            result = subprocess.run(
                ['grep', '-o', 'id="UBERON_[0-9]*"', str(svg_path)],
                capture_output=True, text=True
            )

            ids = set(re.findall(r'UBERON_\d+', result.stdout))
            print(f"  {svg_type.capitalize()}: {len(ids)} UBERON IDs")

            for uberon_id in ids:
                if uberon_id not in self.svg_uberons:
                    self.svg_uberons[uberon_id] = []
                self.svg_uberons[uberon_id].append(svg_type)

        print(f"\nTotal unique UBERON IDs: {len(self.svg_uberons)}")

    def query_ols_api(self, uberon_id: str) -> Optional[Dict]:
        """Query EBI OLS API for UBERON term details."""
        url = f"{self.API_BASE}?q={quote(uberon_id)}&ontology=uberon&exact=true"

        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                with request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    docs = data.get('response', {}).get('docs', [])

                    if not docs:
                        return None

                    doc = docs[0]
                    return {
                        'label': doc.get('label', ''),
                        'description': doc.get('description', [''])[0] if doc.get('description') else '',
                        'iri': doc.get('iri', ''),
                        'obo_id': doc.get('obo_id', ''),
                        'short_form': doc.get('short_form', '')
                    }

            except Exception as e:
                if attempt < self.RETRY_ATTEMPTS - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))
                else:
                    print(f"      ERROR querying {uberon_id}: {e}")
                    return None

        return None

    def fetch_ols_data(self) -> None:
        """Fetch OLS data for all UBERON IDs."""
        print(f"\nQuerying EBI OLS API for {len(self.svg_uberons)} UBERON IDs...")
        print("This will take a few minutes with rate limiting...\n")

        total = len(self.svg_uberons)
        for i, uberon_id in enumerate(sorted(self.svg_uberons.keys()), 1):
            print(f"  [{i}/{total}] {uberon_id}...", end=' ')

            ols_data = self.query_ols_api(uberon_id)

            if ols_data:
                self.ols_data[uberon_id] = ols_data
                print(f"✅ \"{ols_data['label']}\"")
            else:
                print(f"❌ NOT FOUND")

            # Rate limiting
            if i < total:
                time.sleep(self.REQUEST_DELAY)

    def decide_label(self, uberon_id: str, ols_label: str) -> str:
        """
        Decide which label to use: current simplified or OLS official.

        Keep simplified if:
        - Current label exists and refers to same organ (just simpler wording)

        Use OLS label if:
        - No current label exists (new from brain SVG)
        - Current label is wrong (different organ)
        """
        if uberon_id in self.current_map:
            current = self.current_map[uberon_id]
            current_lower = current.lower()
            ols_lower = ols_label.lower()

            # Check if they're semantically equivalent (same organ, different wording)
            # Simple heuristic: one contains the other, or very similar
            if current_lower in ols_lower or ols_lower in current_lower:
                # Same organ, keep simplified version
                return current
            elif current_lower.replace(' gland', '') == ols_lower.replace(' gland', ''):
                # "Thyroid" vs "thyroid gland" - keep simplified
                return current
            elif current_lower.replace('the ', '') == ols_lower.replace('the ', ''):
                # Handle "the" differences
                return current
            else:
                # Potentially different organs - use OLS label and flag for review
                print(f"\n      ⚠️  Label mismatch for {uberon_id}:")
                print(f"          Current: \"{current}\"")
                print(f"          OLS: \"{ols_label}\"")
                print(f"          Using OLS label (verify if correct)")
                return ols_label.title()  # Capitalize for consistency
        else:
            # New ID from brain SVG - use OLS label
            return ols_label.title()

    def build_references(self) -> None:
        """Build label map and metadata dictionaries."""
        print("\nBuilding reference dictionaries...")

        for uberon_id, svg_sources in sorted(self.svg_uberons.items()):
            if uberon_id in self.ols_data:
                ols_info = self.ols_data[uberon_id]

                # Decide on label
                label = self.decide_label(uberon_id, ols_info['label'])
                self.label_map[uberon_id] = label

                # Build metadata
                self.metadata[uberon_id] = {
                    'label': label,
                    'ols_label': ols_info['label'],
                    'description': ols_info['description'],
                    'iri': ols_info['iri'],
                    'obo_id': ols_info['obo_id'],
                    'svg_sources': svg_sources,
                    'simplified': label.lower() != ols_info['label'].lower()
                }
            else:
                # No OLS data found - keep current or mark as unknown
                label = self.current_map.get(uberon_id, f"UNKNOWN_{uberon_id}")
                self.label_map[uberon_id] = label

                self.metadata[uberon_id] = {
                    'label': label,
                    'ols_label': '',
                    'description': 'OLS data not found',
                    'iri': '',
                    'obo_id': '',
                    'svg_sources': svg_sources,
                    'simplified': False,
                    'error': 'NOT_FOUND_IN_OLS'
                }

        print(f"  Generated {len(self.label_map)} label mappings")
        print(f"  Generated {len(self.metadata)} metadata entries")

    def generate_report(self) -> str:
        """Generate summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append("UBERON Reference Build Report")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"SVG Files Processed: male, female, brain")
        lines.append(f"Total UBERON IDs: {len(self.svg_uberons)}")
        lines.append(f"Successfully queried OLS: {len(self.ols_data)}")
        lines.append(f"Not found in OLS: {len(self.svg_uberons) - len(self.ols_data)}")
        lines.append("")

        # New IDs from brain SVG
        new_ids = [uid for uid in self.svg_uberons if uid not in self.current_map]
        if new_ids:
            lines.append(f"New IDs from brain SVG ({len(new_ids)}):")
            for uid in sorted(new_ids):
                label = self.label_map.get(uid, 'UNKNOWN')
                lines.append(f"  {uid}: \"{label}\"")
            lines.append("")

        # Removed IDs (in old map but not in SVGs)
        removed_ids = [uid for uid in self.current_map if uid not in self.svg_uberons]
        if removed_ids:
            lines.append(f"Removed IDs (not in any SVG) ({len(removed_ids)}):")
            for uid in sorted(removed_ids):
                lines.append(f"  {uid}: \"{self.current_map[uid]}\"")
            lines.append("")

        # Label changes
        label_changes = []
        for uid, new_label in self.label_map.items():
            if uid in self.current_map and self.current_map[uid] != new_label:
                label_changes.append((uid, self.current_map[uid], new_label))

        if label_changes:
            lines.append(f"Label Changes ({len(label_changes)}):")
            for uid, old, new in sorted(label_changes):
                lines.append(f"  {uid}:")
                lines.append(f"    OLD: \"{old}\"")
                lines.append(f"    NEW: \"{new}\"")
            lines.append("")

        lines.append("Files to be generated:")
        lines.append("  - anatomagram/data/uberon_id_map.json (label mappings)")
        lines.append("  - anatomagram/data/uberon_descriptions.json (full metadata)")

        return "\n".join(lines)

    def save_files(self, dry_run: bool = False) -> None:
        """Save generated reference files."""
        if dry_run:
            print("\n🔍 DRY RUN - Files not written")
            return

        print("\nSaving reference files...")

        # Save label map (backwards compatible)
        label_map_path = self.data_dir / "uberon_id_map.json"
        with open(label_map_path, 'w') as f:
            json.dump(self.label_map, f, indent=2, sort_keys=True)
        print(f"  ✅ Saved: {label_map_path}")

        # Save metadata (new file)
        metadata_path = self.data_dir / "uberon_descriptions.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)
        print(f"  ✅ Saved: {metadata_path}")

    def build(self, dry_run: bool = False) -> None:
        """Run full build process."""
        self.extract_svg_uberons()
        self.fetch_ols_data()
        self.build_references()

        print("\n" + self.generate_report())

        self.save_files(dry_run)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build comprehensive UBERON reference from SVG files"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving files (preview only)'
    )

    args = parser.parse_args()

    # Paths
    anatomagram_dir = Path('anatomagram')
    current_map = Path('anatomagram/data/uberon_id_map.json')

    if not anatomagram_dir.exists():
        print(f"❌ Error: anatomagram directory not found")
        sys.exit(1)

    if not current_map.exists():
        print(f"❌ Error: Current uberon_id_map.json not found")
        sys.exit(1)

    # Build reference
    builder = UBERONReferenceBuilder(anatomagram_dir, current_map)
    builder.build(dry_run=args.dry_run)

    if not args.dry_run:
        print("\n✅ UBERON reference build complete!")
        print("\nNext steps:")
        print("  1. Review changes in anatomagram/data/uberon_id_map.json")
        print("  2. Verify anatomagram/data/uberon_descriptions.json")
        print("  3. Run: python scripts/verify_uberon_mappings.py")
        print("  4. Test tooltips in anatomagram notebooks")


if __name__ == '__main__':
    main()

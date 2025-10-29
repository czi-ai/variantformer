#!/usr/bin/env python3
"""
Verify UBERON ID mappings against the EBI Ontology Lookup Service (OLS) API.

This script validates that all UBERON IDs in anatomagram/data/uberon_id_map.json
are correctly mapped to their official anatomical structure names according to
the UBERON ontology.

Usage:
    python scripts/verify_uberon_mappings.py
    python scripts/verify_uberon_mappings.py --output report.md
"""

import json
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
from urllib import request, error
from urllib.parse import quote


class UBERONVerifier:
    """Verifies UBERON ID mappings against the EBI OLS API."""

    API_BASE = "https://www.ebi.ac.uk/ols4/api/search"
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0  # seconds
    REQUEST_DELAY = 0.5  # seconds between requests to avoid rate limiting

    def __init__(self, mapping_file: Path):
        """Initialize verifier with path to UBERON mapping file."""
        self.mapping_file = mapping_file
        self.mappings = self._load_mappings()
        self.results = {
            'correct': [],
            'mismatches': [],
            'not_found': [],
            'errors': []
        }

    def _load_mappings(self) -> Dict[str, str]:
        """Load UBERON mappings from JSON file."""
        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    def _query_ols_api(self, uberon_id: str) -> Tuple[bool, str, str]:
        """
        Query EBI OLS API for a UBERON term.

        Returns:
            Tuple of (success, official_label, error_message)
        """
        url = f"{self.API_BASE}?q={quote(uberon_id)}&ontology=uberon&exact=true"

        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                with request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    docs = data.get('response', {}).get('docs', [])

                    if not docs:
                        return (False, '', 'NOT_FOUND')

                    # Get the label from the first matching document
                    label = docs[0].get('label', '')
                    return (True, label, '')

            except error.HTTPError as e:
                if e.code == 404:
                    return (False, '', 'NOT_FOUND')
                elif attempt < self.RETRY_ATTEMPTS - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))  # exponential backoff
                else:
                    return (False, '', f'HTTP_ERROR_{e.code}')

            except Exception as e:
                if attempt < self.RETRY_ATTEMPTS - 1:
                    time.sleep(self.RETRY_DELAY * (2 ** attempt))
                else:
                    return (False, '', f'ERROR: {str(e)}')

        return (False, '', 'MAX_RETRIES_EXCEEDED')

    def _normalize_name(self, name: str) -> str:
        """Normalize anatomical structure name for comparison."""
        return name.lower().strip()

    def verify_all(self) -> None:
        """Verify all UBERON mappings against OLS API."""
        total = len(self.mappings)
        print(f"Verifying {total} UBERON ID mappings...")
        print(f"Using API: {self.API_BASE}\n")

        for i, (uberon_id, our_name) in enumerate(self.mappings.items(), 1):
            print(f"[{i}/{total}] Checking {uberon_id}: \"{our_name}\"...", end=' ')

            success, ols_label, error_msg = self._query_ols_api(uberon_id)

            if not success:
                if error_msg == 'NOT_FOUND':
                    print(f"❌ NOT FOUND in OLS")
                    self.results['not_found'].append((uberon_id, our_name))
                else:
                    print(f"⚠️  ERROR: {error_msg}")
                    self.results['errors'].append((uberon_id, our_name, error_msg))
            else:
                # Compare normalized names
                if self._normalize_name(our_name) == self._normalize_name(ols_label):
                    print(f"✅ CORRECT (OLS: \"{ols_label}\")")
                    self.results['correct'].append((uberon_id, our_name, ols_label))
                else:
                    print(f"⚠️  MISMATCH (OLS: \"{ols_label}\")")
                    self.results['mismatches'].append((uberon_id, our_name, ols_label))

            # Rate limiting: wait between requests
            if i < total:
                time.sleep(self.REQUEST_DELAY)

    def generate_report(self, output_file: Path = None) -> str:
        """Generate verification report."""
        total = len(self.mappings)
        correct_count = len(self.results['correct'])
        mismatch_count = len(self.results['mismatches'])
        not_found_count = len(self.results['not_found'])
        error_count = len(self.results['errors'])

        accuracy = (correct_count / total * 100) if total > 0 else 0

        report = []
        report.append("# UBERON ID Mapping Verification Report")
        report.append("=" * 70)
        report.append("")
        report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Mapping File**: {self.mapping_file}")
        report.append(f"**Total Mappings**: {total}")
        report.append(f"**Accuracy**: {accuracy:.1f}%")
        report.append("")
        report.append("## Summary")
        report.append(f"- ✅ Correct: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
        report.append(f"- ⚠️  Mismatches: {mismatch_count}/{total} ({mismatch_count/total*100:.1f}%)")
        report.append(f"- ❌ Not Found: {not_found_count}/{total} ({not_found_count/total*100:.1f}%)")
        report.append(f"- 🔴 Errors: {error_count}/{total} ({error_count/total*100:.1f}%)")
        report.append("")

        # Correct mappings
        if self.results['correct']:
            report.append(f"## ✅ Correct Mappings ({correct_count})")
            report.append("")
            for uberon_id, our_name, ols_label in sorted(self.results['correct']):
                report.append(f"- **{uberon_id}**: \"{our_name}\" ✓ (OLS: \"{ols_label}\")")
            report.append("")

        # Mismatches
        if self.results['mismatches']:
            report.append(f"## ⚠️  Mismatched Mappings ({mismatch_count})")
            report.append("")
            report.append("These mappings differ from the official UBERON ontology labels:")
            report.append("")
            for uberon_id, our_name, ols_label in sorted(self.results['mismatches']):
                report.append(f"- **{uberon_id}**:")
                report.append(f"  - Our mapping: \"{our_name}\"")
                report.append(f"  - OLS official: \"{ols_label}\"")
                report.append(f"  - **Suggestion**: Update to \"{ols_label}\"")
                report.append("")

        # Not found
        if self.results['not_found']:
            report.append(f"## ❌ Not Found in OLS ({not_found_count})")
            report.append("")
            report.append("These UBERON IDs were not found in the OLS database:")
            report.append("")
            for uberon_id, our_name in sorted(self.results['not_found']):
                report.append(f"- **{uberon_id}**: \"{our_name}\"")
                report.append(f"  - **Action**: Verify ID is correct or remove if obsolete")
            report.append("")

        # Errors
        if self.results['errors']:
            report.append(f"## 🔴 API Errors ({error_count})")
            report.append("")
            report.append("These mappings could not be verified due to API errors:")
            report.append("")
            for uberon_id, our_name, error_msg in sorted(self.results['errors']):
                report.append(f"- **{uberon_id}**: \"{our_name}\"")
                report.append(f"  - Error: {error_msg}")
            report.append("")

        # Critical fixes verification
        report.append("## 🔍 Recently Fixed Mappings")
        report.append("")
        report.append("Verification of critical mappings that were recently corrected:")
        report.append("")

        critical_ids = {
            'UBERON_0002048': ('Lung', 'Fixed from incorrect "Heart" label'),
            'UBERON_0000948': ('Heart', 'Added missing Heart mapping'),
            'UBERON_0002106': ('Spleen', 'Removed duplicate "Lung" entry')
        }

        for uberon_id, (expected, description) in critical_ids.items():
            status = "✅ VERIFIED"
            for uid, our_name, ols_label in self.results['correct']:
                if uid == uberon_id:
                    report.append(f"- **{uberon_id}**: \"{our_name}\" = \"{ols_label}\" {status}")
                    report.append(f"  - {description}")
                    break
            else:
                # Check if it's in mismatches or not found
                for uid, our_name, ols_label in self.results['mismatches']:
                    if uid == uberon_id:
                        report.append(f"- **{uberon_id}**: \"{our_name}\" vs \"{ols_label}\" ⚠️  MISMATCH")
                        report.append(f"  - {description}")
                        break
                else:
                    report.append(f"- **{uberon_id}**: ❌ NOT VERIFIED")
                    report.append(f"  - {description}")

        report.append("")
        report.append("---")
        report.append("")
        report.append("**Note**: Minor differences in capitalization or formatting are considered matches.")
        report.append("For example, \"Heart\" matches \"heart\", \"Cerebral Cortex\" matches \"cerebral cortex\".")

        report_text = "\n".join(report)

        # Save to file if specified
        if output_file:
            output_file.write_text(report_text)
            print(f"\n✅ Report saved to: {output_file}")

        return report_text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify UBERON ID mappings against EBI OLS API"
    )
    parser.add_argument(
        '--mapping-file',
        type=Path,
        default=Path('anatomagram/data/uberon_id_map.json'),
        help='Path to UBERON mapping JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for verification report (markdown)'
    )

    args = parser.parse_args()

    # Check if mapping file exists
    if not args.mapping_file.exists():
        print(f"❌ Error: Mapping file not found: {args.mapping_file}")
        sys.exit(1)

    # Run verification
    verifier = UBERONVerifier(args.mapping_file)
    verifier.verify_all()

    # Generate and print report
    print("\n" + "=" * 70)
    print(verifier.generate_report(args.output))

    # Exit with error code if there are mismatches or errors
    if verifier.results['mismatches'] or verifier.results['not_found'] or verifier.results['errors']:
        sys.exit(1)
    else:
        print("\n✅ All UBERON mappings verified successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()

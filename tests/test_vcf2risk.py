#!/usr/bin/env python3
"""
Comprehensive test script for VCF2Risk workflow.

Tests the ADriskFromVCF class with the correct API usage:
- One-row-per-tissue query format (parallel gene_ids and tissue_ids lists)
- Validates output format for anatomagram visualization
- Measures performance

Run on cluster after pulling latest code to validate the workflow.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from processors import ad_risk


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def test_model_initialization():
    """Test 1: Model initialization with AWS profile."""
    print_section("TEST 1: Model Initialization")

    start = time.time()
    adrisk = ad_risk.ADriskFromVCF()
    elapsed = time.time() - start

    print(f"✅ ADriskFromVCF initialized in {elapsed:.2f}s")
    print(f"   - Model loaded: {adrisk.model is not None}")
    print(f"   - Tissue map: {len(adrisk.tissue_map)} tissues")
    print(f"   - Gene map: {len(adrisk.genes_map)} genes")
    print(f"   - AD predictors: ManifestLookup initialized")

    return adrisk


def test_single_tissue_prediction(adrisk, vcf_path):
    """Test 2: Single gene × single tissue prediction."""
    print_section("TEST 2: Single Gene × Single Tissue")

    gene_id = "ENSG00000000457.13"
    tissue_id = 7

    print(f"Gene: {gene_id}")
    print(f"Tissue ID: {tissue_id} ({adrisk.tissue_map.loc[tissue_id, 'tissue']})")

    start = time.time()
    predictions_df = adrisk(vcf_path, [gene_id], [tissue_id])
    elapsed = time.time() - start

    print(f"\n✅ Prediction completed in {elapsed:.2f}s")
    print(f"   - Output shape: {predictions_df.shape}")
    print(f"   - Columns: {list(predictions_df.columns)}")

    assert len(predictions_df) == 1, f"Expected 1 row, got {len(predictions_df)}"
    assert 'ad_risk' in predictions_df.columns, "Missing 'ad_risk' column"
    assert 'tissue_name' in predictions_df.columns, "Missing 'tissue_name' column"
    assert 'gene_name' in predictions_df.columns, "Missing 'gene_name' column"

    print(f"\n   Result:")
    print(f"   - Gene: {predictions_df.iloc[0]['gene_name']}")
    print(f"   - Tissue: {predictions_df.iloc[0]['tissue_name']}")
    print(f"   - AD Risk: {predictions_df.iloc[0]['ad_risk']:.6f}")

    return predictions_df


def test_multi_tissue_prediction(adrisk, vcf_path):
    """Test 3: Single gene × multiple tissues (main use case)."""
    print_section("TEST 3: Single Gene × Multiple Tissues")

    gene_id = "ENSG00000000457.13"
    tissue_ids = [7, 14, 15, 16, 17]  # 5 tissues

    print(f"Gene: {gene_id}")
    print(f"Tissues: {len(tissue_ids)} tissues")
    for tid in tissue_ids:
        print(f"   - {tid}: {adrisk.tissue_map.loc[tid, 'tissue']}")

    # Correct API: parallel lists (same gene repeated for each tissue)
    gene_ids = [gene_id] * len(tissue_ids)

    print(f"\nAPI call: adrisk(vcf_path, gene_ids={len(gene_ids)} items, tissue_ids={len(tissue_ids)} items)")

    start = time.time()
    predictions_df = adrisk(vcf_path, gene_ids, tissue_ids)
    elapsed = time.time() - start

    print(f"\n✅ Prediction completed in {elapsed:.2f}s")
    print(f"   - Output shape: {predictions_df.shape}")
    print(f"   - Expected rows: {len(tissue_ids)}")

    assert len(predictions_df) == len(tissue_ids), \
        f"Expected {len(tissue_ids)} rows, got {len(predictions_df)}"
    assert 'ad_risk' in predictions_df.columns, "Missing 'ad_risk' column"

    print(f"\n   Results Summary:")
    print(f"   - Mean AD risk: {predictions_df['ad_risk'].mean():.6f}")
    print(f"   - Std AD risk: {predictions_df['ad_risk'].std():.6f}")
    print(f"   - Min AD risk: {predictions_df['ad_risk'].min():.6f}")
    print(f"   - Max AD risk: {predictions_df['ad_risk'].max():.6f}")

    print(f"\n   Top 3 Risk Tissues:")
    top3 = predictions_df.nlargest(3, 'ad_risk')[['tissue_name', 'ad_risk']]
    for idx, row in top3.iterrows():
        print(f"   - {row['tissue_name']}: {row['ad_risk']:.6f}")

    return predictions_df


def validate_anatomagram_format(predictions_df):
    """Test 4: Validate output format for anatomagram visualization."""
    print_section("TEST 4: Anatomagram Data Format Validation")

    required_columns = ['gene_id', 'gene_name', 'tissue_id', 'tissue_name', 'ad_risk']

    print(f"Checking required columns...")
    for col in required_columns:
        present = col in predictions_df.columns
        status = "✅" if present else "❌"
        print(f"   {status} {col}")
        if not present:
            raise ValueError(f"Missing required column: {col}")

    print(f"\nData types:")
    for col in required_columns:
        dtype = predictions_df[col].dtype
        print(f"   - {col}: {dtype}")

    print(f"\nSample row:")
    sample = predictions_df.iloc[0]
    for col in required_columns:
        print(f"   - {col}: {sample[col]}")

    # Check for NaN values
    nan_counts = predictions_df[required_columns].isna().sum()
    if nan_counts.any():
        print(f"\n⚠️  Warning: Found NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"   - {col}: {count} NaN values")
    else:
        print(f"\n✅ No NaN values in required columns")

    print(f"\n✅ Format validated - ready for anatomagram visualization")


def main():
    """Run all tests."""
    print("\n" + "█" * 80)
    print(" VCF2RISK COMPREHENSIVE TEST SUITE")
    print("█" * 80)

    # Configuration
    vcf_path = '/mnt/czi-sci-ai/intrinsic-variation-gene-ex-2/project_gene_regulation/dna2cell_training/v2_pcg_flash2/sample_vcf/HG00096.vcf.gz'

    print(f"\nConfiguration:")
    print(f"   - VCF file: {vcf_path}")
    print(f"   - Sample: HG00096")

    try:
        # Test 1: Initialization
        adrisk = test_model_initialization()

        # Test 2: Single tissue
        single_result = test_single_tissue_prediction(adrisk, vcf_path)

        # Test 3: Multiple tissues (main use case)
        multi_result = test_multi_tissue_prediction(adrisk, vcf_path)

        # Test 4: Validate format
        validate_anatomagram_format(multi_result)

        # Summary
        print_section("TEST SUMMARY")
        print("✅ All tests passed successfully!")
        print(f"\nKey takeaways:")
        print(f"   1. ADriskFromVCF initializes correctly with AWS profile")
        print(f"   2. Single gene × single tissue works")
        print(f"   3. Single gene × multiple tissues works (notebook use case)")
        print(f"   4. Output format is correct for anatomagram visualization")
        print(f"\nAPI Usage for Notebook:")
        print(f"   predictions_df = adrisk(vcf_path, [gene_id] * n_tissues, tissue_ids_list)")
        print()

        return 0

    except Exception as e:
        print_section("TEST FAILED")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

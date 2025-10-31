# Essential imports
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path.cwd().parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from processors.variantprocessor import VariantProcessor

# Configure plotting
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)



# Initialize VariantFormer
print("🚀 Initializing VariantFormer Variant Processor...")
vep = VariantProcessor()

print("✅ System initialized!")
print(f"🌍 Populations: {', '.join(vep.populations)}")
print(f"🧬 Tissues: {len(vep.tissue_vocab)} available")


# Example VCF integration parameters
vcf_path = os.path.join(str(Path.cwd()),"_artifacts/HG00096.vcf.gz")
sample_name = "HG00096"  # European sample from 1000 Genomes Project

print(f"📁 VCF File: {vcf_path}")
print(f"👤 Sample: {sample_name}")

# Same variant but now with individual genotype context
vcf_variant_data = {
    "chr": ["chr13"],
    "pos": [113978728],
    "ref": ["A"],
    "alt": ["G"],
    "tissue": ["whole blood"],
    "gene_id": ["ENSG00000185989.10"],
}

vcf_variant_df = pd.DataFrame(vcf_variant_data)
print("\n🧬 Analyzing the same variant with individual genotype data:")
print(vcf_variant_df.to_string(index=False))


# Run VariantFormer with VCF integration
print("\n🔬 Running VariantFormer analysis with VCF integration...")
print("⏳ Processing individual genotype data...")

vcf_predictions = vep.predict(
    var_df=vcf_variant_df,
    output_dir="/tmp/vep_output_vcf",
    vcf_path=vcf_path,
    sample_name=sample_name,
)

print("✅ VCF-based predictions completed!")
print(f"📊 VCF predictions shape: {vcf_predictions.shape}")
print("\n🔍 Key differences from population-based analysis:")
print("   • Individual genotype information included")
print("   • Sample-specific vs population-average analysis")
print("   • More precise zygosity determination")


# Format scores for VCF-based predictions
vcf_formatted_scores = vep.format_scores(vcf_predictions)

print("📊 VCF-based score formatting completed!")
print(f"📋 Includes sample-specific genotype: {sample_name}")
print(
    f"🧬 Available populations in results: {vcf_formatted_scores.columns[vcf_formatted_scores.columns.str.contains('-exp')].tolist()}"
)

# Display VCF-based formatted results
print("📋 VCF-based Formatted Results:")
print("=" * 30)

expression_cols_vcf = [col for col in vcf_formatted_scores.columns if "-exp" in col]
print("🎯 Expression values by population/sample:")

for col in expression_cols_vcf:
    if pd.notna(vcf_formatted_scores[col].iloc[0]):
        value = vcf_formatted_scores[col].iloc[0]
        label = col.replace("-exp", "").replace("SAMPLE", f"Sample {sample_name}")
        print(f"   {label:20}: {value:.4f}")

# Show comparison with population analysis
print(f"\n📊 Sample {sample_name} vs Population Analysis:")
print(vcf_formatted_scores.head())

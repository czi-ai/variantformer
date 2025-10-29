# VCF2Expression Data Generation Scripts

This directory contains scripts for generating VCF2Expression prediction data, both mock and real predictions.

## Scripts

### 1. `generate_mock_vcf2exp_data.py` 

**Purpose**: Generate realistic simulated expression data quickly for immediate use.

**Usage**:
```bash
# Basic usage (generates mock data for all genes)
python scripts/generate_mock_vcf2exp_data.py

# Custom output directory
python scripts/generate_mock_vcf2exp_data.py --output-dir /tmp/mock_predictions

# Generate smaller dataset for testing
python scripts/generate_mock_vcf2exp_data.py --max-genes 1000

# Custom random seed for reproducibility
python scripts/generate_mock_vcf2exp_data.py --seed 123
```

**Features**:
- ⚡ **Fast generation** (minutes, not hours)
- 🎭 **Realistic expression values** using biologically plausible distributions
- 📊 **Multiple dataset sizes** (full, 10k, 1k, 100, 10, 1 gene)
- 🚚 **SSH transfer commands** automatically generated
- 🔄 **Reproducible** with seed parameter

---

### 2. `generate_vcf2exp_predictions.py`

**Purpose**: Generate real model predictions using gene-level batching for tractable computation.

**Usage**:
```bash
# Basic usage (all genes, 100 per batch)
python scripts/generate_vcf2exp_predictions.py

# Custom batch size
python scripts/generate_vcf2exp_predictions.py --batch-size 50

# Resume interrupted run
python scripts/generate_vcf2exp_predictions.py --resume

# Test with smaller gene set
python scripts/generate_vcf2exp_predictions.py --max-genes 500

# Only combine existing batch results (skip prediction)
python scripts/generate_vcf2exp_predictions.py --combine-only

# Custom VCF file and output directory
python scripts/generate_vcf2exp_predictions.py \
  --vcf-path /path/to/your/file.vcf.gz \
  --output-dir /tmp/real_predictions
```

**Features**:
- 🔄 **Gene-level batching** (100 genes per batch = ~30 min per batch)
- 💾 **Resume capability** (automatically skips completed batches)
- 📊 **Progress tracking** with ETA estimates
- 🛡️ **Intermediate saves** (prevents data loss on interruption)
- 📈 **Scalable** (~185 batches for all genes vs 90-day single job)

---

## Workflow Recommendations

### For Immediate Use (Meeting/Demo)
```bash
# Generate mock data quickly
python scripts/generate_mock_vcf2exp_data.py --output-dir /tmp/mock_data

# Transfer to local machine
scp cluster:/tmp/mock_data/*.parquet ./
```

### For Production Data
```bash
# Start real predictions (long-running)
nohup python scripts/generate_vcf2exp_predictions.py \
  --output-dir /tmp/real_predictions \
  --batch-size 100 > prediction.log 2>&1 &

# Check progress
tail -f prediction.log

# Resume if interrupted
python scripts/generate_vcf2exp_predictions.py \
  --output-dir /tmp/real_predictions \
  --resume
```

## Output Structure

Both scripts generate parquet files with identical structure:

```python
{
  'gene_id': str,           # e.g., "ENSG00000000419.12"
  'gene_name': str,         # e.g., "DPM1"  
  'tissues': List[str],     # List of 63 tissue/cell line names
  'predicted_expression': List[float]  # Expression values for each tissue
}
```

## Dataset Sizes

Both scripts create multiple dataset sizes:
- `vcf2exp_sample_pred_full.parquet`: All genes (~18,439)
- `vcf2exp_sample_pred_10k.parquet`: 10,000 genes
- `vcf2exp_sample_pred_1k.parquet`: 1,000 genes
- `vcf2exp_sample_pred_100.parquet`: 100 genes
- `vcf2exp_sample_pred_10.parquet`: 10 genes
- `vcf2exp_sample_pred_1.parquet`: 1 gene

## Time Estimates

| Script | Dataset Size | Estimated Time |
|--------|--------------|----------------|
| Mock | All genes | 2-5 minutes |
| Real | 100 genes (1 batch) | 30 minutes |
| Real | All genes (185 batches) | 90-100 hours |

## Requirements

- Python 3.7+
- DNA2Cell environment activated
- Access to cluster filesystem paths
- GPU available (for real predictions)

## Troubleshooting

### Mock Data Issues
- **Import errors**: Ensure DNA2Cell environment is activated
- **Permission errors**: Check write permissions to output directory

### Real Prediction Issues  
- **Model loading fails**: Verify GPU access and model checkpoint paths
- **Memory errors**: Reduce batch size (`--batch-size 50`)
- **Interrupted runs**: Use `--resume` flag to continue
- **Batch combination fails**: Use `--combine-only` to retry combination step
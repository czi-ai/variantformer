# Running Marimo Notebooks

This directory contains interactive Marimo notebooks for VariantFormer analysis. Marimo provides reactive execution and modern UI components for genomic data exploration.

## Available Notebooks

### vcf2exp.py - Expression Prediction
Predicts tissue-specific gene expression from VCF files using VariantFormer's 1.2B parameter model.

**Analysis workflow:**
- Load VCF file with genetic variants
- Select gene and tissues for analysis
- Run VariantFormer inference (expression prediction across 63 tissues)
- Visualize results on interactive anatomograms
- Explore pre-computed genome-wide expression heatmaps

**Runtime:** ~3-4 minutes for one gene across all tissues (H100 GPU)

### vcf2risk.py - AD Risk Prediction
Predicts Alzheimer's disease risk from genetic variants using VariantFormer embeddings and tissue-specific risk models.

**Analysis workflow:**
- Load VCF file with genetic variants
- Select gene and tissues with AD risk predictors
- Run VariantFormer + AD risk pipeline (45 tissues with trained models)
- Visualize tissue-specific risk scores on anatomograms
- Identify high-risk tissues and potential disease mechanisms

**Runtime:** ~3-4 minutes for one gene across 45 tissues (H100 GPU)

## Prerequisites

See the main [README](../README.md#setup) for:
- Hardware requirements (H100 GPU, 40GB+ VRAM)
- Software dependencies (PyTorch, CUDA, flash-attention)
- Installation instructions (install_local.sh or Docker)
- Model checkpoint downloads (download_artifacts.py)

## Installing Marimo

After running `install.sh` in the parent directory, ensure Marimo dependencies are installed:

```bash
source .venv/bin/activate
uv pip install -e .[marimo] --no-build-isolation
```

This installs:
- marimo (notebook framework)
- anywidget (anatomogram visualization)
- plotly (interactive plots)
- Additional visualization dependencies

## Running Notebooks

Activate the virtual environment and launch Marimo:

```bash
source .venv/bin/activate
marimo edit notebooks/vcf2exp.py
```

Or for AD risk analysis:

```bash
marimo edit notebooks/vcf2risk.py
```

Marimo opens a web interface in your browser. The notebook executes reactively - cells automatically re-run when dependencies change.

## Data Requirements

### VCF Files
- **Format:** Standard VCF v4.2+ (bgzipped or uncompressed)
- **Reference:** GRCh38/hg38 (required - must match training data)
- **Sample VCF:** Downloaded to `_artifacts/HG00096.vcf.gz` by `download_artifacts.py`

### Model Checkpoints
Downloaded automatically by `download_artifacts.py` to `_artifacts/`:
- `v4_pcg_epoch11_checkpoint.pth` (14GB) - VariantFormer model
- Gene-specific AD risk predictors (downloaded on-demand from S3)

See main [README](../README.md#downloading-artifacts) for download instructions.

## Notebook Features

### Interactive Components
- **Searchable gene selection:** Type to search 18,000+ genes
- **Tissue multiselect:** Select specific tissues or organ systems
- **VCF file browser:** Browse and select custom VCF files
- **Reactive execution:** Changes automatically trigger re-analysis
- **Anatomogram visualization:** Three anatomical views (male, female, brain)
- **Interactive heatmaps:** Pan/zoom genome-wide expression patterns

### Visualization Controls
- Color palette selection (viridis, plasma, inferno, etc.)
- Scale type (linear, log)
- Aggregation strategy (mean, max, min, weighted_mean)
- Tissue filtering and drill-down

## Troubleshooting

### Marimo not picking up virtual environment
Ensure marimo is installed **inside** the virtual environment:
```bash
source .venv/bin/activate
which marimo  # Should point to .venv/bin/marimo
marimo --version
```

If marimo is installed globally, reinstall in venv:
```bash
uv pip install -e .[marimo] --no-build-isolation
```

### Verbose logging from markdown extensions
The notebooks suppress debug logging by default. If you see excessive logs, check that the first cell contains:
```python
logging.getLogger('MARKDOWN').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
```

### Slow execution with large VCF files
The sample VCF (_artifacts/HG00096.vcf.gz) is a full whole-genome file (7.2GB). For faster tutorial execution:
- Use the file browser to select a smaller VCF subset
- Or update DEFAULT_VCF_PATH to point to a sample file
- Processing time scales with VCF size

### Analysis hangs or times out
Ensure GPU is available and CUDA is configured:
```bash
nvidia-smi  # Check GPU status
python -c "import torch; print(torch.cuda.is_available())"
```

## Marimo vs Jupyter

### Why Marimo?

**Reactive execution:**
- Cells automatically re-run when inputs change
- No manual "run all" or execution order management
- Dependency graph ensures correctness

**Modern UI components:**
- Searchable dropdowns for 18,000+ gene database
- File browsers for VCF selection
- Interactive anatomogram widgets
- Real-time visualization updates

**Reproducibility:**
- Notebooks are pure Python scripts (.py files)
- Version control friendly (standard git diff)
- No hidden state - execution is deterministic

### Jupyter Notebooks Still Available

Classic Jupyter notebooks (`.ipynb` files) are also provided:
- `vcf2exp.ipynb`
- `vcf2risk.ipynb`
- `variant2exp.ipynb`
- `variant2risk.ipynb`

See main [README](../README.md#running-notebooks) for Jupyter setup with Docker.

## Model Details

VariantFormer architecture and training methodology are described in:

**Citation:**
Ghosal, S., et al. (2025). VariantFormer: A hierarchical transformer integrating DNA sequences with genetic variation and regulatory landscapes for personalized gene expression prediction. *bioRxiv* 2025.10.31.685862. [DOI: 10.1101/2025.10.31.685862](https://doi.org/10.1101/2025.10.31.685862)

**Training data:** GTEx v8 paired whole-genome sequencing and RNA-seq data (21,004 samples from 2,330 donors)

## Additional Resources

- [VariantFormer GitHub](https://github.com/czi-ai/variantformer)
- [GTEx Portal](https://gtexportal.org/)
- [Main README](../README.md) - Installation and setup
- [Responsible Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy)

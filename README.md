
# VariantFormer
![Model Overview](figs/model_overview.png)


## Model Description

VariantFormer is a 1.2-billion-parameter hierarchical transformer model that predicts tissue-specific gene expression from personalized diploid genomes. Unlike traditional reference-based models, VariantFormer directly incorporates individual genetic variants to generate tissue-conditioned, person-specific expression predictions across the genome.

## Citation
VariantFormer: A hierarchical transformer integrating DNA sequences with genetic variations and regulatory landscapes for personalized gene expression prediction. Sayan Ghosal, Youssef Barhomi, Tejaswini Ganapathi, Amy Krystosik, Lakshmi Krishnan,Sashidhar Guntury, Donghui Li, Alzheimer’s Disease Neuroimaging Initiative, Francesco Paolo Casale and Theofanis Karaletsos. 2025 bioRxiv. DOI: https://doi.org/10.1101/2025.10.31.685862

## Key Features

- **Personalized genomic modeling:** Encodes individual genetic variants using IUPAC ambiguity codes, enabling native representation of heterozygous and homozygous genotypes
- **Long-range regulatory context:** Processes megabase-scale cis-regulatory windows to capture enhancer-promoter interactions and distal regulatory elements
- **Tissue-specific predictions:** Employs hierarchical cross-attention to model regulatory influence in a tissue-conditioned manner across 54 tissues and 7 cell lines
- **Comprehensive gene coverage:** Excels at predicting both protein-coding genes and non-coding RNAs (lncRNAs, regulatory RNAs), which comprise over 60% of annotated genes

## Training Data

VariantFormer was trained on 21,004 bulk RNA-seq samples from 2,330 donors with paired whole-genome sequencing data, sourced from:

- GTEx (Genotype-Tissue Expression)
- 1000 Genomes Project
- ENCODE
- ADNI (Alzheimer's Disease Neuroimaging Initiative)
- MAGE

This represents the largest curated collection of paired whole-genome sequencing and bulk RNA-seq data to date.

## Intended Use

- Predict tissue-specific gene expression from individual genomes
- Score and prioritize genetic variants for gene expression change
- Generate tissue specific gene embeddings for disease risk prediction
- Enable variant impact on gene expression across diverse ancestries
- Facilitate in silico genetic perturbation analysis


# Setup

## Requirements

This is the hardware this model has been tested on:
- GPU: H100 or L40 gpu (this code has been tested on only these 2, other gpus may work)
- Cuda: 12
- OS: Ubuntu 24.04


## Quickstart 
### Step 1: get the source and install dependencies
This step takes less than 5min

```sh
# get this package's source code, use gh (https://cli.github.com/) for convenience
gh repo clone czi-ai/variantformer
cd variantformer
./install.sh
source .venv/bin/activate
```

### Step 2: install flash attention

```sh
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Note: the flash attention above works with torch 2.8, cuda 12, python 3.12 and ABI=True. If you have other dependency constraints, you may find the corresponding flash attention wheel [here](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3). You can also install flash attention from scratch while following instructions [here](https://github.com/Dao-AILab/flash-attention)

### Step 3: fetch data assets
The model uses many data artifacts, like model checkpoints, gene and cre coordinates, example whole genome sequences representing different ancestries. All artifacts get downloaded into the `./_artifacts` directory in the package's root. No special credentials are needed to download all artifacts. This should take less than 5min on a 1Gb connection. It is about 43GB of data. 

```sh
./download_artifacts.py
```

### Step 4: run the model
To get started, you may run your first jupyter [notebook](https://github.com/czi-ai/variantformer/blob/main/notebooks/vcf2exp.ipynb)
```sh
source .venv/bin/activate
jupyter ./notebooks/vcf2exp.ipynb
```
for more details, see the section below about notebooks.


Note: if you want to install this package using a docker container, please see the next section


### (Optional) Step 5: run the unit tests 
```
pytest 
```

## Troubleshooting
### flash attention undefined symbol
```
ImportError .venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEab
``` 
If you have installed flash_attn from the wheel above which works with torch=2.8 and you may have a different torch version. Ensure you install the version in the flash attention wheel that maps to your torch version. The more robust way of dealing with this dependency issue is to install flash attention from scratch with `uv pip install flash_attn` but it may take a long time (hours) to install. 

Cause: This error indicates a version mismatch between the installed FlashAttention and PyTorch binaries.

Resolution: Ensure that the FlashAttention wheel version exactly matches your installed PyTorch and CUDA versions.
Alternatively, build FlashAttention from source for maximum compatibility:
```uv pip install flash_attn```
(Note: building from source can take an hour or more, depending on system resources.)


## Installing package with Docker

A `Dockerfile` is provided in this repo. It is used to create a self-contained
"container" with all dependencies for VariantFormer installed. You will need
[Docker](https://www.docker.com/) or a compatible tool like [Podman](https://podman.io/)
 set up on your system.

To build the container:

```sh
make build
# or
docker build -t variantformer .
```

**Note:** due to their size, the model checkpoints and other artifacts are not baked
into the container. They must be mounted as a volume when running the container (for
example, via `-v .:/app` or `-v ./_artifacts:/app/_artifacts/`).


# Notebooks

## 📓 Available Notebooks

| Notebook | Description |
|----------|-------------|
| [`vcf2exp.ipynb`](notebooks/vcf2exp.ipynb) | Predict tissue-specific gene expression from VCF files containing genetic variants |
| [`variant2exp.ipynb`](notebooks/variant2exp.ipynb) | Analyze variant impact on gene expression across populations and tissues |
| [`vcf2risk.ipynb`](notebooks/vcf2risk.ipynb) | Estimate Alzheimer's Disease risk from genetic variants in VCF files |
| [`variant2risk.ipynb`](notebooks/variant2risk.ipynb) | Perform in silico mutagenesis to assess variant effects on AD risk |
| [`eqtl_analysis.ipynb`](notebooks/eqtl_analysis.ipynb) | Benchmark VariantFormer against baseline models for eQTL prediction |

**Note:** All expression prediction notebooks (`vcf2exp`, `variant2exp`) generate gene-specific embeddings for each gene-tissue pair that can be used for downstream tasks such as disease risk prediction, variant prioritization, or custom machine learning models. 

## Running Notebooks
If you are not using a docker container, you may run the notebooks like your regular jupyter notebooks:
```sh
jupyter notebooks/<NOTEBOOK_FILE>
```

If you are using docker, launch a containerized Jupyter notebook instance:

```sh
make notebook
```

Once the notebook server is running, it will print a URL out to the console.
This can be opened in a web browser to access the notebooks.

If you are running the notebook on a remote server, you may need to allow access
to the notebook server's port and use the remote host's IP address to connect to
the notebook, rather than the IP address Jupyter prints to the console.


# Tests

Tests can be run using the `make test` command. This requires the Docker container.

Some expected values in the tests were calculated using H100 GPUs. Running the tests
on other GPUs may fail due to differences in floating point implementations.


# Shell

The `make shell` command will open a shell inside the Docker container.


# Runtime Data Access

The model downloads data for specific genes and tissues on-demand
from AWS S3 using the data loaders defined in `utils/assets.py`. No special
credentials are required.


# Code of Conduct / Contribution Guidelines

[See `CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)

# Responsible Use
We are committed to advancing the responsible development and use of artificial intelligence. Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when engaging with the model. The model is intended to be used for research purposes only and was not designed for clinical, diagnostic, or therapeutic purposes. You can find more information about the model and its intended uses here [link model card].

# Reporting Security Issues

[See `SECURITY.md`](./SECURITY.md)


# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
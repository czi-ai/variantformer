# VariantFormer

# Setup

## Downloading Artifacts

Before running VariantFormer, you will need to download the model
checkpoints and other artifacts. The `download_artifacts.py` script handles
downloading these. No special credentials are needed to download them.

This script has a dependency on `boto3`. You will either need to `pip install boto3`,
possibly into a virtual environment, or use a tool like [uv](https://docs.astral.sh/uv/)
that understands PEP723 inline metadata.

To download artifacts:

```sh
python download_artifacts.py
# or
uv run download_artifacts.py
# or
make download
```


## Docker Option

A `Dockerfile` is provided in this repo. It is used to create a self-contained
"container" with all dependencies for VariantFormer installed. You will need
[Docker](https://www.docker.com/) or a compatrible tool like [Podman](https://podman.io/)
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


## Direct Install Option

Alternatively, VariantFormer can be installed directly onto your system. This
is difficult to automate due to dependencies between VariantFormer, PyTorch,
Flash-Attention, and CUDA.

The `install_local.sh` script will attempt to install all necessary dependencies,
and has been tested on Ubuntu 24.04, but you may need to modify it for your
environment.


### Note on Flash Attention

VariantFormer uses [flash-attention](https://github.com/Dao-AILab/flash-attention).
Wheels for this are not available on PyPI, so you will either need to manually
obtain one from their [Github release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1),
or compile from source.

This dependency may require special handling to install since PyTorch is used
when parsing its `setup.py` file.

If you are using a recent version of `uv` that supports [extra-build-dependencies](https://docs.astral.sh/uv/reference/settings/#extra-build-dependencies),
then it should handle everything in a single step:

```sh
uv pip install -e ".[notebook]"
```

If you are using an older version of `uv`, you will need to first install PyTorch,
then install VariantFormer (and with it flash-attention) with [build isolation](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation)
turned off:

```sh
uv pip install torch
uv pip install -e ".[notebook]" --no-build-isolation
```

If you are using vanilla `pip`, you will also need to install PyTorch first:

```sh
pip install torch psutil
pip install -e ".[notebook]" --no-build-isolaton
```


# Notebooks

##  Available Notebooks

- `notebooks/vcf2exp.ipynb`: VCF to gene expression prediction demo


## Running Notebooks

The Docker container option is the most straightforward way to run
the example notebooks.

To launch a containerized Jupyter notebook instance:

```sh
make notebook
```

One the notebook server is running, it will print a URL out to the console.
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

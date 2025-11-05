#!/usr/bin/env -S bash -eu

# This script should install everything needed for VariantFormer to run the example notebooks.
# It may need to be run using `sudo` on your system.
# It's intended to run on Ubuntu (tested on 24.04) with CUDA drivers already installed.
# 
# If you don't want to install all these packages, the separate Dockerfile can be used to
# create a self-contained image for running VariantFormer.

sudo apt-get update && sudo apt-get install -y \
    bedtools \
    zlib1g \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libncurses5-dev \
    build-essential \
    autotools-dev \
    autoconf

# Install htslib 1.21 from source
wget -O htslib-1.21.tar.bz2 https://github.com/samtools/htslib/releases/download/1.21/htslib-1.21.tar.bz2
tar -xjf htslib-1.21.tar.bz2
cd htslib-1.21
make
sudo make install
cd ..
rm -rf htslib-1.21 htslib-1.21.tar.bz2

# Install samtools 1.21 from source
wget -O samtools-1.21.tar.bz2 https://github.com/samtools/samtools/releases/download/1.21/samtools-1.21.tar.bz2
tar -xjf samtools-1.21.tar.bz2
cd samtools-1.21
make
sudo make install
cd ..
rm -rf samtools-1.21 samtools-1.21.tar.bz2

# Install bcftools 1.21 from source
wget -O bcftools-1.21.tar.bz2 https://github.com/samtools/bcftools/releases/download/1.21/bcftools-1.21.tar.bz2
tar -xjf bcftools-1.21.tar.bz2
cd bcftools-1.21
make
sudo make install
cd ..
rm -rf bcftools-1.21 bcftools-1.21.tar.bz2

# Install uv to manage Python installation
curl -LsSf https://astral.sh/uv/install.sh | sh

# setup python virtual environment
export UV_PROJECT_ENVIRONMENT=.venv
uv venv $UV_PROJECT_ENVIRONMENT --python=3.12
source $UV_PROJECT_ENVIRONMENT/bin/activate

# install all dependencies
uv pip install -e .[notebook,test,marimo]

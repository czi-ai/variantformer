import pybedtools
import yaml
import pandas as pd
import numpy as np
import torch
from utils.constants import IGNORE_CHRS
import os
import scipy.stats as stats
import time


def precision2dtype(precision_str: str) -> torch.dtype:
    """
    Convert a Lightning precision string (e.g., "16-mixed", "bf16-mixed", "32", "32-true")
    into the corresponding torch.dtype.
    """
    # Normalize the string by lowering and stripping spaces
    precision_str = precision_str.lower().strip()

    # If "bf16" is present, return bfloat16
    if "bf16" in precision_str:
        return torch.bfloat16

    # If "16" is present (and not bf16), return float16
    if "16" in precision_str:
        return torch.float16

    # If "32" is present, return float32
    if "32" in precision_str:
        return torch.float32

    raise ValueError(f"Unknown precision string: {precision_str}")


class Gene(object):
    def __init__(self, gene_id, donor, tissue, TPM, FPKM):
        self.gene_id = gene_id
        self.donor = donor
        self.tissue = tissue
        self.TPM = TPM
        self.FPKM = FPKM
        self.log1pTPM = np.log1p(TPM)
        self.log1pFPKM = np.log1p(FPKM)


class AllRnaSeq(object):
    def __init__(self):
        self.rna_seq = {}

    def add_gene(self, gene_obj):
        gene_id = gene_obj.gene_id
        donor = gene_obj.donor
        tissue = gene_obj.tissue
        self.rna_seq[(gene_id, donor, tissue)] = gene_obj


def gpu_call():
    print("GPU call")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    model = torch.nn.Linear(3, 3).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for i in range(2):
        y_hat = model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("Done")
    return


def load_config(fname):
    """
    Load a YAML configuration file
    """
    with open(fname) as file:
        config = yaml.safe_load(file)
    return config


def merge_across_dfs(dfs, on=["chrom", "start", "end"]):
    """Merge dataframes across the same columns"""
    # the input is a list of data frames with the same columns. The output is a single data frame with all the rows and merged columns
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = pd.merge(df, dfs[i], on=on)
    return df


def load_bed_regions(bed_file):
    """Load regions from BED file."""
    print(f"Loading BED regions from {bed_file}")
    bed = pybedtools.BedTool(bed_file)
    return bed


def load_bed_regions_as_df(bed_file):
    """Load regions from BED file."""
    print(f"Loading BED regions from {bed_file}")
    bed = pd.read_csv(bed_file, sep="\t", header=None)
    return bed


def split_across_chromosomes(bed_file):
    """Split BED file across chromosomes."""
    bed = load_bed_regions(bed_file)
    bed_files = []
    for chrom in bed.chromsizes:
        chrom_bed = bed.filter(lambda x: x.chrom == chrom)
        bed_files.append(chrom_bed)
    return bed_files


def split_dataframe_across_chromosomes(df):
    """Split dataframe across chromosomes."""
    bed_files = []
    chrs = []
    for chrom in df["chrom"].unique():
        if chrom in IGNORE_CHRS:
            continue
        chrom_bed = df[df["chrom"] == chrom]
        chrom_bed = chrom_bed.reset_index(drop=True)
        bed_files.append(chrom_bed)
        chrs.append(chrom)
    return bed_files, chrs


def reverse_complement(sequence):
    "Reverse complement a DNA sequence"
    complement = {
        "A": "T",
        "a": "t",
        "C": "G",
        "c": "g",
        "G": "C",
        "g": "c",
        "T": "A",
        "t": "a",
        "R": "Y",
        "r": "y",
        "Y": "R",
        "y": "r",
        "S": "S",
        "s": "s",
        "W": "W",
        "w": "w",
        "K": "M",
        "k": "m",
        "M": "K",
        "m": "k",
        "B": "V",
        "b": "v",
        "D": "H",
        "d": "h",
        "H": "D",
        "h": "d",
        "V": "B",
        "v": "b",
        "N": "N",
        "n": "n",
        "-": "-",  # gap character
        ".": ".",  # placeholder
    }

    # Reverse the sequence
    rev_seq = sequence[::-1]

    # Build the reverse complement
    rev_comp_seq = "".join([complement.get(base, base) for base in rev_seq])

    return rev_comp_seq


def count_parameters(model):
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
            print(f"{n}: {p.numel()}")
    print(f"Total params: {total}")


def merge_pop_stat(df, af_path):
    chrs = df["chr"].unique()
    allele_merged_dfs = []
    for chr in chrs:
        af_file = os.path.join(af_path, f"1KG_hg38_af_{chr}.tsv")
        af_df = pd.read_csv(af_file, sep="\t")
        chr_df = df[df["chr"] == chr].copy()
        merged_df = chr_df.merge(
            af_df,
            left_on=["chr", "pos", "ref", "alt"],
            right_on=["chr", "pos", "ref", "alt"],
            how="left",
        ).reset_index(drop=True)
        allele_merged_dfs.append(merged_df)
    all_merged_dfs = pd.concat(allele_merged_dfs, ignore_index=True)
    print(all_merged_dfs)
    all_merged_dfs["AF_EUR"] = all_merged_dfs["AF_EUR"].replace(".", np.nan).astype(float)
    all_merged_dfs["AF_AFR"] = all_merged_dfs["AF_AFR"].replace(".", np.nan).astype(float)
    all_merged_dfs["AF_EAS"] = all_merged_dfs["AF_EAS"].replace(".", np.nan).astype(float)
    all_merged_dfs["AF_SAS"] = all_merged_dfs["AF_SAS"].replace(".", np.nan).astype(float)
    all_merged_dfs["AF_AMR"] = all_merged_dfs["AF_AMR"].replace(".", np.nan).astype(float)
    return all_merged_dfs


def gene_pop_agg_score(df, score_cols, score_type="log2fc"):
    pop_agg_scores = []
    # remove D2C-REF_HG38-2-Poisson or D2C-REF_HG38-2-exp-log2fc from score_cols if it exists
    if f"D2C-REF_HG38-2-exp-{score_type}" in score_cols:
        score_cols = [col for col in score_cols if "REF_HG38-2" not in col]
    pop_af_cols = [
        "AF_" + c.split("-")[1]
        for c in score_cols
        if c.startswith("D2C-AFR-2")
        or c.startswith("D2C-AMR-2")
        or c.startswith("D2C-EAS-2")
        or c.startswith("D2C-EUR-2")
        or c.startswith("D2C-SAS-2")
    ]
    for i, row in df.iterrows():
        pop_scores = row[score_cols].values.astype(float)
        pop_af = row[pop_af_cols].values.astype(float)

        # Find indices where scores are not NaN
        valid_indices = ~np.isnan(pop_scores)

        if np.sum(valid_indices) > 0:
            # Filter out NaN values from both scores and allele frequencies
            valid_scores = pop_scores[valid_indices]
            valid_af = pop_af[valid_indices]
            valid_af = valid_af / sum(valid_af)

            # Calculate weighted average using allele frequencies as weights
            # Handle case where all AFs are zero
            if np.sum(valid_af) > 0:
                weighted_score = np.average(valid_scores, weights=valid_af)
            else:
                # If all AFs are zero, use simple mean
                weighted_score = np.mean(valid_scores)
        else:
            # If all scores are NaN
            weighted_score = np.nan

        pop_agg_scores.append(weighted_score)
    df["D2C-agg-" + score_type + "-weighted"] = pop_agg_scores

    return df


def generate_log2fc_score(df, af_path):
    ref_col = ["REF_HG38-0-exp"]
    pop_columns = [
        col
        for col in df.columns
        if col.startswith("AFR-2")
        or col.startswith("AMR-2")
        or col.startswith("EAS-2")
        or col.startswith("EUR-2")
        or col.startswith("SAS-2")
        or col.startswith("REF_HG38-2")
        or col.startswith("SAMPLE-2")
    ]
    df = df[
        ref_col
        + pop_columns
        + ["variant_id", "genes", "tissues", "ref", "alt", "chr", "pos"]
    ]
    df = df.reset_index(drop=True)
    score_cols = []
    sample_cols = [col for col in pop_columns if col.startswith("SAMPLE-2")]
    for i, col in enumerate(pop_columns):
        pop = np.array(df[col]).flatten()
        ref = np.array(df[ref_col]).flatten()
        ep = 1e-10
        score = np.log2((pop + ep) / (ref + ep))
        df["D2C-" + col + "-log2fc"] = score
        score_cols.append("D2C-" + col + "-log2fc")
    df[score_cols] = df[score_cols].astype(float)
    if len(sample_cols) == 0:
        df = gene_pop_agg_score(
            merge_pop_stat(df, af_path), score_cols, score_type="log2fc"
        )
        df = df[
            [
                "variant_id",
                "genes",
                "tissues",
                "ref",
                "alt",
                "chr",
                "pos",
                "D2C-agg-log2fc-weighted",
            ]
            + score_cols
        ]
    else:
        df = df[
            ["variant_id", "genes", "tissues", "ref", "alt", "chr", "pos"] + score_cols
        ]
    return df


def generate_poisson_score(df, af_path):
    ref_col = ["REF_HG38-0-exp"]
    pop_columns = [
        col
        for col in df.columns
        if col.startswith("AFR-2")
        or col.startswith("AMR-2")
        or col.startswith("EAS-2")
        or col.startswith("EUR-2")
        or col.startswith("SAS-2")
        or col.startswith("REF_HG38-2")
        or col.startswith("SAMPLE-2")
    ]
    df = df[
        ref_col
        + pop_columns
        + ["variant_id", "genes", "tissues", "ref", "alt", "chr", "pos"]
    ]
    df = df.reset_index(drop=True)
    score_cols = []
    sample_cols = [col for col in pop_columns if col.startswith("SAMPLE-2")]
    for i, col in enumerate(pop_columns):
        pop = np.array(df[col]).flatten()
        ref = np.array(df[ref_col]).flatten()

        score = stats.poisson.cdf(pop, ref)
        df["D2C-" + col + "-Poisson"] = score
        score_cols.append("D2C-" + col + "-Poisson")
    df[score_cols] = df[score_cols].astype(float)
    if len(sample_cols) == 0:
        df = gene_pop_agg_score(
            merge_pop_stat(df, af_path), score_cols, score_type="Poisson"
        )
        df = df[
            [
                "variant_id",
                "genes",
                "tissues",
                "ref",
                "alt",
                "chr",
                "pos",
                "D2C-agg-Poisson-weighted",
            ]
            + score_cols
        ]
    else:
        df = df[
            ["variant_id", "genes", "tissues", "ref", "alt", "chr", "pos"] + score_cols
        ]
    return df


def multi_try_load_csv(file_path):
    """Load a CSV file, retrying with an exponential backoff

    Args:
        file_path: Path to the CSV file

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file
    """
    start = time.time()
    flag = False
    delay = 0.1
    while True:
        try:
            data = pd.read_csv(file_path)
            flag = True
            break
        except Exception:
            print(f"Error reading file: {file_path}")
            time.sleep(delay)
            delay *= 2
            flag = False
        if time.time() - start > 600:
            print("Error reading file. Exiting...")
            break
    assert flag, f"Error reading file: {file_path}"
    return data


def multi_try_load_pickle(file_path):
    """Load a pickle file, retrying with an exponential backoff

    Args:
        file_path: Path to the pickle file

    Returns:
        Any: The data from the pickle file
    """
    start = time.time()
    flag = False
    delay = 0.1
    while True:
        try:
            data = pd.read_pickle(file_path)
            flag = True
            break
        except Exception:
            print(f"Error reading file: {file_path}")
            time.sleep(delay)
            delay *= 2
            flag = False
        if time.time() - start > 600:
            print("Error reading file. Exiting...")
            break
    assert flag, f"Error reading file: {file_path}"
    return data


def multi_try_load_npz(file_path):
    """Load a npz file, retrying with an exponential backoff

    Args:
        file_path: Path to the npz file

    Returns:
        np.ndarray: A numpy array containing the data from the npz file
    """
    start = time.time()
    flag = False
    delay = 0.1
    while True:
        try:
            data = np.load(file_path)
            flag = True
            break
        except Exception:
            print(f"Error reading file: {file_path}")
            time.sleep(delay)
            delay *= 2
            flag = False
        if time.time() - start > 600:
            print("Error reading file. Exiting...")
            break
    assert flag, f"Error reading file: {file_path}"
    return data

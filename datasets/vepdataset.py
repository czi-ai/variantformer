import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from utils.seq import BPEEncoder
from utils.constants import MAP_REF_CRE_TO_IDX, SPECIAL_TOKENS
import copy
from utils.data_process import ExtractSeqFromBed
from utils.functions import multi_try_load_csv
from utils.assets import GeneManifestLookup, GeneSequencesManifestLookup, CreSequencesManifestLookup


@dataclass
class Variant:
    """Represents a genetic variant"""

    chrom: str
    pos: int
    ref: str
    alt: str
    tissue: str
    gene_id: List[str]
    consequence: Optional[str] = None
    label: Optional[int] = None

    def __post_init__(self):
        if not self.chrom.startswith("chr"):
            self.chrom = "chr" + self.chrom


def collate_fn(batch):
    assert len(batch) == 1, "Batch size must be 1 for VEPDataset collate function"
    return batch[0]


class SequenceProcessor:
    """Handles DNA sequence manipulation and IUPAC encoding"""

    COMPLEMENT_MAP = {
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
        "-": "-",
        ".": ".",
    }

    IUPAC_CODES = {
        "AA": "A",
        "AC": "M",
        "CA": "M",
        "AG": "R",
        "GA": "R",
        "AT": "W",
        "TA": "W",
        "CC": "C",
        "CG": "S",
        "GC": "S",
        "CT": "Y",
        "TC": "Y",
        "GG": "G",
        "GT": "K",
        "TG": "K",
        "TT": "T",
    }

    @classmethod
    def reverse_complement(cls, sequence: str) -> str:
        """Generate reverse complement of DNA sequence"""
        rev_seq = sequence[::-1]
        return "".join([cls.COMPLEMENT_MAP.get(base, base) for base in rev_seq])

    @classmethod
    def get_iupac_code(cls, ref: str, alt: str) -> str:
        """Get IUPAC ambiguity code for heterozygous variant"""
        combo = ref + alt
        return cls.IUPAC_CODES.get(combo, "N")

    @classmethod
    def apply_variant(
        cls, sequence: str, variant: Variant, pos_on_seq: int
    ) -> Tuple[str, str]:
        """Apply variant to sequence, returning heterozygous and homozygous versions"""
        forward_seq = sequence.split(",")[0]

        # Heterozygous: use IUPAC ambiguity code
        iupac_code = cls.get_iupac_code(variant.ref, variant.alt)
        forward_seq_het = (
            forward_seq[:pos_on_seq] + iupac_code + forward_seq[pos_on_seq + 1 :]
        )

        # Homozygous: replace with alternate allele
        forward_seq_hom = (
            forward_seq[:pos_on_seq] + variant.alt + forward_seq[pos_on_seq + 1 :]
        )

        # Generate reverse complements
        reverse_seq_het = cls.reverse_complement(forward_seq_het)
        reverse_seq_hom = cls.reverse_complement(forward_seq_hom)

        return (
            forward_seq_het + "," + reverse_seq_het,
            forward_seq_hom + "," + reverse_seq_hom,
        )


class VEPDataset:
    """Variant Effect Prediction Dataset for generating model-ready batches"""

    def __init__(
        self,
        bpe_encoder: BPEEncoder,
        gene_cre_manifest: GeneManifestLookup,
        gene_seq_manifest: GeneSequencesManifestLookup,
        cre_seq_manifest: CreSequencesManifestLookup,
        *,
        max_length: int = 200,
        context_window: int = 200,
        cre_neighbour_hood: int = 50,
        gene_upstream_neighbour_hood: int = 1000,
        gene_downstream_neighbour_hood: int = 300000,
        gene_variant_pairs: List[Dict[str, Any]] = None,
        data_vocab: Dict[str, Any] = None,
        fasta_path: str = None,
    ):
        self.bpe = bpe_encoder
        self.max_length = max_length
        self.context_window = context_window
        self.ref_cre_to_idx = MAP_REF_CRE_TO_IDX
        self.vocab = bpe_encoder.tokenizer.get_vocab()
        self.pad_token_id = self.vocab.get(SPECIAL_TOKENS["pad_token"])
        self.sequence_processor = SequenceProcessor()
        self.gene_cre_manifest = gene_cre_manifest
        self.gene_seq_manifest = gene_seq_manifest
        self.cre_seq_manifest = cre_seq_manifest
        self.cre_neighbour_hood = cre_neighbour_hood
        self.gene_upstream_neighbour_hood = gene_upstream_neighbour_hood
        self.gene_downstream_neighbour_hood = gene_downstream_neighbour_hood
        self.gene_variant_pairs = gene_variant_pairs
        self.data_vocab = data_vocab
        self.fasta_path = fasta_path

    def _map_files(self, cre_file, gene_id):
        """Map the files

        Args:
            i: Index of the file to map
            cre_file: CRE file path
            gene_id: Gene ID

        Returns:
            tuple: A tuple containing:
                - gene_cres_df: CRE dataframe
                - gene_id: Gene ID
        """
        gene_cre_map_path = self.gene_cre_manifest.get_file_path(gene_id)
        gene_df = pd.read_csv(gene_cre_map_path)
        # columns: chromosome	start	end	gene_id	gene_name	strand	start_cre	end_cre	cre_id	score	strand_cre	att1	att2	cre_color	cre_name	embedding	start_gene	end_gene
        gene_df["start_cre"] = gene_df["start_cre"] - self.cre_neighbour_hood
        gene_df["end_cre"] = gene_df["end_cre"] + self.cre_neighbour_hood
        df = pd.read_pickle(cre_file)
        df = df.rename(columns={"start": "start_cre", "end": "end_cre"})

        if (
            df["start_cre"].is_monotonic_increasing
            and gene_df["start_cre"].is_monotonic_increasing
        ):
            first_cre = gene_df.iloc[0]["start_cre"]
            last_cre = gene_df.iloc[-1]["end_cre"]
            start_index = df["start_cre"].searchsorted(first_cre, side="left")
            end_index = df["end_cre"].searchsorted(last_cre, side="right") - 1
            gene_cres_df = df.iloc[start_index : end_index + 1].reset_index(drop=True)
            gene_cres_df["strand"] = [gene_df["strand"].iloc[0]] * len(gene_cres_df)
            gene_cres_df["cre_name"] = gene_cres_df["cCRE"]

        else:
            gene_cres_df = pd.merge(
                gene_df,
                df,
                left_on=["chromosome", "start_cre", "end_cre"],
                right_on=["chrom", "start_cre", "end_cre"],
            )

        if gene_df.iloc[0]["strand"] == "-":
            gene_cres_df = gene_cres_df.iloc[::-1]

        gene_cres_df = gene_cres_df.reset_index(drop=True)
        D = {}
        for item in gene_cres_df.columns:
            if item.endswith("_sequence"):
                D[item] = "sequence"
            if item.endswith("_encoded_seq"):
                D[item] = "encoded_seq"
            if item.endswith("cre_name"):
                D[item] = "cCRE"

        gene_cres_df = gene_cres_df.rename(columns=D)
        gene_cres_df = gene_cres_df[
            ["strand", "cCRE", "sequence", "encoded_seq", "start_cre", "end_cre"]
        ].copy()
        to_drop_f = [
            it
            for it in range(len(gene_cres_df["encoded_seq"]))
            if len(gene_cres_df["encoded_seq"].iloc[it][0]) == 0
        ]
        to_drop_r = [
            it
            for it in range(len(gene_cres_df["encoded_seq"]))
            if len(gene_cres_df["encoded_seq"].iloc[it][1]) == 0
        ]
        to_drop = list(set(to_drop_f + to_drop_r))
        if len(to_drop) > 0:
            gene_cres_df = gene_cres_df.drop(index=to_drop)
        gene_cres_df = gene_cres_df.reset_index(drop=True)

        return gene_cres_df

    def load_gene_data(
        self,
        gene: Dict[str, Any],
        chromosome: str,
        sample_name: str,
        population: str,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Load gene CRE and sequence data"""
        # Load cre sequence data
        cre_seq_path = self.cre_seq_manifest.get_file_path(chromosome, population)
        cre_df = self._map_files(cre_seq_path, gene["gene_id"])

        # Load gene sequence data
        gene_seq_path = self.gene_seq_manifest.get_file_path(gene['gene_id'], population)
        gene_data = np.load(gene_seq_path, allow_pickle=True)

        gene_dict = {key: str(gene_data[key]) for key in gene_data.files}
        strand = gene["strand"]
        start = gene["start"]
        end = gene["end"]
        if strand == "-":
            seq_start = max(int(start), int(end) - self.gene_downstream_neighbour_hood)
            seq_end = int(end) + self.gene_upstream_neighbour_hood
        else:
            seq_start = max(0, int(start) - self.gene_upstream_neighbour_hood)
            seq_end = min(int(end), int(start) + self.gene_downstream_neighbour_hood)
        gene_dict["start"] = seq_start
        gene_dict["end"] = seq_end
        assert (
            len(gene_dict["sequence"].split(",")[0]) == (seq_end - seq_start)
        ), f"Gene sequence length mismatch: {len(gene_dict['sequence'].split(',')[0])} != {seq_end - seq_start}"
        return cre_df, gene_dict

    def load_gene_data_from_vcf(
        self, gene: Dict[str, Any], chromosome: str, sample_name: str, vcf_path: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Load gene CRE and sequence data"""
        # Load gene CREs
        mutated_seq = ExtractSeqFromBed(
            neighbour_hood=self.gene_downstream_neighbour_hood,
            ref_fasta=self.fasta_path,
            upstream_neighbour_hood=self.gene_upstream_neighbour_hood,
        ).process_gene(gene, vcf_path, variant_type="SNP")
        genes_cre_map_path = self.gene_cre_manifest.get_file_path(gene["gene_id"])
        genes_cre_map = multi_try_load_csv(genes_cre_map_path)
        bed_regions = genes_cre_map[["chromosome", "start_cre", "end_cre", "cre_name"]]
        bed_regions = bed_regions.rename(
            columns={
                "chromosome": "chrom",
                "start_cre": "start",
                "end_cre": "end",
                "cre_name": "cCRE",
            }
        )
        cre_df = ExtractSeqFromBed(
            neighbour_hood=self.cre_neighbour_hood, ref_fasta=self.fasta_path
        ).process_subject(
            vcf_file=vcf_path, bed_regions=bed_regions, variant_type="SNP"
        )
        encoded_seqs = []
        seqs = []
        for i, row in cre_df.iterrows():
            forward_seq = row["sequence"].split(",")[0]
            reverse_complement = self.sequence_processor.reverse_complement(forward_seq)
            seqs.append(forward_seq + "," + reverse_complement)
            encoded_f, _, encoded_r, _ = self.bpe.encode(
                [forward_seq, reverse_complement]
            )
            encoded_seqs.append(
                [[float(x) for x in encoded_f], [float(x) for x in encoded_r]]
            )
        cre_df["encoded_seq"] = encoded_seqs
        cre_df["strand"] = gene["strand"]
        cre_df["sequence"] = seqs
        # Load gene sequence data
        gene_dict = {
            "sequence": mutated_seq
            + ","
            + self.sequence_processor.reverse_complement(mutated_seq),
            "strand": gene["strand"],
        }
        strand = gene["strand"]
        start = gene["start"]
        end = gene["end"]
        if strand == "-":
            seq_start = max(int(start), int(end) - self.gene_downstream_neighbour_hood)
            seq_end = int(end) + self.gene_upstream_neighbour_hood
        else:
            seq_start = max(0, int(start) - self.gene_upstream_neighbour_hood)
            seq_end = min(int(end), int(start) + self.gene_downstream_neighbour_hood)
        gene_dict["start"] = seq_start
        gene_dict["end"] = seq_end
        assert (
            len(gene_dict["sequence"].split(",")[0]) == (seq_end - seq_start)
        ), f"Gene sequence length mismatch: {len(gene_dict['sequence'].split(',')[0])} != {seq_end - seq_start}"
        if gene["strand"] == "-":
            cre_df = cre_df.iloc[::-1].reset_index(
                drop=True
            )  # reverse the cres if the gene is on the minus strand

        return cre_df, gene_dict

    def apply_variant_to_data(
        self, variant: Variant, cre_df: pd.DataFrame, gene_dict: Dict, sample_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, bool, bool]:
        """Apply variant to gene CRE and sequence data"""
        df_het = cre_df.copy(deep=True)
        df_hom = cre_df.copy(deep=True)
        gene_het = gene_dict.copy()
        gene_hom = gene_dict.copy()

        overlap_cre = False
        overlap_gene_ref = False
        overlap_gene_het = False
        overlap_gene_hom = False
        encoded_seq_het = [copy.deepcopy(item) for item in df_het["encoded_seq"]]
        encoded_seq_hom = [copy.deepcopy(item) for item in df_hom["encoded_seq"]]
        cre_token_position = np.nan
        gene_token_position_ref = np.nan
        gene_token_position_het = np.nan
        gene_token_position_hom = np.nan
        # Apply variant to CREs
        for i, row in cre_df.iterrows():
            if row["start_cre"] > variant.pos and row["strand"] == "+":
                # [variant]>>>>>[cre]
                break
            if row["end_cre"] < variant.pos and row["strand"] == "-":
                # [cre]<<<<<[variant]
                break
            if (
                row["start_cre"] < variant.pos <= row["end_cre"]
            ):  # variant.pos is 1-indexed but CREs are 0-indexed
                cre_token_position = i
                overlap_cre = True
                pos_on_seq = variant.pos - row["start_cre"] - 1

                # Validate reference match
                forward_seq = row["sequence"].split(",")[0]
                if sample_name == "hg38":
                    assert (
                        forward_seq[pos_on_seq] == variant.ref
                    ), f"Reference mismatch: {forward_seq[pos_on_seq]} != {variant.ref}"

                # Apply variant
                seq_het, seq_hom = self.sequence_processor.apply_variant(
                    row["sequence"], variant, pos_on_seq
                )

                df_het.at[i, "sequence"] = seq_het
                df_hom.at[i, "sequence"] = seq_hom

                # Re-encode sequences
                encoded_f_het, _, encoded_r_het, _ = self.bpe.encode(seq_het)
                encoded_f_hom, _, encoded_r_hom, _ = self.bpe.encode(seq_hom)

                encoded_seq_het[i] = [
                    [float(x) for x in encoded_f_het],
                    [float(x) for x in encoded_r_het],
                ]
                encoded_seq_hom[i] = [
                    [float(x) for x in encoded_f_hom],
                    [float(x) for x in encoded_r_hom],
                ]

        df_het["encoded_seq"] = encoded_seq_het
        df_hom["encoded_seq"] = encoded_seq_hom
        # Apply variant to gene sequence
        gene_seq = str(gene_dict["sequence"])
        seq_start = int(gene_dict.get("start", 0))
        seq_end = int(gene_dict.get("end", len(gene_seq.split(",")[0]) + seq_start))

        if seq_start < variant.pos <= seq_end:
            overlap_gene_ref = True
            pos_on_gene_seq = variant.pos - seq_start - 1

            # Validate reference match
            forward_seq = gene_seq.split(",")[0]
            if sample_name == "hg38":
                assert (
                    forward_seq[pos_on_gene_seq] == variant.ref
                ), f"Gene reference mismatch: {forward_seq[pos_on_gene_seq]} != {variant.ref}"
            strand = gene_dict["strand"]
            sequence = gene_dict["sequence"]

            # Apply variant to gene
            seq_het, seq_hom = self.sequence_processor.apply_variant(
                gene_seq, variant, pos_on_gene_seq
            )

            # Check if the variant is in the gene context for ref
            overlap_gene_ref, gene_token_position_ref = (
                self.check_if_variant_in_gene_context(strand, sequence, pos_on_gene_seq)
            )
            gene_token_position_ref = min(
                gene_token_position_ref, self.context_window - 1
            )

            overlap_gene_ref = True
            # Note: We are setting the overlap_gene_ref to True to addressthe case where the variant is outside the gene context for ref, we set the token position to the last context window
            # if not overlap_gene_ref:
            #    return df_het, df_hom, gene_het, gene_hom, overlap_cre, overlap_gene_ref, cre_token_position, gene_token_position_ref, gene_token_position_het, gene_token_position_hom

            # Check if the variant is in the gene context for het, in the rare case that the variant is outside the gene context for het, we set the token position to the last context window
            overlap_gene_het, gene_token_position_het = (
                self.check_if_variant_in_gene_context(strand, seq_het, pos_on_gene_seq)
            )
            gene_token_position_het = min(
                gene_token_position_het, self.context_window - 1
            )  # if overlap_gene_het else np.nan

            # Check if the variant is in the gene context for hom, in the rare case that the variant is outside the gene context for hom, we set the token position to the last context window
            overlap_gene_hom, gene_token_position_hom = (
                self.check_if_variant_in_gene_context(strand, seq_hom, pos_on_gene_seq)
            )
            gene_token_position_hom = min(
                gene_token_position_hom, self.context_window - 1
            )  # if overlap_gene_hom else np.nan

            gene_het["sequence"] = seq_het
            gene_hom["sequence"] = seq_hom

        return (
            df_het,
            df_hom,
            gene_het,
            gene_hom,
            overlap_cre,
            overlap_gene_ref,
            cre_token_position,
            gene_token_position_ref,
            gene_token_position_het,
            gene_token_position_hom,
        )

    def check_if_variant_in_gene_context(self, strand, sequence, pos_on_gene_seq):
        """Check if the variant is in the gene context"""
        # Check if the variant is in the gene context
        strand = 0 if strand == "+" else 1
        seq = sequence.split(",")[strand]
        if strand == 1:
            pos_on_gene_seq = len(seq) - pos_on_gene_seq - 1

        d = self.bpe.encode_with_position(seq, pos_on_gene_seq)
        token_position = d["position_id"]
        # if token_position < self.max_length*self.context_window:
        token_window = token_position // self.max_length
        return True, token_window
        # else:
        #    return False, np.nan

    def _adjust_length(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Adjust sequence length with padding or truncation"""
        attention_mask = [0] * len(token_ids)

        if len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids += [self.pad_token_id] * padding_length
            attention_mask += [1] * padding_length
        else:
            token_ids = token_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        return token_ids, attention_mask

    def _chunkify_gene_sequence(
        self, gene_data: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process gene sequence into chunks for the model"""
        strand = gene_data["strand"]
        gene_seq = str(gene_data["sequence"])

        # Select appropriate strand
        if strand == "+":
            token_ids, _, _, _ = self.bpe.encode([gene_seq.split(",")[0], "A"])
        else:
            token_ids, _, _, _ = self.bpe.encode([gene_seq.split(",")[1], "A"])

        tokens_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.zeros_like(tokens_tensor)

        # Create chunks
        chunk_list = []
        chunk_attention_list = []
        max_chunks = self.context_window

        for start in range(0, tokens_tensor.size(1), self.max_length):
            if max_chunks <= 0:
                break

            end = start + self.max_length
            chunk = tokens_tensor[:, start:end]
            chunk_attention = attention_mask[:, start:end]

            # Pad if necessary
            if chunk.size(1) < self.max_length:
                padding_size = self.max_length - chunk.size(1)
                padding = torch.full(
                    (chunk.size(0), padding_size), self.pad_token_id, dtype=chunk.dtype
                )
                chunk = torch.cat([chunk, padding], dim=1)

                attention_padding = torch.ones(
                    (chunk_attention.size(0), padding_size), dtype=chunk_attention.dtype
                )
                chunk_attention = torch.cat([chunk_attention, attention_padding], dim=1)

            chunk_list.append(chunk)
            chunk_attention_list.append(chunk_attention)
            max_chunks -= 1

        # Stack and format for model
        chunks = torch.vstack(chunk_list).unsqueeze(1)  # .repeat(1, 2, 1)
        attention_masks = torch.vstack(chunk_attention_list).unsqueeze(
            1
        )  # .repeat(1, 2, 1)

        return chunks.long(), attention_masks.bool()

    def create_batch(
        self,
        cre_df: pd.DataFrame,
        gene_data: Dict,
        tissue: int,
        cre_token_position: int,
        gene_token_position: int,
    ) -> Dict[str, torch.Tensor]:
        """Create a model-ready batch from CRE and gene data"""
        if len(cre_df) == 0:
            return None

        # Process CRE sequences
        X = []
        attention_masks = []
        ref_labels = []

        for i, row in cre_df.iterrows():
            # Get encoded sequences
            token_ids_f = row["encoded_seq"][0]
            token_ids_r = row["encoded_seq"][1]
            strand = row["strand"]
            # Adjust sequence lengths and create attention masks
            if strand == "-":
                token_ids_strand, attention_mask_strand = self._adjust_length(
                    token_ids_r
                )
            else:
                token_ids_strand, attention_mask_strand = self._adjust_length(
                    token_ids_f
                )

            X.append(
                [
                    token_ids_strand,
                ]
            )
            attention_masks.append(
                [
                    attention_mask_strand,
                ]
            )

            # Get reference CRE label
            cre_type = (
                row["cCRE"].iloc[0] if hasattr(row["cCRE"], "iloc") else row["cCRE"]
            )
            ref_labels.append(self.ref_cre_to_idx.get(cre_type, 0))

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.long)
        attention_tensor = torch.tensor(attention_masks, dtype=torch.bool)
        ref_labels_tensor = torch.tensor(ref_labels, dtype=torch.long)

        # Process gene sequence
        gene_embeddings, gene_attention = self._chunkify_gene_sequence(gene_data)

        # Create batch dictionary
        batch = {
            "cre_sequences": [X_tensor],
            "cre_attention_masks": [attention_tensor],
            "tissue_context": [torch.tensor(tissue, dtype=torch.long)],
            "labels": [torch.zeros(len(X), dtype=torch.long)],
            "ref_labels": [ref_labels_tensor],
            "gene_expression": [torch.tensor([1.0], dtype=torch.float)],
            "strand": torch.tensor(
                [0 if gene_data["strand"] == "+" else 1], dtype=torch.long
            ).unsqueeze(0),
            "gene_embeddings": [gene_embeddings],
            "gene_attention_masks": [gene_attention],
            "cre_token_position": torch.tensor([cre_token_position]).unsqueeze(0),
            "gene_token_position": torch.tensor([gene_token_position]).unsqueeze(0),
        }

        return batch

    def process_variant_gene_pair(
        self,
        variant: Variant,
        gene: Dict[str, Any],
        population: str,
        sample_name: str,
        tissue: List[int],
        vcf_path: str = None,
    ) -> Dict[str, Any]:

        """Process a variant-gene pair and return all three conditions (ref, het, hom)"""
        try:
            # Load gene data
            if vcf_path is not None:
                cre_df, gene_dict = self.load_gene_data_from_vcf(
                    gene, variant.chrom, sample_name, vcf_path
                )
            else:
                cre_df, gene_dict = self.load_gene_data(
                    gene,
                    chromosome=variant.chrom,
                    sample_name=sample_name,
                    population=population,
                )

            # Apply variant
            (
                df_het,
                df_hom,
                gene_het,
                gene_hom,
                overlap_cre,
                overlap_gene,
                cre_token_position,
                gene_token_position_ref,
                gene_token_position_het,
                gene_token_position_hom,
            ) = self.apply_variant_to_data(
                variant, cre_df, gene_dict, sample_name=sample_name
            )
            # Determine variant type
            if not overlap_cre and not overlap_gene:
                variant_type = "No overlap"
                return {
                    "gene_id": gene["gene_id"],
                    "variant_type": variant_type,
                    "ref_batch": None,
                    "het_batch": None,
                    "hom_batch": None,
                    "overlap_cre": False,
                    "overlap_gene": False,
                }

            # Create batches
            ref_batch = self.create_batch(
                cre_df, gene_dict, tissue, cre_token_position, gene_token_position_ref
            )
            het_batch = (
                self.create_batch(
                    df_het,
                    gene_het,
                    tissue,
                    cre_token_position,
                    gene_token_position_het,
                )
                if overlap_cre or overlap_gene
                else None
            )
            hom_batch = (
                self.create_batch(
                    df_hom,
                    gene_hom,
                    tissue,
                    cre_token_position,
                    gene_token_position_hom,
                )
                if overlap_cre or overlap_gene
                else None
            )

            # Determine specific variant type
            if overlap_cre and overlap_gene:
                variant_type = "Gene and CRE overlap"
            elif overlap_cre:
                variant_type = "CRE overlap only"
            elif overlap_gene:
                variant_type = "Gene overlap only"
            else:
                variant_type = "No overlap"

            return {
                "gene_id": gene["gene_id"],
                "variant_type": variant_type,
                "ref_batch": ref_batch,
                "het_batch": het_batch,
                "hom_batch": hom_batch,
                "overlap_cre": overlap_cre,
                "overlap_gene": overlap_gene,
            }

        except Exception as e:
            raise RuntimeError(
                f"Error processing variant {variant.chrom}:{variant.pos} with gene {gene['gene_id']}: {str(e)}"
            ) from e

    def __len__(self):
        return len(self.gene_variant_pairs)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def create_empty_batch(self):
        batch = {
            "cre_sequences": [],
            "cre_attention_masks": [],
            "tissue_context": [],
            "labels": [],
            "ref_labels": [],
            "gene_expression": [],
            "strand": [],
            "gene_embeddings": [],
            "gene_attention_masks": [],
            "variant_type": "No overlap",
        }
        return batch

    def load_data(self, idx):
        variant, gene, sample_name, population, vcf_path = (
            self.gene_variant_pairs[idx]["variant"],
            self.gene_variant_pairs[idx]["gene"],
            self.gene_variant_pairs[idx]["sample_name"],
            self.gene_variant_pairs[idx]["population"],
            self.gene_variant_pairs[idx]["vcf_path"],
        )
        prebatch = self.process_variant_gene_pair(
            variant,
            gene,
            population,
            sample_name=sample_name,
            tissue=variant.tissue,
            vcf_path=vcf_path,
        )

        if prebatch["variant_type"] == "No overlap":
            batch = self.create_empty_batch()
            return batch
        ref_batch = prebatch["ref_batch"]
        het_batch = prebatch["het_batch"]
        hom_batch = prebatch["hom_batch"]
        batch = {}
        for key in ref_batch.keys():
            if isinstance(ref_batch[key], List):
                batch[key] = ref_batch[key] + het_batch[key] + hom_batch[key]
            elif isinstance(ref_batch[key], torch.Tensor):
                batch[key] = torch.cat(
                    [ref_batch[key], het_batch[key], hom_batch[key]], dim=0
                )
            else:
                raise ValueError(f"Unsupported type: {type(ref_batch[key])}")
        batch["variant_type"] = prebatch["variant_type"]
        return batch

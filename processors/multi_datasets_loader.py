import pandas as pd
from typing import List, Set
from omegaconf import OmegaConf
from datasets.vepdataset import Variant
import logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
   )
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class MultiDatasetsLoader:
    """Handles loading variant data and gene annotations"""

    def __init__(self, config: OmegaConf):
        self.config = config
        self.gencode_genes = None
        self.all_cres = None
        self.variants = None

    def load_annotations(self):
        """Load gene annotations and CRE data"""
        log.info("Loading gene annotations...")
        self.gencode_genes = pd.read_csv(self.config.gencode)

        log.info("Loading CRE annotations...")
        self.all_cres = pd.read_csv(self.config.all_cres, sep="\t", header=None)
        self.all_cres.columns = [
            "chromosome",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "type",
        ]

    def _load_variants(self, var_df: pd.DataFrame) -> pd.DataFrame:
        """Load custom variant file"""
        df = var_df
        if "chr" in df.columns:
            df = df.rename(columns={"chr": "chrom"})
        columns_to_have = ["chrom", "pos", "ref", "alt", "tissue"]
        for column in columns_to_have:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in {var_df.columns}")

        df = df.sort_values(by=["chrom", "pos"]).reset_index(drop=True)
        log.info(f"Loaded {len(df)} variants")
        return df

    def get_probable_genes(
        self, variant: Variant, window_size: int = 1000000
    ) -> Set[str]:
        """Get genes within specified window of variant"""
        if self.gencode_genes is None:
            raise RuntimeError(
                "Gene annotations not loaded. Call load_annotations() first."
            )

        # Filter genes by chromosome
        chrom_genes = self.gencode_genes[
            self.gencode_genes["chromosome"] == variant.chrom
        ].reset_index(drop=True)

        probable_genes = []

        # Find genes within window
        for _, gene in chrom_genes.iterrows():
            gene_start = gene["start"]
            gene_end = gene["end"]

            # Check if variant is within window of gene
            if gene_start - window_size < variant.pos < gene_end + window_size:
                probable_genes.append(
                    {
                        "gene_id": gene["gene_id"],
                        "start": gene_start,
                        "end": gene_end,
                        "gene_name": gene["gene_name"],
                        "strand": gene["strand"],
                        "chromosome": gene["chromosome"],
                    }
                )

        return probable_genes

    def create_variant_objects(
        self, df: pd.DataFrame, tissue_vocab: dict
    ) -> List[Variant]:
        """Convert DataFrame to Variant objects"""
        variants = []
        for _, row in df.iterrows():
            tissues = row["tissue"].split(",")
            tissue_idx = [tissue_vocab[t] for t in tissues]
            genes = row.get("gene_id", "").split(",") if "gene_id" in row else []
            genes = [gene.split(".")[0] for gene in genes]  # Clean up gene IDs
            variant = Variant(
                chrom=row["chrom"],
                pos=row["pos"],
                ref=row["ref"],
                alt=row["alt"],
                consequence=row.get("consequence", "NA"),
                label=row.get("label", "NA"),
                tissue=tissue_idx,
                gene_id=genes,
            )
            variants.append(variant)
        return variants

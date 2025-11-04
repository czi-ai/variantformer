import os
import torch
from pathlib import Path
from typing import List
from omegaconf import OmegaConf
import pandas as pd
from processors.multi_datasets_loader import MultiDatasetsLoader
from datasets.vepdataset import VEPDataset, collate_fn, Variant
from processors.model_manager import ModelManager
from utils.seq import BPEEncoder
from torch.utils.data import DataLoader
import numpy as np
from lightning.pytorch import Trainer
import logging
from utils.functions import generate_log2fc_score
from utils.assets import GeneManifestLookup, GeneSequencesManifestLookup, CreSequencesManifestLookup

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
   )
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class VariantProcessor:
    """Main class for processing variants and predicting pathogenicity"""

    # TODO: needs a major refactor: building paths like this is brittle, we may need a master artifact manifest
    # and a just in time artifact fetcher from s3/local filesystem that abstracts path building like below
    def __init__(self, model_class: str = "v4_pcg"):
        # Load configurations
        base_dir = Path(__file__).parent.parent.resolve()
        self.config_location = base_dir / "configs"
        model_config_path = self.config_location / "vf_model.yaml"
        model_config = OmegaConf.load(model_config_path)[model_class]
        vep_loader_config = OmegaConf.load(base_dir / "configs" / "veploader.yaml")

        self.gene_cre_manifest = GeneManifestLookup()
        self.gene_seq_manifest = GeneSequencesManifestLookup()
        self.cre_seq_manifest = CreSequencesManifestLookup()

        # Resolve all paths relative to project root
        if not os.path.isabs(vep_loader_config.CRE_BED):
            vep_loader_config.CRE_BED = str(base_dir / vep_loader_config.CRE_BED)
        if not os.path.isabs(vep_loader_config.fasta_path):
            vep_loader_config.fasta_path = str(base_dir / vep_loader_config.fasta_path)
        if not os.path.isabs(vep_loader_config.af_path):
            vep_loader_config.af_path = str(base_dir / vep_loader_config.af_path)
        if not os.path.isabs(model_config.dataset.gencode_v24):
            model_config.dataset.gencode_v24 = str(
                base_dir / model_config.dataset.gencode_v24
            )
        if not os.path.isabs(model_config.model.checkpoint_path):
            model_config.model.checkpoint_path = str(
                base_dir / model_config.model.checkpoint_path
            )
        if not os.path.isabs(model_config.model.cre_tokenizer.path):
            model_config.model.cre_tokenizer.path = str(
                base_dir / model_config.model.cre_tokenizer.path
            )
        if not os.path.isabs(model_config.model.gene_tokenizer.path):
            model_config.model.gene_tokenizer.path = str(
                base_dir / model_config.model.gene_tokenizer.path
            )

        self.vep_loader_config = vep_loader_config
        vep_config = self._create_vep_config(model_config, vep_loader_config)
        self.config = vep_config
        self.multi_data_loader = MultiDatasetsLoader(vep_config)
        # Pass the model config directly to avoid configuration structure issues
        self.model_manager = ModelManager(model_config.model.copy())
        self.vep_dataset = None
        # Populations to consider
        self.populations = ["REF_HG38", "EAS", "EUR", "AFR", "SAS", "AMR"]
        # Load vocabularies
        base_dir = Path(__file__).parent.parent.resolve()
        vocab_path = base_dir / "vocabs" / "dataset_vocab.yaml"
        vocab_path = str(vocab_path)
        self.data_vocab = OmegaConf.load(vocab_path)

        # Resolve dataset_vocab paths relative to repo root
        for dataset_key in self.data_vocab.keys():
            if "cre_location" in self.data_vocab[dataset_key] and not os.path.isabs(
                self.data_vocab[dataset_key].cre_location
            ):
                self.data_vocab[dataset_key].cre_location = str(
                    base_dir / self.data_vocab[dataset_key].cre_location
                )
            if "gene_location" in self.data_vocab[dataset_key] and not os.path.isabs(
                self.data_vocab[dataset_key].gene_location
            ):
                self.data_vocab[dataset_key].gene_location = str(
                    base_dir / self.data_vocab[dataset_key].gene_location
                )

        self.tissue_vocab = OmegaConf.load(base_dir / "vocabs" / "tissue_vocab.yaml")
        self.tissue_idx_to_name = {v: k for k, v in self.tissue_vocab.items()}
        self.bpe_vocab_path = str(
            base_dir / "vocabs" / "bpe_vocabulary_500_using_huggingface.json"
        )
        self.fasta_path = vep_loader_config.fasta_path

    def _create_vep_config(self, model_config, data_config):
        D = {}
        D["all_cres"] = data_config.CRE_BED
        D["checkpoint_path"] = model_config.model.checkpoint_path
        D["train_config"] = model_config
        D["gencode"] = model_config.dataset.gencode_v24
        D["emb_dim"] = model_config.model.emb_dim
        D["cre_neighbour_hood"] = model_config.dataset.cre_neighbour_hood
        D["gene_upstream_neighbour_hood"] = (
            model_config.dataset.gene_upstream_neighbour_hood
        )
        D["gene_downstream_neighbour_hood"] = (
            model_config.dataset.gene_downstream_neighbour_hood
        )
        D["max_length"] = model_config.dataset.max_length
        D["context_window"] = model_config.dataset.max_chunks
        D["fasta_path"] = data_config.fasta_path
        D["precision"] = model_config.model.precision
        D["af_path"] = data_config.af_path
        return OmegaConf.create(D)

    def _setup_output_directory(self, output_dir: str = None):
        """Create output directory if it doesn't exist"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        var_df: pd.DataFrame,
        output_dir: str,
        vcf_path: str = None,
        sample_name: str = None,
    ):
        """Initialize all components"""
        log.info("Initializing Variant Processor...")
        self.config.output_location = output_dir
        self._setup_output_directory(output_dir)

        # Load annotations
        self.multi_data_loader.load_annotations()

        # Load variants
        variants = self.load_variants(var_df)
        # variants = self.multi_data_loader.create_variant_objects(variants, self.tissue_vocab)

        # Create gene-variant pairs
        self.gene_variant_pairs = []
        mapped = 0
        if self._check_variant_exists():
            raise ValueError(
                f"Variants already processed at {self._get_variant_output_path()}. To reprocess, change the output directory or remove the existing file."
            )

        for variant in variants:
            variant_genes = variant.gene_id if hasattr(variant, "gene_id") else []
            probable_genes = self.multi_data_loader.get_probable_genes(variant)
            if len(variant_genes) != 0:
                probable_genes = [
                    gene
                    for gene in probable_genes
                    if gene["gene_id"].split(".")[0] in variant_genes
                ]
            if len(probable_genes) != 0:
                mapped += 1
            for gene in probable_genes:
                if vcf_path is not None and sample_name is not None:
              
                    self.gene_variant_pairs.append(
                        {
                            "variant": variant,
                            "gene": gene,
                            "population": "SAMPLE",
                            "sample_name": sample_name,
                            "vcf_path": vcf_path,
                        }
                    )

                    self.gene_variant_pairs.append(
                        {
                            "variant": variant,
                            "gene": gene,
                            "population": "REF_HG38",
                            "sample_name": "hg38",
                            "vcf_path": None,
                        }
                    )
                else:
                    for pop in self.populations:
                        self.gene_variant_pairs.append(
                            {
                                "variant": variant,
                                "gene": gene,
                                "population": pop,
                                "sample_name": self.data_vocab[pop].sample_name,
                                "vcf_path": None,
                            }
                        )
        log.info(f"Mapped {mapped} gene-variant pairs")
        if mapped == 0:
            raise ValueError(
                "No gene-variant pairs found. Check your input data and annotations."
            )
        # Setup BPE encoder
        log.info("Loading BPE encoder...")
        bpe = BPEEncoder()
        bpe.load_vocabulary(self.bpe_vocab_path)
        # Initialize VEP dataset
        vep_dataset = VEPDataset(
            bpe_encoder=bpe,
            gene_cre_manifest=self.gene_cre_manifest,
            gene_seq_manifest=self.gene_seq_manifest,
            cre_seq_manifest=self.cre_seq_manifest,
            max_length=self.config.max_length,
            context_window=self.config.context_window,
            cre_neighbour_hood=self.config.cre_neighbour_hood,
            gene_upstream_neighbour_hood=self.config.gene_upstream_neighbour_hood,
            gene_downstream_neighbour_hood=self.config.gene_downstream_neighbour_hood,
            gene_variant_pairs=self.gene_variant_pairs,
            data_vocab=self.data_vocab,
            fasta_path=self.fasta_path,
        )

        # ################# only for debugging
        # log.warn('Generating the dataloader with num_workers=0 for debugging')
        # self.vep_loader_config.dataloader.num_workers = 0
        # dataloader = DataLoader(
        #     vep_dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=self.vep_loader_config.dataloader.num_workers,
        #     # prefetch_factor=self.vep_loader_config.dataloader.prefetch_factor,
        #     pin_memory=self.vep_loader_config.dataloader.pin_memory,
        #     collate_fn=collate_fn,
        # )
        ########################

        dataloader = DataLoader(
            vep_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.vep_loader_config.dataloader.num_workers,
            prefetch_factor=self.vep_loader_config.dataloader.prefetch_factor,
            pin_memory=self.vep_loader_config.dataloader.pin_memory,
            collate_fn=collate_fn,
        )

        # Load model
        log.info("Loading model...")
        model, ckpt_path = self.model_manager.load_model()

        log.info("Initializing trainer...")
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            precision=self.config.precision,
            enable_checkpointing=False,
        )
        # Set VEP flag to True
        model.vep = True

        log.info("Initialization complete!")
        return vep_dataset, dataloader, model, trainer, ckpt_path

    def load_variants(self, var_df: pd.DataFrame) -> List[Variant]:
        """Load and prepare variants for processing"""
        log.info("Loading variants...")
        variants_df = self.multi_data_loader._load_variants(var_df)

        # Get chunk if specified
        if hasattr(self.config, "chunks") and hasattr(self.config, "chunk_id"):
            variants_df = self.multi_data_loader.get_variants_chunk(
                self.config.chunk_id, self.config.chunks
            )

        variants = self.multi_data_loader.create_variant_objects(
            variants_df, self.tissue_vocab
        )
        log.info(f"Loaded {len(variants)} variants for processing")
        return variants

    def _check_variant_exists(self) -> bool:
        """Check if variant has already been processed"""
        output_file = self._get_variant_output_path()
        return os.path.exists(output_file)

    def _get_variant_output_path(self) -> str:
        """Get output file path for a variant"""
        input_file = (
            self.config.variants_file
            if hasattr(self.config, "variants_file")
            else "vep"
        )
        if "chunks" in self.config and self.config.chunks > 1:
            # If processing in chunks, append chunk_id to filename
            filename = f"{input_file.split('/')[-1].split('.')[0]}_chunk{self.config.chunk_id}_VF.parquet"
        else:
            filename = f"{input_file.split('/')[-1].split('.')[0]}_VF.parquet"
        return os.path.join(self.config.output_location, filename)

    def compile_predictions(self, predictions, vcf_path=None):
        """Compile predictions"""
        D = {
            "chrom": [],
            "pos": [],
            "ref": [],
            "alt": [],
            "genes": [],
            "tissues": [],
            "variant_type": [],
            "population": [],
            "sample_name": [],
            "zygosity": [],
            "gene_exp": [],
            "gene_emb": [],
            "gene_token_embedding": [],
            "cre_token_embedding": [],
        }
        for i, attr in enumerate(self.gene_variant_pairs):
            variant = attr["variant"]
            gene = attr["gene"]
            population = attr["population"]
            sample_name = attr["sample_name"]
            variant_type = predictions[i]["variant_type"]
            if len(predictions[i]["pred_gene_exp"]) == 0:
                log.info(f"No predictions for {variant.chrom}:{variant.pos} and {gene}")
                ref_gene_exp = np.full(
                    (len(variant.tissue), 1), np.nan, dtype=np.float32
                )
                het_gene_exp = np.full(
                    (len(variant.tissue), 1), np.nan, dtype=np.float32
                )
                hom_gene_exp = np.full(
                    (len(variant.tissue), 1), np.nan, dtype=np.float32
                )
                ref_gene_emb = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                het_gene_emb = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                hom_gene_emb = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                ref_gene_token_embedding = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                het_gene_token_embedding = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                hom_gene_token_embedding = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                ref_cre_token_embedding = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                het_cre_token_embedding = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
                hom_cre_token_embedding = np.full(
                    (len(variant.tissue), self.config.emb_dim), np.nan, dtype=np.float32
                )
            else:
                ref_gene_exp = predictions[i]["pred_gene_exp"][0]  # (num_tissues, 1)
                het_gene_exp = predictions[i]["pred_gene_exp"][1]  # (num_tissues, 1)
                hom_gene_exp = predictions[i]["pred_gene_exp"][2]  # (num_tissues, 1)
                ref_gene_token_embedding = predictions[i]["gene_token_embedding"][
                    0
                ]  # (num_tissues, emb_dim)
                het_gene_token_embedding = predictions[i]["gene_token_embedding"][
                    1
                ]  # (num_tissues, emb_dim)
                hom_gene_token_embedding = predictions[i]["gene_token_embedding"][
                    2
                ]  # (num_tissues, emb_dim)
                ref_cre_token_embedding = predictions[i]["cre_token_embedding"][
                    0
                ]  # (num_tissues, emb_dim)
                het_cre_token_embedding = predictions[i]["cre_token_embedding"][
                    1
                ]  # (num_tissues, emb_dim)
                hom_cre_token_embedding = predictions[i]["cre_token_embedding"][
                    2
                ]  # (num_tissues, emb_dim)

                ref_gene_emb = predictions[i]["embd"][0]  # (num_tissues, emb_dim)
                het_gene_emb = predictions[i]["embd"][1]  # (num_tissues, emb_dim)
                hom_gene_emb = predictions[i]["embd"][2]  # (num_tissues, emb_dim)

            for tissue_idx, tissue in enumerate(variant.tissue):
                for (
                    zyg,
                    gene_exp,
                    gene_emb,
                    gene_token_embedding,
                    cre_token_embedding,
                ) in zip(
                    ["2", "1", "0"],
                    [hom_gene_exp, het_gene_exp, ref_gene_exp],
                    [hom_gene_emb, het_gene_emb, ref_gene_emb],
                    [
                        hom_gene_token_embedding,
                        het_gene_token_embedding,
                        ref_gene_token_embedding,
                    ],
                    [
                        hom_cre_token_embedding,
                        het_cre_token_embedding,
                        ref_cre_token_embedding,
                    ],
                ):
                    D["chrom"].append(variant.chrom)
                    D["pos"].append(variant.pos)
                    D["ref"].append(variant.ref)
                    D["alt"].append(variant.alt)
                    D["genes"].append(gene["gene_id"])
                    D["tissues"].append(
                        self.tissue_idx_to_name[variant.tissue[tissue_idx]]
                    )
                    D["variant_type"].append(variant_type)
                    D["population"].append(population)
                    D["sample_name"].append(sample_name)
                    D["zygosity"].append(zyg)
                    D["gene_exp"].append(gene_exp[tissue_idx, 0])
                    D["gene_emb"].append(gene_emb[tissue_idx, :])
                    D["gene_token_embedding"].append(
                        gene_token_embedding[tissue_idx, :]
                    )
                    D["cre_token_embedding"].append(cre_token_embedding[tissue_idx, :])
        df = pd.DataFrame(D)
        if vcf_path is None:
            # Remove zygosity = 0 rows except for population = REF_HG38
            df = df[
                (df["zygosity"] != "0")
                | ((df["zygosity"] == "0") & (df["population"] == "REF_HG38"))
            ].reset_index(drop=True)
        output_file = self._get_variant_output_path()
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
        elif output_file.endswith(".parquet"):
            df.to_parquet(output_file)
        log.info(f"Predictions saved to {output_file}")
        return df

    def eqtl_scores(self, df: pd.DataFrame):
        """Calculate log2FC scores"""
        af_path = self.config.af_path
        # df = generate_poisson_score(df, af_path)
        df = generate_log2fc_score(df, af_path)
        return df

    def format_scores(self, df: pd.DataFrame):
        """Calculate EQTL score"""
        df["variant_id"] = (
            df[["chrom", "pos", "ref", "alt"]].astype(str).agg("_".join, axis=1)
        )
        df["gt-exp"] = df["population"] + "-" + df["zygosity"] + "-exp"
        df = df.rename(columns={"chrom": "chr"})
        df_exp = (
            df[
                [
                    "variant_id",
                    "genes",
                    "tissues",
                    "variant_type",
                    "gt-exp",
                    "gene_exp",
                    "chr",
                    "pos",
                    "ref",
                    "alt",
                ]
            ]
            .drop_duplicates(
                subset=["variant_id", "genes", "tissues", "variant_type", "gt-exp"],
                keep="first",
            )
            .pivot(
                index=[
                    "variant_id",
                    "genes",
                    "tissues",
                    "chr",
                    "pos",
                    "ref",
                    "alt",
                    "variant_type",
                ],
                columns="gt-exp",
                values="gene_exp",
            )
            .reset_index()
        )
        vf_vep = df_exp.dropna(subset=["REF_HG38-0-exp"]).reset_index(drop=True)
        return vf_vep

    def predict(
        self,
        var_df: pd.DataFrame,
        output_dir: str,
        vcf_path: str = None,
        sample_name: str = None,
    ):
        """Predict pathogenicity"""
        vep_dataset, dataloader, model, trainer, ckpt_path = self.initialize(
            var_df, output_dir, vcf_path, sample_name
        )
        predictions = trainer.predict(model, dataloader)
        df = self.compile_predictions(predictions, vcf_path=vcf_path)
        self.cleanup()
        return df

    def cleanup(self):
        """Clean up resources"""
        # Clear GPU memory if needed
        if (
            hasattr(self.model_manager, "model")
            and self.model_manager.model is not None
        ):
            del self.model_manager.model
            if hasattr(torch, "cuda"):
                torch.cuda.empty_cache()

        log.info("Cleanup complete")

import os
from omegaconf import OmegaConf
from pathlib import Path
from datasets.vcfdataset import VCFDataset, collate_fn_batching
import pandas as pd
from torch.utils.data import DataLoader
from processors.model_manager import ModelManager
from lightning.pytorch import Trainer
import torch
import logging

from utils.assets import GeneManifestLookup

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
   )
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class VCFProcessor:
    def __init__(self, model_class: str = "v4_pcg"):
        base_dir = Path(__file__).parent.parent.resolve()
        self.config_location = base_dir / "configs"
        model_config_path = self.config_location / "vf_model.yaml"
        self.model_config = OmegaConf.load(model_config_path)[model_class]
        self.tissue_vocab = OmegaConf.load(base_dir / "vocabs" / "tissue_vocab.yaml")
        self.vcf_loader_config = OmegaConf.load(base_dir / "configs" / "vcfloader.yaml")

        self.gene_cre_manifest = GeneManifestLookup()

        # Resolve all paths relative to project root
        if not os.path.isabs(self.vcf_loader_config.CRE_BED):
            self.vcf_loader_config.CRE_BED = str(
                base_dir / self.vcf_loader_config.CRE_BED
            )
        if not os.path.isabs(self.vcf_loader_config.fasta_path):
            self.vcf_loader_config.fasta_path = str(
                base_dir / self.vcf_loader_config.fasta_path
            )
        if not os.path.isabs(self.model_config.dataset.gencode_v24):
            self.model_config.dataset.gencode_v24 = str(
                base_dir / self.model_config.dataset.gencode_v24
            )
        if not os.path.isabs(self.model_config.model.checkpoint_path):
            self.model_config.model.checkpoint_path = str(
                base_dir / self.model_config.model.checkpoint_path
            )
        if not os.path.isabs(self.model_config.model.cre_tokenizer.path):
            self.model_config.model.cre_tokenizer.path = str(
                base_dir / self.model_config.model.cre_tokenizer.path
            )
        if not os.path.isabs(self.model_config.model.gene_tokenizer.path):
            self.model_config.model.gene_tokenizer.path = str(
                base_dir / self.model_config.model.gene_tokenizer.path
            )

        assert torch.cuda.is_available(), "GPU is not available"
        self.accelerator = "gpu"

    def get_tissues(self):
        return self.tissue_vocab.keys()

    def get_genes(self):
        df = pd.read_csv(self.model_config.dataset.gencode_v24)
        return df

    def create_data(self, vcf_path: str, query_df: pd.DataFrame, **kwargs):
        dataloader_config = self.vcf_loader_config.dataloader
        dataloader_config.update(kwargs)  # update the dataloader config with the kwargs
        vcf_dataset = VCFDataset(
            max_length=self.model_config.dataset.max_length,
            max_chunks=self.model_config.dataset.max_chunks,
            cre_neighbour_hood=self.model_config.dataset.cre_neighbour_hood,
            gencode_v24=self.model_config.dataset.gencode_v24,
            gene_cre_manifest=self.gene_cre_manifest,
            gene_upstream_neighbour_hood=self.model_config.dataset.gene_upstream_neighbour_hood,
            gene_downstream_neighbour_hood=self.model_config.dataset.gene_downstream_neighbour_hood,
            query_df=query_df,
            fasta_path=self.vcf_loader_config.fasta_path,
            vcf_path=vcf_path,
        )
        dataloader = DataLoader(
            vcf_dataset, collate_fn=collate_fn_batching, **dataloader_config
        )  # create the dataloader
        return vcf_dataset, dataloader

    def load_model(self):
        model_manager = ModelManager(self.model_config.model)
        model, checkpoint_path = model_manager.load_model()
        trainer = Trainer(
            accelerator=self.accelerator,
            devices=1,
            logger=False,
            precision=self.model_config.model.precision,
            enable_checkpointing=False,
        )
        return model, checkpoint_path, trainer

    def predict(self, model, checkpoint_path, trainer, dataloader, vcf_dataset):
        predictions = trainer.predict(model, dataloader, ckpt_path=checkpoint_path)
        query_df = vcf_dataset.query_df
        output_df = self.format_output(query_df, predictions)
        return output_df

    def format_output(self, df, predictions):
        pred_exp = []
        embd = []
        for i in range(len(predictions)):
            pred_exp.extend(predictions[i]["pred_gene_exp"])
            embd.extend(predictions[i]["embeddings"])
        pred_df = pd.DataFrame({"predicted_expression": pred_exp, "embeddings": embd})
        assert len(df) == len(pred_df), "DataFrame and predictions length mismatch"
        df["predicted_expression"] = pred_df["predicted_expression"]
        df["embeddings"] = pred_df["embeddings"]
        return df


def test_vcf_processor():
    """Test VCFProcessor functionality with sample data."""
    vcf_processor = VCFProcessor()

    # Test getting tissues and genes
    tissues = vcf_processor.get_tissues()
    genes = vcf_processor.get_genes()
    log.info("Available tissues:")
    log.info(tissues)
    log.info("Available genes first 5:")
    log.info(genes.head())

    # Test data creation and prediction
    vcf_path = "/app/_artifacts/HG00096.vcf.gz"
    query_df = {
        "gene_id": ["ENSG00000000457.13"] * 2,
        "tissues": ["whole blood,K562,thyroid,artery - aorta"] * 2,
    }
    query_df = pd.DataFrame(query_df)

    vcf_dataset, dataloader = vcf_processor.create_data(vcf_path, query_df)
    model, checkpoint_path, trainer = vcf_processor.load_model()
    predictions = vcf_processor.predict(
        model, checkpoint_path, trainer, dataloader, vcf_dataset
    )

    log.info(predictions)
    log.info("Finished predictions")

    # Basic assertions to validate the test
    assert tissues is not None, "Tissues should not be None"
    assert len(genes) > 0, "Genes dataframe should not be empty"
    assert predictions is not None, "Predictions should not be None"
    assert len(predictions) == len(
        query_df
    ), "Predictions length should match query dataframe length"


if __name__ == "__main__":
    test_vcf_processor()

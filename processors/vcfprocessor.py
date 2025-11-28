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
    def create_vcf_from_variant(self, variant_df: pd.DataFrame, output_path: str, vcf_path: str = None):
        """
        Create a VCF file from a variant dataframe.
        Args:
            variant_df: DataFrame with columns: chrom, pos, ref, alt, GT
            output_path: Path to output VCF file (will be compressed as .vcf.gz)
            vcf_path: Optional path to existing VCF to merge with
        """
        import subprocess
        import tempfile
        from pathlib import Path
        
        assert 'chrom' in variant_df.columns, "chrom column is required"
        assert 'pos' in variant_df.columns, "pos column is required"
        assert 'ref' in variant_df.columns, "ref column is required"
        assert 'alt' in variant_df.columns, "alt column is required"
        assert 'GT' in variant_df.columns, "GT column is required"
        
        if len(variant_df) == 0:
            raise ValueError("variant_df is empty")
        
        # Get fasta path from config
        fasta_path = self.vcf_loader_config.fasta_path
        
        # Validate reference alleles
        log.info("Validating reference alleles...")
        for idx, row in variant_df.iterrows():
            chrom = row['chrom']
            pos = int(row['pos'])
            ref = row['ref']
            ref_len = len(ref)
            
            # Extract reference sequence from fasta
            region = f"{chrom}:{pos}-{pos + ref_len - 1}"
            cmd = ["samtools", "faidx", fasta_path, region]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise ValueError(f"Failed to extract reference at {region}: {result.stderr}")
            
            # Parse fasta output (skip header line)
            fasta_ref = "".join(result.stdout.strip().split("\n")[1:]).upper()
            
            if fasta_ref != ref.upper():
                raise ValueError(
                    f"Reference mismatch at {chrom}:{pos}: "
                    f"expected '{ref}' but found '{fasta_ref}' in reference genome"
                )
        
        log.info("Reference validation successful")
        
        # Create temporary VCF file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', dir=output_dir, delete=False) as tmp_vcf:
            tmp_vcf_path = tmp_vcf.name
            
            # Write VCF header
            tmp_vcf.write("##fileformat=VCFv4.2\n")
            tmp_vcf.write(f"##reference={fasta_path}\n")
            
            # Write contig headers for chromosomes in the dataframe
            for chrom in sorted(variant_df['chrom'].unique()):
                tmp_vcf.write(f"##contig=<ID={chrom}>\n")
            
            tmp_vcf.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            tmp_vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
            
            # Sort variants by chromosome and position
            sorted_df = variant_df.sort_values(by=['chrom', 'pos']).reset_index(drop=True)
            
            # Write variant records
            for _, row in sorted_df.iterrows():
                chrom = row['chrom']
                pos = int(row['pos'])
                ref = row['ref']
                alt = row['alt']
                gt = row['GT']
                
                # Write VCF line with default values
                line = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t{gt}\n"
                tmp_vcf.write(line)
        
        log.info(f"Created temporary VCF: {tmp_vcf_path}")
        
        # Handle merging or direct output
        try:
            if vcf_path is None:
                # No merging needed, just compress and index
                log.info("Creating new VCF file...")
                final_path = output_path if output_path.endswith('.vcf.gz') else f"{output_path}.vcf.gz"
                
                # Compress with bgzip
                subprocess.run(["bgzip", "-c", tmp_vcf_path], 
                             stdout=open(final_path, 'wb'), check=True)
                
                # Index with tabix
                subprocess.run(["tabix", "-p", "vcf", final_path], check=True)
                
                log.info(f"Created and indexed VCF: {final_path}")
            else:
                # Merge with existing VCF - insert variants into same sample
                log.info(f"Merging with existing VCF: {vcf_path}")
                
                # Extract sample name from original VCF
                sample_cmd = ["bcftools", "query", "-l", vcf_path]
                result = subprocess.run(sample_cmd, capture_output=True, text=True, check=True)
                original_sample = result.stdout.strip()
                
                # Update tmp VCF to use the same sample name
                with open(tmp_vcf_path, 'r') as f:
                    vcf_content = f.read()
                vcf_content = vcf_content.replace('\tSAMPLE\n', f'\t{original_sample}\n')
                with open(tmp_vcf_path, 'w') as f:
                    f.write(vcf_content)
                
                # Compress and index new VCF
                sorted_vcf_path = tmp_vcf_path.replace('.vcf', '.sorted.vcf.gz')
                subprocess.run(["bcftools", "sort", "-o", sorted_vcf_path, "-O", "z", tmp_vcf_path], check=True)
                subprocess.run(["tabix", "-p", "vcf", sorted_vcf_path], check=True)
                
                # Concatenate VCFs (same sample, new variants)
                final_path = output_path if output_path.endswith('.vcf.gz') else f"{output_path}.vcf.gz"
                concat_cmd = [
                    "bcftools", "concat",
                    "-a",  # Allow duplicates
                    "-D",  # Remove duplicate positions
                    "-o", final_path,
                    "-O", "z",
                    vcf_path,
                    sorted_vcf_path
                ]
                subprocess.run(concat_cmd, check=True)
                
                # Sort and index final VCF
                temp_sorted = final_path.replace('.vcf.gz', '.temp.sorted.vcf.gz')
                subprocess.run(["bcftools", "sort", "-o", temp_sorted, "-O", "z", final_path], check=True)
                subprocess.run(["mv", temp_sorted, final_path], check=True)
                subprocess.run(["tabix", "-p", "vcf", final_path], check=True)
                
                log.info(f"Concatenated and indexed VCF: {final_path}")
                
                # Clean up sorted temp file
                os.remove(sorted_vcf_path)
                if os.path.exists(sorted_vcf_path + ".tbi"):
                    os.remove(sorted_vcf_path + ".tbi")
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_vcf_path):
                os.remove(tmp_vcf_path)
        
        return final_path if output_path.endswith('.vcf.gz') else f"{output_path}.vcf.gz"


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
        # log.warn('Generating the dataloader with num_workers=0 for debugging')
        # dataloader_config['num_workers'] = 0
        # dataloader_config['batch_size'] = 2
        # del dataloader_config['prefetch_factor']  # Remove prefetch_factor to avoid issues with num_workers=0
        # import ipdb; ipdb.set_trace()
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

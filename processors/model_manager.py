import os
import time
import torch
import mlflow
from omegaconf import OmegaConf
from typing import Tuple, Optional
from seq2gene.model import Seq2GenePredictor
from seq2reg.model import Seq2RegPredictor
from seq2gene.model_combined_modulator import Seq2GenePredictorCombinedModulator

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ModelManager:
    """Handles model loading and inference operations"""

    def __init__(self, config: OmegaConf):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("http://mlflow-api.mlflow.svc.cluster.local:5000")

    def _download_with_retry(self, download_func, timeout: int = 600) -> bool:
        """Helper function to download with retry logic"""
        start_time = time.time()
        while True:
            try:
                download_func()
                return True
            except Exception as e:
                if time.time() - start_time > timeout:
                    log.info(f"Download failed after {timeout}s: {e}")
                    return False
                time.sleep(1)

    def _load_seq2reg(self, train_cfg: OmegaConf) -> Seq2RegPredictor:
        """Load Seq2Reg model"""
        chk = torch.load(
            train_cfg.cre_tokenizer.path, map_location="cpu", weights_only=False
        )
        seq2reg = Seq2RegPredictor(**chk["hyper_parameters"])
        seq2reg.load_state_dict(chk["state_dict"])
        return seq2reg

    def _load_seq2reg_gene(self, train_cfg: OmegaConf) -> Optional[Seq2RegPredictor]:
        """Load Seq2Reg gene model (optional)"""
        chk = torch.load(
            train_cfg.gene_tokenizer.path, map_location="cpu", weights_only=False
        )
        seq2reg_gene = Seq2RegPredictor(**chk["hyper_parameters"])
        seq2reg_gene.load_state_dict(chk["state_dict"])
        return seq2reg_gene

    def _download_seq2gene_checkpoint(self, run_id: str, output_path: str) -> str:
        """Download Seq2Gene checkpoint"""
        # Get latest checkpoint
        epoch_paths = mlflow.artifacts.list_artifacts(
            run_id=run_id, artifact_path="checkpoints"
        )
        epoch_paths.sort(key=lambda x: int(x.path.split("_")[-1]))
        latest_epoch = epoch_paths[-1].path

        log.info(f"Downloading checkpoint from {latest_epoch}")
        artifact_path = f"{latest_epoch}/checkpoint.pth"

        def download():
            return mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=f"{output_path}/ckpts",
            )

        checkpoint_path = None
        success = self._download_with_retry(
            lambda: setattr(self, "_temp_checkpoint_path", download())
        )
        if success:
            checkpoint_path = self._temp_checkpoint_path
            delattr(self, "_temp_checkpoint_path")

        if checkpoint_path is None:
            raise RuntimeError("Failed to download Seq2Gene checkpoint")

        return checkpoint_path

    def load_model(self) -> Tuple[Seq2GenePredictor, str]:
        """Load the complete Seq2Gene model"""
        train_cfg = self.config.copy()

        # Load Seq2Reg models
        log.info("Loading Seq2Reg model...")
        seq2reg = self._load_seq2reg(train_cfg)

        log.info("Loading Seq2Reg gene model...")
        seq2reg_gene = self._load_seq2reg_gene(train_cfg)

        delattr(train_cfg, "cre_tokenizer")
        delattr(train_cfg, "gene_tokenizer")

        # Configure model parameters
        train_cfg.token_dim = seq2reg.hparams.embedding_dim

        # Map model class names to actual classes
        model_classes = {
            "Seq2GenePredictor": Seq2GenePredictor,
            "Seq2GenePredictorCombinedModulator": Seq2GenePredictorCombinedModulator,
        }
        model_class = model_classes[train_cfg.get("model_class", "Seq2GenePredictor")]
        # Create Seq2Gene model
        log.info("Creating Seq2Gene model...")
        gene_model = model_class(
            cre_tokenizer=seq2reg, gene_tokenizer=seq2reg_gene, **train_cfg
        )

        log.info(f"Model class: {model_class}")
        # Log model abstraction (layer/module structure)
        log.info("Model architecture:")
        log.info(f"Model: {gene_model.__class__.__name__}")
        for name, module in gene_model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            log.info(f"  {name}: {num_params:,} params")
        # Log total number of parameters
        total_params = sum(p.numel() for p in gene_model.parameters())
        log.info(f"Total number of parameters: {total_params:,}")
        # Load checkpoint
        checkpoint_path = self.config.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise ValueError("Checkpoint not found")

        log.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load the state dict into the model
        if "state_dict" in checkpoint:
            gene_model.load_state_dict(checkpoint["state_dict"])
        else:
            gene_model.load_state_dict(checkpoint)

        # Setup for inference
        gene_model.eval()
        gene_model.to(self.device)
        gene_model.vep = False  # A model to get all gene expression for all tissues present in the batch

        log.info(f"Model loaded successfully on {self.device}")
        return gene_model, checkpoint_path

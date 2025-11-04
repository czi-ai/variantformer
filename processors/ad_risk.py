#!/usr/bin/env python
"""Predicts alzheimer's disease risk on gene/tissue based given an input genome sequence"""

import pandas as pd
from pathlib import Path
import numpy as np
import treelite
from utils.assets import GeneTissueManifestLookup
from typing import List


from processors import vcfprocessor

_REPO_ROOT = Path(__file__).parent.parent.resolve()
VF_DIMS = 1536
PATH_TO_AD_PREDICTORS = _REPO_ROOT / "_artifacts/predictions_variantformer"
TISSUE_MAP_GUID = "be73e19a"


class ADrisk:
    def __init__(self, gene_id: str, tissue_id: int, model_class: str = "v4_pcg"):
        """
        Initialize the ADrisk predictor.

        Args:
            gene_id (str): Gene ID.
            tissue_id (str): Tissue ID.
            model_class (str): v4_pcg
        """
        assert model_class in [
            "v4_ag", # not yet available on the public bucket
            "v4_pcg",
        ], "model_class should be either 'v4_ag' or 'v4_pcg'"
        assert type(tissue_id) is int, "tissue_id should be an integer"
        assert type(gene_id) is str, "gene_id should be a string"
        self.gene_id = gene_id
        self.tissue_id = tissue_id
        self.model_class = model_class
        self.gene_tissue_manifest = GeneTissueManifestLookup(model_class=self.model_class)
        self.ad_preds = self._load_ad_predictor()

    def __call__(self, gene_tissue_embeds: np.ndarray) -> pd.DataFrame:
        """
        Predict Alzheimer's disease risk for the given gene/tissue embeddings.

        Args:
            gene_tissue_embeds (np.ndarray): Array with gene/tissue embeddings. shape: (num_samples, embedding_dim)
        Returns:
            preds_probas (np.array): Array with AD risk predictions. shape: (num_samples,)
        """
        preds_proba = treelite.gtil.predict(self.predictor, gene_tissue_embeds)
        # preds_proba shape (n_samples, 1, 2) where last dim is [prob_no_ad, prob_ad]
        return preds_proba[:, 0, 1]

    def _load_ad_predictor(self):
        """
        Load AD predictors from the specified directory.
        """
        predictor_fname = self.gene_tissue_manifest.get_file_path(
            self.gene_id, self.tissue_id
        )
        if predictor_fname is None:
            raise FileNotFoundError(
                f"AD predictor not found for gene {self.gene_id} and tissue {self.tissue_id}"
            )
        self.predictor = treelite.Model.deserialize(predictor_fname)


class ADriskFromVCF:
    def __init__(self, model_class: str = "v4_pcg"):
        """
        Initialize the ADrisk predictor.

        Args:
            model_class (str): The model class to use for predictions.
        """
        self.model_class = model_class
        self.ad_preds = GeneTissueManifestLookup(model_class=self.model_class)
        self._init_model()

    def _init_model(self):
        """
        Initialize the VCF processor, tissue and gene mappings, and load the prediction model.
        """
        self.vcf_processor = vcfprocessor.VCFProcessor(model_class=self.model_class)
        tissues_dict = self.vcf_processor.tissue_vocab
        self.tissue_map = pd.DataFrame(
            {"tissue": list(tissues_dict.keys())},
            index=pd.Index(list(tissues_dict.values()), name="tissue_id"),
        )
        self.genes_map = self.vcf_processor.get_genes()
        self.genes_map.set_index("gene_id", inplace=True)
        self.model, self.checkpoint_path, self.trainer = self.vcf_processor.load_model()

    def _init_dataloader(
        self, vcf_path: str, gene_ids: List[str], tissue_ids: List[int]
    ):
        """
        Initialize the dataloader for the given VCF file, gene IDs, and tissue IDs.

        Args:
            vcf_path (str): Path to the VCF file.
            gene_ids (List[str]): List of gene IDs.
            tissue_ids (List[int]): List of tissue IDs.

        Returns:
            Tuple: vcf_dataset and dataloader objects.
        """
        query = self._format_query(gene_ids, tissue_ids)
        vcf_dataset, dataloader = self.vcf_processor.create_data(vcf_path, query)
        return vcf_dataset, dataloader

    def _format_query(self, gene_ids: List[str], tissue_ids: List[int]):
        """
        Format the query DataFrame for the VCF processor.

        Args:
            gene_ids (List[str]): List of gene IDs.
            tissue_ids (List[int]): List of tissue IDs.

        Returns:
            pd.DataFrame: Query DataFrame mapping gene IDs to tissue names.
        """
        assert (
            len(gene_ids) == len(tissue_ids)
        ), "Please map gene ids to tissue ids, there should be 2 lists of the same length"
        for gene_id, tissue_id in zip(gene_ids, tissue_ids):
            assert type(tissue_id) is int, "tissue_id should be an integer"
            assert type(gene_id) is str, "gene_id should be a string"
        query = {
            "gene_id": gene_ids,
            "tissues": self.tissue_map.loc[tissue_ids]["tissue"].tolist(),
        }
        return pd.DataFrame(query)

    def __call__(
        self, vcf_file: str, gene_ids: List[str], tissue_ids: List[int]
    ) -> pd.DataFrame:
        """
        Predict Alzheimer's disease risk for the given VCF file, gene IDs, and tissue IDs.

        Args:
            vcf_file (str): Path to the VCF file.
            gene_ids (List[str]): List of gene IDs.
            tissue_ids (List[int]): List of tissue IDs.

        Returns:
            pd.DataFrame: DataFrame with AD risk predictions.
        """
        vcf_dataset, dataloader = self._init_dataloader(vcf_file, gene_ids, tissue_ids)
        preds_df = self.vcf_processor.predict(
            self.model, self.checkpoint_path, self.trainer, dataloader, vcf_dataset
        )
        preds_df = self._reformat_predictions(preds_df)
        return self._predict_ad_risk(preds_df)

    def _predict_ad_risk(self, preds_df):
        """
        Predict AD risk using pre-trained predictors for each gene/tissue pair.

        Args:
            preds_df (pd.DataFrame): DataFrame with gene/tissue embeddings.

        Returns:
            pd.DataFrame: DataFrame with added AD risk predictions.
        """
        all_preds_probas = []
        for _, row in preds_df.iterrows():
            predictor_fname = self.ad_preds.get_file_path(row.gene_id, row.tissue_id)
            predictor = treelite.Model.deserialize(predictor_fname)
            preds_proba = treelite.gtil.predict(predictor, row.embedding)
            preds_proba = np.squeeze(preds_proba)[1]
            all_preds_probas.append(preds_proba)
        preds_df["ad_risk"] = all_preds_probas
        preds_df["gene_name"] = preds_df["gene_id"].map(self.genes_map["gene_name"])
        return preds_df

    def _reformat_predictions(self, preds_df):
        """
        Reformat the predictions DataFrame to standardize column names and structure.

        Args:
            preds_df (pd.DataFrame): Raw predictions DataFrame.

        Returns:
            pd.DataFrame: Reformatted predictions DataFrame.
        """
        preds_df.rename(
            {
                "tissues": "tissue_id",
                "tissue_names": "tissue_name",
                "embeddings": "embedding",
            },
            axis=1,
            inplace=True,
        )
        assert preds_df.tissue_id.apply(lambda tissue_id: len(tissue_id) == 1).all()
        preds_df.tissue_id = preds_df.tissue_id.apply(lambda x: x[0])
        preds_df.tissue_name = preds_df.tissue_name.apply(lambda x: x[0])
        preds_df.predicted_expression = preds_df.predicted_expression.apply(
            lambda x: x[0]
        )
        preds_df.predicted_expression = preds_df.predicted_expression.apply(
            lambda x: x[0]
        )
        return preds_df

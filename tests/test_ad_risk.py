import unittest
import sys
from pathlib import Path
import pandas as pd
# Add parent directory to path
CURRENT_PATH = Path(__file__).parent
sys.path.insert(0, str(CURRENT_PATH.parent))

from processors.vcfprocessor import VCFProcessor
from processors import ad_risk

ARTIFACTS_DIR = "/tmp/variantformer_artifacts"
VCF_EXAMPLE = ARTIFACTS_DIR / "HG00096.vcf.gz"
TISSUE_MAP_PATH = ARTIFACTS_DIR / "be73e19a.pq"
APOE_GENE_ID = 'ENSG00000130203.9'


class TestADrisk(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case with a sample ADrisk instance and test data.
        """
        self.model_class = 'v4_pcg'
        self.gene_id = APOE_GENE_ID
        self.tissue_id = 7
        self.adrisk = ad_risk.ADrisk(self.gene_id, self.tissue_id, model_class=self.model_class)
        self.vcf_path = VCF_EXAMPLE
        self._init_vcf_processor()
        self._init_dataloader()
        self._init_vcf_processor()
    
    def _init_vcf_processor(self):
        self.vcf_processor = VCFProcessor(model_class=self.model_class)
        tissues_dict = self.vcf_processor.tissue_vocab
        self.tissue_map = pd.DataFrame({'tissue': list(tissues_dict.keys())}, index=pd.Index(list(tissues_dict.values()), name='tissue_id'))
        self.genes_map = self.vcf_processor.get_genes()
        self.genes_map.set_index('gene_id', inplace=True)
        self.model, self.checkpoint_path, self.trainer = self.vcf_processor.load_model()
    
    def _init_dataloader(self):
        self.tissue_map = pd.read_parquet(TISSUE_MAP_PATH)
        tissue_name = self.tissue_map.loc[self.tissue_id, 'tissue']
        self.query_df = pd.DataFrame({'gene_id': [self.gene_id], 'tissues': [tissue_name]})
        self.vcf_dataset, self.dataloader = self.vcf_processor.create_data(self.vcf_path, self.query_df)

    def test_1(self) -> None:
        """
        Test the ADrisk prediction from gene embedding
        """
        model_outputs = self.vcf_processor.predict(self.model, self.checkpoint_path, self.trainer, self.dataloader, self.vcf_dataset)
        gene_tissue_embeds = model_outputs['embeddings'].iloc[0]
        preds = self.adrisk(gene_tissue_embeds)
        self.assertAlmostEqual(preds[0], 0.66763765, places=1) # tested on h100, dev coreweave on 10/31/25


class TestADriskFromVCF(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case with a sample ADrisk instance and test data.
        """
        self.adrisk = ad_risk.ADriskFromVCF()
        self.vcf_path = VCF_EXAMPLE
        self.gene_ids = [APOE_GENE_ID] * 2
        self.tissue_ids = [7, 47] # run it on 2 tissues

    def test_1(self) -> None:
        """
        Test the ADrisk prediction pipeline with an input VCF file
        """
        preds = self.adrisk(self.vcf_path, self.gene_ids, self.tissue_ids)
        self.assertAlmostEqual(preds.iloc[0].ad_risk.item(), 0.66763765, places=1) # tested on h100, dev coreweave on 10/31/25


if __name__ == "__main__":
    unittest.main()
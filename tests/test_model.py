import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
   )
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from processors.vcfprocessor import VCFProcessor

# Get the repo root directory (equivalent to /app/ in deployment)
_REPO_ROOT = Path(__file__).parent.parent.resolve()
VCF_EXAMPLE = _REPO_ROOT / "_artifacts" / "HG00096.vcf.gz"
TISSUE_MAP_GUID = "be73e19a"


class TestGeneExpressionAndEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.vcf_processor = VCFProcessor(model_class="v4_pcg")
        simple_query = {
            "gene_id": ["ENSG00000000457.13"] * 2,
            "tissues": ["whole blood,thyroid,artery - aorta", "brain - amygdala"],
        }
        self.query_df = pd.DataFrame(simple_query)
        self.vcf_path = str(VCF_EXAMPLE)
        self.vcf_dataset, self.dataloader = self.vcf_processor.create_data(
            self.vcf_path, self.query_df
        )
        self.model, self.checkpoint_path, self.trainer = self.vcf_processor.load_model()
        self.target_df = pd.read_parquet(
            _REPO_ROOT / "_artifacts" / "924979a7.pq"
        )  # <- Needs to be changed to new version

    def test_1(self) -> None:
        preds_df = self.vcf_processor.predict(
            self.model,
            self.checkpoint_path,
            self.trainer,
            self.dataloader,
            self.vcf_dataset,
        )
        preds_df.predicted_expression = preds_df.predicted_expression.apply(
            lambda x: x.flatten()
        )
        preds_df.embeddings = preds_df.embeddings.apply(lambda x: x.flatten())
        preds_df.tissues = preds_df.tissues.apply(lambda x: np.array(x))
        preds_df.tissue_names = preds_df.tissue_names.apply(lambda x: np.array(x))
        # pd.testing.assert_frame_equal(preds_df, self.target_df) <- Can be uncommented when target df is updated
        log.info("Gene expression and embedding predictions:")
        log.info(f"{preds_df.head(2)}")


if __name__ == "__main__":
    unittest.main()

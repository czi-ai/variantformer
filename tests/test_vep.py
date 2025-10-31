import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from processors.variantprocessor import VariantProcessor
import tempfile
import shutil
import stash as st


_REPO_ROOT = Path(__file__).parent.parent.resolve()
VCF_EXAMPLE = _REPO_ROOT / "_artifacts" / "HG00096.vcf.gz"


class Test(unittest.TestCase):
    def setUp(self) -> None:
        # Load target predictions for regression testing
        self.target_predictions = np.load(_REPO_ROOT / "_artifacts" / "befd2388.npz")
        model_class = "D2C_PCG"
        # model_class = "D2C_AG"
        self.processor = VariantProcessor(model_class=model_class)
        print("Model class: ", model_class)
        # Create test data similar to the notebook
        self.test_data = {
            "chr": ["chr13"],
            "pos": [113978728],
            "ref": ["A"],
            "alt": ["G"],
            "tissue": ["whole blood"],
            "gene_id": ["ENSG00000185989.10"],
        }
        self.test_df = pd.DataFrame(self.test_data)
        self.temp_dir = tempfile.mkdtemp()
        self.vcf_df = pd.read_parquet(_REPO_ROOT / "_artifacts" / "f9bbc0ba.pq")
        self.sample_name = self.vcf_df["name"].iloc[0]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1(self) -> None:
        with tempfile.TemporaryDirectory() as raw_test_dir:
            raw_predictions = self.processor.predict(
                var_df=self.test_df, output_dir=raw_test_dir
            )

        with tempfile.TemporaryDirectory() as vcf_test_dir:
            vcf_predictions = self.processor.predict(
                var_df=self.test_df,
                output_dir=vcf_test_dir,
                vcf_path=str(VCF_EXAMPLE),
                sample_name=self.sample_name,
            )

        # Compare predictions
        vcf_exp = vcf_predictions[
            (vcf_predictions["sample_name"] == self.sample_name)
            & (vcf_predictions["zygosity"] == "2")
        ]["gene_exp"]
        raw_exp = raw_predictions[
            (raw_predictions["sample_name"] == self.sample_name)
            & (raw_predictions["zygosity"] == "2")
        ]["gene_exp"]
        print(
            vcf_predictions[
                (vcf_predictions["sample_name"] == self.sample_name)
                & (vcf_predictions["zygosity"] == "2")
            ]
        )
        print(
            raw_predictions[
                (raw_predictions["sample_name"] == self.sample_name)
                & (raw_predictions["zygosity"] == "2")
            ]
        )
        self.assertTrue(
            np.allclose(vcf_exp, raw_exp, atol=1),
            f"VCF predictions: {vcf_exp} do not match raw predictions: {raw_exp}",
        )

        vcf_exp = vcf_predictions[
            (vcf_predictions["sample_name"] == self.sample_name)
            & (vcf_predictions["zygosity"] == "1")
        ]["gene_exp"]
        raw_exp = raw_predictions[
            (raw_predictions["sample_name"] == self.sample_name)
            & (raw_predictions["zygosity"] == "1")
        ]["gene_exp"]
        self.assertTrue(
            np.allclose(vcf_exp, raw_exp, atol=1),
            f"VCF predictions: {vcf_exp} do not match raw predictions: {raw_exp}",
        )
    '''
    def test_2(self) -> None:
        print("checkpoint 0")
        # Initialize the processor and get raw predictions for comparison with target
        vep_dataset, dataloader, model, trainer, ckpt_path = self.processor.initialize(
            var_df=self.test_df, output_dir=self.temp_dir
        )

        print("checkpoint 1")
        # Get raw predictions for comparison with target predictions
        raw_predictions = trainer.predict(
            model=model, dataloaders=dataloader, ckpt_path=ckpt_path
        )

        print("checkpoint 2")
        for key in raw_predictions[0].keys():
            cur_pred = np.array(raw_predictions[0][key])
            target_pred = self.target_predictions[key]
            print(f"Checking predictions for key: {key}")
            np.testing.assert_allclose(cur_pred, target_pred)

        print("checkpoint 3")
        # Compile predictions to DataFrame and validate format
        predictions_df = self.processor.compile_predictions(raw_predictions)
        # Clean up resources
        self.processor.cleanup()

        print("checkpoint 4")
        # Basic assertions to verify the DataFrame format
        self.assertIsInstance(predictions_df, pd.DataFrame)
        self.assertGreater(len(predictions_df), 0)

        # Check that required columns are present
        expected_columns = [
            "chrom",
            "pos",
            "ref",
            "alt",
            "genes",
            "tissues",
            "variant_type",
            "population",
            "sample_name",
            "zygosity",
            "gene_exp",
        ]
        for col in expected_columns:
            self.assertIn(col, predictions_df.columns)

        # Verify the test data is reflected in predictions
        self.assertTrue((predictions_df["chrom"] == "chr13").any())
        self.assertTrue((predictions_df["pos"] == 113978728).any())
        self.assertTrue((predictions_df["ref"] == "A").any())
        self.assertTrue((predictions_df["alt"] == "G").any())
        self.assertTrue((predictions_df["tissues"] == "whole blood").any())

        # Verify gene expression predictions are numeric
        self.assertTrue(np.issubdtype(predictions_df["gene_exp"].dtype, np.number))

        print("✅ Regression test passed! Raw predictions match target predictions.")
        print(f"✅ Format test passed! Generated {len(predictions_df)} prediction rows")
        print(f"Predictions shape: {predictions_df.shape}")
        print(f"Sample predictions:\n{predictions_df.head()}")
        '''

class TestVariantProcessor(unittest.TestCase):
    def setUp(self) -> None:
        # Load target predictions for regression testing
        model_class = "D2C_PCG"
        # model_class = "D2C_AG"
        self.processor = VariantProcessor(model_class=model_class)
        self.processor_ag = VariantProcessor(model_class = 'D2C_AG')
        print("Model class: ", model_class)
        self.test_df = st.get('a0063c48')
        self.test_df['chr'] = self.test_df['variant_id'].apply(lambda x: x.split('_')[0])
        self.test_df['pos'] = self.test_df['variant_id'].apply(lambda x: int(x.split('_')[1]))
        self.test_df = self.test_df
        self.vcf_df = st.get('9a83db58')
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1(self):
        var = self.vcf_df[['chr', 'ref', 'pos', 'alt', 'tissues', 'genes']]
        var = var.rename(columns={'tissues':'tissue', 'genes':'gene_id'})
        # Get raw predictions for comparison with target predictions
        raw_predictions = self.processor.predict(var_df=var, output_dir=self.temp_dir, vcf_path=VCF_EXAMPLE, sample_name='HG00096')
        print("checkpoint 2.1")
        # Compile predictions to DataFrame and validate format
        predictions_df = self.processor.format_scores(raw_predictions)
        final_results = self.processor.eqtl_scores(predictions_df)
        assert(all(final_results==self.vcf_df)), f"Regression test failed: Predictions {final_results} do not match target {self.vcf_df}"
    def test_2(self) -> None:
        print("checkpoint 1")
        # Get raw predictions for comparison with target predictions
        raw_predictions = self.processor_ag.predict(var_df=self.test_df, output_dir=self.temp_dir)

        print("checkpoint 2.1")
        # Compile predictions to DataFrame and validate format
        predictions_df = self.processor_ag.format_scores(raw_predictions)

        final_results = self.processor_ag.eqtl_scores(predictions_df)

        sas_score_pred = final_results['D2C-SAS-2-exp-log2fc'].values
        sas_score_target = self.test_df['D2C-SAS-2-log2fc_ag'].values

        self.assertTrue(
            np.allclose(sas_score_pred, sas_score_target, atol=1e-3),
            f"SAS score predictions: {sas_score_pred} do not match target scores: {sas_score_target}",
        )

        print("checkpoint 2.2")
        eur_score_pred = final_results['D2C-EUR-2-exp-log2fc'].values
        eur_score_target = self.test_df['D2C-EUR-2-log2fc_ag'].values

        self.assertTrue(
            np.allclose(eur_score_pred, eur_score_target, atol=1e-3),
            f"EUR score predictions: {eur_score_pred} do not match target scores: {eur_score_target}",
        )
        print("checkpoint 2.3")
        afr_score_pred = final_results['D2C-AFR-2-exp-log2fc'].values
        afr_score_target = self.test_df['D2C-AFR-2-log2fc_ag'].values
        self.assertTrue(
            np.allclose(afr_score_pred, afr_score_target, atol=1e-3),
            f"AFR score predictions: {afr_score_pred} do not match target scores: {afr_score_target}",
        )

        print("checkpoint 2.4")
        amr_score_pred = final_results['D2C-AMR-2-exp-log2fc'].values
        amr_score_target = self.test_df['D2C-AMR-2-log2fc_ag'].values
        self.assertTrue(
            np.allclose(amr_score_pred, amr_score_target, atol=1e-3),
            f"AMR score predictions: {amr_score_pred} do not match target scores: {amr_score_target}",
        )   

        print("checkpoint 2.5")
        eas_score_pred = final_results['D2C-EAS-2-exp-log2fc'].values
        eas_score_target = self.test_df['D2C-EAS-2-log2fc_ag'].values
        self.assertTrue(
            np.allclose(eas_score_pred, eas_score_target, atol=1e-3),
            f"EAS score predictions: {eas_score_pred} do not match target scores: {eas_score_target}",
        )

        print("checkpoint 2.6")
        weighted_score_pred = final_results['D2C-agg-log2fc-weighted'].values
        weighted_score_target = self.test_df['D2C-agg-log2fc-weighted_ag'].values
        self.assertTrue(
            np.allclose(weighted_score_pred, weighted_score_target, atol=1e-3),
            f"Weighted score predictions: {weighted_score_pred} do not match target scores: {weighted_score_target}",
        )

        # Clean up resources
        self.processor.cleanup()

        print("checkpoint 3")
        # Basic assertions to verify the DataFrame format
        self.assertIsInstance(predictions_df, pd.DataFrame)
        self.assertGreater(len(predictions_df), 0)
    def test_3(self) -> None:
        print("checkpoint 1")
        # Get raw predictions for comparison with target predictions
        raw_predictions = self.processor.predict(var_df=self.test_df, output_dir=self.temp_dir)

        print("checkpoint 2.1")
        # Compile predictions to DataFrame and validate format
        predictions_df = self.processor.format_scores(raw_predictions)

        final_results = self.processor.eqtl_scores(predictions_df)

        sas_score_pred = final_results['D2C-SAS-2-exp-log2fc'].values
        sas_score_target = self.test_df['D2C-SAS-2-log2fc_pcg'].values

        self.assertTrue(
            np.allclose(sas_score_pred, sas_score_target, atol=1e-3),
            f"SAS score predictions: {sas_score_pred} do not match target scores: {sas_score_target}",
        )

        print("checkpoint 2.2")
        eur_score_pred = final_results['D2C-EUR-2-exp-log2fc'].values
        eur_score_target = self.test_df['D2C-EUR-2-log2fc_pcg'].values

        self.assertTrue(
            np.allclose(eur_score_pred, eur_score_target, atol=1e-3),
            f"EUR score predictions: {eur_score_pred} do not match target scores: {eur_score_target}",
        )
        print("checkpoint 2.3")
        afr_score_pred = final_results['D2C-AFR-2-exp-log2fc'].values
        afr_score_target = self.test_df['D2C-AFR-2-log2fc_pcg'].values
        self.assertTrue(
            np.allclose(afr_score_pred, afr_score_target, atol=1e-3),
            f"AFR score predictions: {afr_score_pred} do not match target scores: {afr_score_target}",
        )

        print("checkpoint 2.4")
        amr_score_pred = final_results['D2C-AMR-2-exp-log2fc'].values
        amr_score_target = self.test_df['D2C-AMR-2-log2fc_pcg'].values
        self.assertTrue(
            np.allclose(amr_score_pred, amr_score_target, atol=1e-3),
            f"AMR score predictions: {amr_score_pred} do not match target scores: {amr_score_target}",
        )   

        print("checkpoint 2.5")
        eas_score_pred = final_results['D2C-EAS-2-exp-log2fc'].values
        eas_score_target = self.test_df['D2C-EAS-2-log2fc_pcg'].values
        self.assertTrue(
            np.allclose(eas_score_pred, eas_score_target, atol=1e-3),
            f"EAS score predictions: {eas_score_pred} do not match target scores: {eas_score_target}",
        )

        print("checkpoint 2.6")
        weighted_score_pred = final_results['D2C-agg-log2fc-weighted'].values
        weighted_score_target = self.test_df['D2C-agg-log2fc-weighted_pcg'].values
        self.assertTrue(
            np.allclose(weighted_score_pred, weighted_score_target, atol=1e-3),
            f"Weighted score predictions: {weighted_score_pred} do not match target scores: {weighted_score_target}",
        )

        # Clean up resources
        self.processor.cleanup()

        print("checkpoint 3")
        # Basic assertions to verify the DataFrame format
        self.assertIsInstance(predictions_df, pd.DataFrame)
        self.assertGreater(len(predictions_df), 0)


   
if __name__ == "__main__":
    unittest.main()

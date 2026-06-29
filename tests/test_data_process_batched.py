import shutil
import sys
import unittest
from pathlib import Path
import pandas as pd
# Add parent directory to path
CURRENT_PATH = Path(__file__).parent
sys.path.insert(0, str(CURRENT_PATH.parent))

from utils.data_process import ExtractSeqFromBed


ARTIFACTS_DIR = CURRENT_PATH.parent / "_artifacts"
VCF_EXAMPLE = ARTIFACTS_DIR / "HG00096.vcf.gz"
MODEL_CLASS = "v4_ag"
GENE_IDS = ["ENSG00000001461.16", "ENSG00000000419.12"] # used in notebooks/vcf2exp.ipynb
MAX_REGIONS_PER_GENE = 250 # cap to keep test tractable

_IUPAC_AMBIGUOUS = set("RYSWKMBDHVN")
_VALID_BASES = set("ACGTN") | _IUPAC_AMBIGUOUS

REQUIRED_TOOLS = ("samtools", "bcftools")
_MISSING_TOOLS = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]


@unittest.skipIf(
    _MISSING_TOOLS, f"Required tools not on PATH: {', '.join(_MISSING_TOOLS)}"
)
@unittest.skipUnless(VCF_EXAMPLE.exists(), f"Missing artifact: {VCF_EXAMPLE}")
class TestBatchedConsensusEquivalenceRealVCF(unittest.TestCase):
    """Batched extraction must equal per-region extraction on HG00096."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from processors.vcfprocessor import VCFProcessor

            cls.vcf_processor = VCFProcessor(model_class=MODEL_CLASS)
        except Exception as exc:
            raise unittest.SkipTest(
                f"VCFProcessor unavailable (needs CUDA + artifacts): {exc}"
            )

        cls.fasta_path = str(cls.vcf_processor.vcf_loader_config.fasta_path)
        if not Path(cls.fasta_path).exists():
            raise unittest.SkipTest(f"Reference genome not found: {cls.fasta_path}")

        cls.cre_neighbour_hood = cls.vcf_processor.model_config.dataset.cre_neighbour_hood
        cls.vcf_path = str(VCF_EXAMPLE)

        # build CRE bed_regions for each gene, same as VCFDataset
        cls.bed_by_gene = {}
        for gene_id in GENE_IDS:
            gene_cre_map_path = cls.vcf_processor.gene_cre_manifest.get_file_path(gene_id)
            if gene_cre_map_path is None:
                continue
            genes_cre_map = pd.read_csv(gene_cre_map_path)
            bed_regions = genes_cre_map[
                ["chromosome", "start_cre", "end_cre", "cre_name"]
            ].rename(
                columns={
                    "chromosome": "chrom",
                    "start_cre": "start",
                    "end_cre": "end",
                    "cre_name": "cCRE",
                }
            )
            bed_regions = bed_regions.head(MAX_REGIONS_PER_GENE).reset_index(drop=True)
            if len(bed_regions) > 0:
                cls.bed_by_gene[gene_id] = bed_regions

        if not cls.bed_by_gene:
            raise unittest.SkipTest(
                "No CRE maps could be resolved for the requested genes."
            )

    def _extractor(self) -> ExtractSeqFromBed:
        return ExtractSeqFromBed(
            neighbour_hood=self.cre_neighbour_hood, ref_fasta=self.fasta_path
        )

    def _per_region_sequences(self, bed_regions, vcf_file, variant_type=None) -> dict:
        """Ground truth: the original one-bcftools-call-per-region path."""
        extractor = self._extractor()
        expected = {}
        for _, region in bed_regions.iterrows():
            seq, _ = extractor.apply_bcftools_consensus(
                region, vcf_file, self.fasta_path, variant_type=variant_type
            )
            if seq:
                expected[region.cCRE] = seq
        return expected

    @staticmethod
    def _df_to_dict(df) -> dict:
        return {row["cCRE"]: row["sequence"] for _, row in df.iterrows()}

    def test_personalized_matches_per_region(self) -> None:
        for gene_id, bed_regions in self.bed_by_gene.items():
            with self.subTest(gene_id=gene_id):
                expected = self._per_region_sequences(bed_regions, self.vcf_path)
                batched = self._df_to_dict(
                    self._extractor().process_subject(
                        vcf_file=self.vcf_path, bed_regions=bed_regions
                    )
                )
                self.assertEqual(batched, expected)
                # make sure sequences contain only IUPAC codes
                for seq in batched.values():
                    self.assertTrue(set(seq.upper()) <= _VALID_BASES)

    def test_reference_matches_per_region(self) -> None:
        for gene_id, bed_regions in self.bed_by_gene.items():
            with self.subTest(gene_id=gene_id):
                expected = self._per_region_sequences(bed_regions, "") # ref genome
                batched = self._df_to_dict(
                    self._extractor().process_subject(
                        vcf_file="", bed_regions=bed_regions
                    )
                )
                self.assertEqual(batched, expected)
                # there shouldn't be any ambiguity codes in ref
                for seq in batched.values():
                    self.assertFalse(set(seq.upper()) & _IUPAC_AMBIGUOUS)

    def test_snp_only_variant_type_matches_per_region(self) -> None:
        for gene_id, bed_regions in self.bed_by_gene.items():
            with self.subTest(gene_id=gene_id):
                expected = self._per_region_sequences(
                    bed_regions, self.vcf_path, variant_type="SNP"
                )
                batched = self._df_to_dict(
                    self._extractor().process_subject(
                        vcf_file=self.vcf_path,
                        bed_regions=bed_regions,
                        variant_type="SNP",
                    )
                )
                self.assertEqual(batched, expected)

    def test_result_sorted_by_start(self) -> None:
        gene_id, bed_regions = next(iter(self.bed_by_gene.items()))
        df = self._extractor().process_subject(
            vcf_file=self.vcf_path, bed_regions=bed_regions
        )
        self.assertListEqual(
            list(df.columns), ["chrom", "start_cre", "end_cre", "sequence", "cCRE"]
        )
        # ordered by (chrom, start_cre), start_cre increasing
        expected_order = df.sort_values(["chrom", "start_cre"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_order)
        for _, group in df.groupby("chrom"):
            self.assertTrue(group["start_cre"].is_monotonic_increasing)

    def test_serial_fallback_matches_per_region(self) -> None:
        """If the batched call cannot be matched up, the serial fallback is used."""
        gene_id, bed_regions = next(iter(self.bed_by_gene.items()))
        expected = self._per_region_sequences(bed_regions, self.vcf_path)

        extractor = self._extractor()
        # report failure and fall back to per-region serial path
        extractor._extract_sequences_batched = lambda *a, **k: None
        batched = self._df_to_dict(
            extractor.process_subject(vcf_file=self.vcf_path, bed_regions=bed_regions)
        )
        self.assertEqual(batched, expected)


if __name__ == "__main__":
    unittest.main()

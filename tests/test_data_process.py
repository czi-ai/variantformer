import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

CURRENT_PATH = Path(__file__).parent
sys.path.insert(0, str(CURRENT_PATH.parent))

from utils.data_process import ExtractSeqFromBed


class TestExtractSeqFromBed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="variantformer-data-process-"))
        cls.ref_fasta = cls.tmp_dir / "ref.fa"
        cls.ref_fasta.write_text(">chr1\nACGTACGTACGTACGTACGT\n", encoding="ascii")
        subprocess.run(["samtools", "faidx", str(cls.ref_fasta)], check=True)

        raw_vcf = cls.tmp_dir / "variants.vcf"
        raw_vcf.write_text(
            "\n".join(
                [
                    "##fileformat=VCFv4.2",
                    "##contig=<ID=chr1,length=20>",
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
                    "chr1\t5\t.\tA\tG\t.\tPASS\t.\tGT\t1/1",
                    "chr1\t12\t.\tT\tC\t.\tPASS\t.\tGT\t1/1",
                    "",
                ]
            ),
            encoding="ascii",
        )
        cls.vcf_path = cls.tmp_dir / "variants.vcf.gz"
        subprocess.run(
            [
                "bcftools",
                "view",
                "-Oz",
                "--write-index",
                "-o",
                str(cls.vcf_path),
                str(raw_vcf),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_apply_bcftools_consensus_respects_region_coordinates(self):
        extractor = ExtractSeqFromBed(neighbour_hood=0, ref_fasta=str(self.ref_fasta))
        region = SimpleNamespace(chrom="chr1", start=4, end=5, cCRE="cre-1")

        mutated_seq, mutations = extractor.apply_bcftools_consensus(
            region, str(self.vcf_path), str(self.ref_fasta)
        )

        self.assertEqual(mutated_seq, "G")
        self.assertEqual(mutations, 1)

    def test_process_subject_multiple_regions(self):
        extractor = ExtractSeqFromBed(neighbour_hood=0, ref_fasta=str(self.ref_fasta))
        bed_regions = pd.DataFrame(
            [
                {"chrom": "chr1", "start": 4, "end": 5, "cCRE": "cre-1"},
                {"chrom": "chr1", "start": 11, "end": 12, "cCRE": "cre-2"},
            ]
        )

        result = extractor.process_subject(str(self.vcf_path), bed_regions)

        self.assertEqual(result["sequence"].tolist(), ["G", "C"])
        self.assertEqual(result["start_cre"].tolist(), [4, 11])
        self.assertEqual(result["end_cre"].tolist(), [5, 12])


if __name__ == "__main__":
    unittest.main()

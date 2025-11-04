

import warnings
warnings.warn("The tests in test_manifest_lookup.py are currently disabled.", UserWarning)

import unittest
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.assets import CreSequencesManifestLookup, GeneSequencesManifestLookup

import logging
logging.basicConfig(
   level=logging.DEBUG,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
   )
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class TestGeneSequencesManifestLookup(unittest.TestCase):
    def setUp(self) -> None:
        self.gene_seq_lookup = GeneSequencesManifestLookup()
        self.gene_id = "ENSG00000185989.10"
        self.population = "EUR"

    def test_1(self) -> None:
        file_path = self.gene_seq_lookup.get_file_path(self.gene_id, self.population)
        # self.assertIn("model/common/reference_genomes/data/eur/genes/data/ENSG00000185989.10_HG00096.npz", file_path)


class TestCreSequencesManifestLookup(unittest.TestCase):
    def setUp(self) -> None:
        self.cre_seq_lookup = CreSequencesManifestLookup()

    def test_1(self) -> None:
        chromosome = "chr1"
        population = "EUR"
        file_path = self.cre_seq_lookup.get_file_path(chromosome, population)
        # self.assertIn("model/common/reference_genomes/data_split/eur/cres/data_split/HG00096_chr1.pkl.gz", file_path)

    def test_2(self) -> None:
        chromosome = "chr10"
        population = "EUR"
        file_path = self.cre_seq_lookup.get_file_path(chromosome, population)
        # self.assertIn("model/common/reference_genomes/data_split/eur/cres/data_split/HG00096_chr10.pkl.gz", file_path)

    def test_3(self) -> None:
        chromosome = "chr20"
        population = "EUR"
        file_path = self.cre_seq_lookup.get_file_path(chromosome, population)
        # self.assertIn("model/common/reference_genomes/data_split/eur/cres/data_split/HG00096_chr20.pkl.gz", file_path)


if __name__ == "__main__":
    unittest.main()
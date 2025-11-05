#!/usr/bin/env python3
"""
Utilities for retrieving model assets
"""

import dataclasses
import functools
import logging
import os.path
import fsspec
from filelock import FileLock
from pathlib import Path
import duckdb

import logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
   )
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


DEFAULT_BUCKET="czi-variantformer"
ARTIFACTS_DIR = Path(__file__).parent.parent.resolve() / "_artifacts"
GENE_TISSUE_MANIFEST_FILE_PATH = (
    f"s3://{DEFAULT_BUCKET}/alzheimer_disease/<model_class>/manifest.parquet"
)
GENE_CRE_MANIFEST_FILE_PATH = (
    f"s3://{DEFAULT_BUCKET}/model/common/cres_all_genes_manifest.parquet"
)
GENE_SEQUENCES_MANIFEST_FILE_PATH = (
    f"s3://{DEFAULT_BUCKET}/model/common/reference_genomes/genes_seqs_manifest.parquet"
)
CRE_SEQUENCES_MANIFEST_FILE_PATH = (
    f"s3://{DEFAULT_BUCKET}/model/common/reference_genomes/cres_seqs_manifest.parquet"
)
@dataclasses.dataclass
class GeneRecord:
    gene_id: str
    file_path: str

@dataclasses.dataclass
class GeneSequenceRecord:
    gene_id: str
    file_path: str
    population: str

@dataclasses.dataclass
class CreSequenceRecord:
    chromosome: str
    file_path: str
    population: str


@dataclasses.dataclass
class GeneTissueRecord:
    tissue_id: int
    gene_id: str
    file_path: str


class S3CachedFetcher:
    def __init__(self, bucket: str = DEFAULT_BUCKET, 
                 tmp_dir: str = ARTIFACTS_DIR, 
                 aws_credentials: dict = None):
        self.bucket = bucket
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)
        self._aws_credentials = aws_credentials or {}

    def get(self, s3_path: str) -> str:
        """Thread safe s3 downloads/caching of s3 objects

        Args:
            s3_path (str): The S3 path to the object

        Returns:
            dst: The local path to the cached object
        """

        cache_dir = os.path.join(self.tmp_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        fsspec_storage_opts = {"s3": {"anon": True},
                               "simplecache": {"cache_storage": cache_dir}}
        # normalize and namespace by bucket
        rel = os.path.normpath(s3_path).lstrip(os.sep)
        dst = os.path.realpath(os.path.join(self.tmp_dir, rel))
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(dst):
            log.debug(f"Using cached file: {dst}")
            return dst
        
        lock_path = dst + '.lock'
        # Use FileLock context manager which handles lock acquisition/release
        with FileLock(lock_path, timeout=600):  # 10 minute timeout
            # Double-check after acquiring lock (another process may have downloaded it)
            if os.path.exists(dst):
                return dst

            # populate fsspec simplecache; returns a local hashed path in out_dir
            s3_uri = f"simplecache::s3://{self.bucket}/{rel}"
            cached = fsspec.open_local(s3_uri, **fsspec_storage_opts)

            # same filesystem → zero-copy publish via hardlink
            try:
                os.link(cached, dst)            # atomic on same filesystem
            except FileExistsError:
                pass                             # someone else won the race
            except OSError as e:
                raise
            
            return dst 

# TODO: this can be simplified using pandas on parquet files directly
class _BaseManifestLookup:
    """
    Search and download indexed files from S3 using a parquet manifest

    Subclasses should define their index columns and record type.
    """

    # Subclasses should override these
    INDEX_COLUMNS: tuple[str, ...] = ()
    RECORD_CLASS: type = object
    DEFAULT_MANIFEST_PATH: str = ""

    def __init__(
        self,
        manifest_file_path: str = None,
        tmp_dir: str = ARTIFACTS_DIR,
        aws_credentials: dict = None,
        model_class: str = None,
    ):
        """
        Initialize the ManifestLookup with data from a parquet file.

        Args:
            manifest_file_path (str): Path to the parquet file. If None, uses the default for this class.
            tmp_dir (str, optional): Directory for temporary files
            aws_credentials (dict, optional): AWS credentials for S3 access
        Raises:
            FileNotFoundError: If the parquet file doesn't exist
            ValueError: If the parquet file doesn't have required columns
        """
        if model_class:
            self.DEFAULT_MANIFEST_PATH = self.DEFAULT_MANIFEST_PATH.replace("<model_class>", model_class)
        manifest_file_path = manifest_file_path or self.DEFAULT_MANIFEST_PATH

        if not manifest_file_path:
            raise ValueError(
                f"No manifest_file_path provided and no default set for {self.__class__.__name__}"
            )

        # self.tmp_dir = tmp_dir or tempfile.mkdtemp()

        if manifest_file_path.startswith("s3://"):
            manifest_file_path_split = manifest_file_path.split("/")
            self.bucket = manifest_file_path_split[2]
            self.manifest_file_path = "/".join(manifest_file_path_split[3:])
        else:
            self.bucket = None
            self.manifest_file_path = manifest_file_path
        self.s3_fetcher = S3CachedFetcher(
            bucket=self.bucket or DEFAULT_BUCKET,
            tmp_dir=tmp_dir,
            aws_credentials=aws_credentials,
        )

    @functools.cached_property
    def con(self):
        self._load_manifest()
        con = duckdb.connect(":memory:")
        self._load_data(con)
        return con

    def get_unique(self, column_name: str) -> list[str]:
        """
        Get unique values for a specific index column.

        Args:
            column_name: Name of the column

        Returns:
            list: List of unique values
        """
        if column_name not in self.INDEX_COLUMNS:
            raise ValueError(f"column_name must be one of {self.INDEX_COLUMNS}")

        result = self.con.execute(
            f"SELECT DISTINCT {column_name} FROM manifest"
        ).fetchall()
        log.info(f"Found {len(result)} distinct values for {column_name}")
        return [row[0] for row in result]

    def _read_s3_file(self, s3_path: str) -> str:
        """Read a file from S3, caching it locally.

        Args:
            s3_path (str): The S3 path to the object
        Returns:
            str: The local path to the cached object
        """
        return self.s3_fetcher.get(s3_path)

    def _load_manifest(self):
        self.local_file_path = None
        if self.bucket:
            self.local_file_path = self._read_s3_file(self.manifest_file_path)
        else:
            self.local_file_path = self.manifest_file_path

        if not self.local_file_path or not os.path.exists(self.local_file_path):
            log.error(f"Parquet file not found: {self.local_file_path}")
            raise FileNotFoundError(f"Parquet file not found: {self.local_file_path}")

    def _load_data(self, duck_db_con):
        """Load the data from parquet file."""
        try:
            log.info(f"Loading parquet file: {self.local_file_path}")

            # Load and index the data - explicitly specify parquet format
            duck_db_con.execute(
                f"CREATE TABLE manifest AS SELECT * FROM read_parquet('{self.local_file_path}')"
            )

            # Validate required columns exist
            columns = duck_db_con.execute("PRAGMA table_info(manifest)").fetchall()
            column_names = {col[1] for col in columns}  # col[1] is the column name
            required_columns = set(self.INDEX_COLUMNS) | {"file_path"}

            missing_columns = required_columns - column_names
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            log.info(f"Validated schema - found columns: {column_names}")

            # Create indexes for better performance
            for column in self.INDEX_COLUMNS:
                duck_db_con.execute(f"CREATE INDEX idx_{column} ON manifest({column})")

        except Exception as e:
            log.error(f"Error loading manifest file: {e}")
            raise ValueError(f"Error loading manifest file: {e}")

    def _query(self, query_params: dict[str, int | str]) -> list:
        """
        Query the manifest for records matching the provided index values.

        Args:
            query_params: Dictionary mapping column names to values

        Returns:
            list[Record]: A list of Record objects matching the query
        """
        if not query_params:
            raise ValueError(f"At least one of {self.INDEX_COLUMNS} must be provided.")

        # Build column list for SELECT, maintaining order from RECORD_CLASS
        select_columns = ", ".join(
            field.name for field in dataclasses.fields(self.RECORD_CLASS)
        )
        query = [f"SELECT {select_columns} FROM manifest WHERE"]
        params = []
        conditions = []
        assert set(query_params.keys()).issubset(set(self.INDEX_COLUMNS)), (
            f"query_params keys must be a subset of {self.INDEX_COLUMNS}"
        )

        for column in self.INDEX_COLUMNS:
            if column in query_params:
                value = query_params[column]
                conditions.append(f"{column} = ?")
                params.append(value)

        query.append(" AND ".join(conditions))
        result = self.con.execute(" ".join(query), params).fetchall()

        # Convert to list of Record objects
        return [self.RECORD_CLASS(*row) for row in result]

    def close(self):
        """Close the DuckDB connection."""
        if hasattr(self, "con") and self.con:
            self.con.close()
            self.con = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

    def __repr__(self) -> str:
        """Representation of the ManifestLookup object."""
        return f"{self.__class__.__name__}('{self.manifest_file_path}')"


class GeneManifestLookup(_BaseManifestLookup):
    """
    Search and download gene -indexed files from S3 using a parquet manifest
    """

    INDEX_COLUMNS = ("gene_id",)
    RECORD_CLASS = GeneRecord
    DEFAULT_MANIFEST_PATH = GENE_CRE_MANIFEST_FILE_PATH

    def get_record(self, gene_id: str) -> GeneRecord | None:
        """
        Get the record for a specific gene_id.
        This does not download the file (use get_file_path)

        Args:
            gene_id (str): The gene identifier

        Returns:
            GeneRecord | None: The record if found, None otherwise
        """
        results = self._query({"gene_id": gene_id})
        return results[0] if results else None

    def get_file_path(self, gene_id: str) -> str | None:
        """
        Returns local file_path after downloading from S3 if needed.

        Args:
            gene_id (str): The gene identifier

        Returns:
            str | None: Local file path if gene exists, None otherwise
        """
        record = self.get_record(gene_id)
        if not record:
            return None
        return self._read_s3_file(record.file_path)

    def exists(self, gene_id: str) -> bool:
        """
        Check if a gene_id exists in the data.

        Args:
            gene_id (str): The gene identifier

        Returns:
            bool: True if the gene exists, False otherwise
        """
        results = self._query({"gene_id": gene_id})
        return len(results) > 0

    def get_records_for_gene(self, gene_id: str) -> list[GeneRecord]:
        """
        Get all records associated with a specific gene ID.

        Args:
            gene_id (str): The gene identifier

        Returns:
            list[GeneRecord]: List of records associated with the gene
        """
        return self._query({"gene_id": gene_id})


class GeneSequencesManifestLookup(_BaseManifestLookup):
    """
    Search and download gene/cre sequences -indexed files from S3 using a parquet manifest
    """

    INDEX_COLUMNS = ("gene_id", "population")
    RECORD_CLASS = GeneSequenceRecord
    DEFAULT_MANIFEST_PATH = GENE_SEQUENCES_MANIFEST_FILE_PATH

    def get_record(self, gene_id: str, population: str) -> GeneSequenceRecord | None:
        """
        Get the record for a specific gene_id and population combination.
        This does not download the file (use get_file_path)

        Args:
            gene_id (str): The gene identifier
            population (str): The population identifier

        Returns:
            GeneSequenceRecord | None: The record if found, None otherwise
        """
        results = self._query({"gene_id": gene_id, "population": population})
        return results[0] if results else None

    def get_file_path(self, gene_id: str, population: str) -> str | None:
        """
        Returns local file_path after downloading from S3 if needed.

        Args:
            gene_id (str): The gene identifier
            population (str): The population identifier

        Returns:
            str | None: Local file path if gene exists, None otherwise
        """
        record = self.get_record(gene_id, population)
        if not record:
            return None
        return self._read_s3_file(record.file_path)

    def exists(self, gene_id: str, population: str) -> bool:
        """
        Check if a gene_id exists in the data.

        Args:
            gene_id (str): The gene identifier
            population (str): The population identifier

        Returns:
            bool: True if the gene exists, False otherwise
        """
        results = self._query({"gene_id": gene_id, "population": population})
        return len(results) > 0

class CreSequencesManifestLookup(_BaseManifestLookup):
    """
    Search and download gene/cre sequences -indexed files from S3 using a parquet manifest
    """

    INDEX_COLUMNS = ("chromosome", "population")
    RECORD_CLASS = CreSequenceRecord
    DEFAULT_MANIFEST_PATH = CRE_SEQUENCES_MANIFEST_FILE_PATH

    def get_record(self, chromosome: str, population: str) -> CreSequenceRecord | None:
        """
        Get the record for a specific chromosome and population combination.
        This does not download the file (use get_file_path)

        Args:
            chromosome (str): The chromosome identifier
            population (str): The population identifier

        Returns:
            CreSequenceRecord | None: The record if found, None otherwise
        """
        results = self._query({"chromosome": chromosome, "population": population})
        return results[0] if results else None

    def get_file_path(self, chromosome: str, population: str) -> str | None:
        """
        Returns local file_path after downloading from S3 if needed.

        Args:
            chromosome (str): The chromosome identifier
            population (str): The population identifier

        Returns:
            str | None: Local file path if chromosome exists, None otherwise
        """
        record = self.get_record(chromosome, population)
        if not record:
            return None
        return self._read_s3_file(record.file_path)

    def exists(self, chromosome: str, population: str) -> bool:
        """
        Check if a chromosome exists in the data.

        Args:
            chromosome (str): The chromosome identifier
            population (str): The population identifier

        Returns:
            bool: True if the chromosome exists, False otherwise
        """
        results = self._query({"chromosome": chromosome, "population": population})
        return len(results) > 0


class GeneTissueManifestLookup(_BaseManifestLookup):
    """
    Search and download tissue and gene -indexed files from S3 using a parquet manifest
    """

    INDEX_COLUMNS = ("tissue_id", "gene_id")
    RECORD_CLASS = GeneTissueRecord
    DEFAULT_MANIFEST_PATH = GENE_TISSUE_MANIFEST_FILE_PATH

    def _normalize_tissue_id(self, tissue_id: str | int) -> int:
        """
        Normalize tissue_id to an integer, handling various string formats.

        Args:
            tissue_id: The tissue identifier (int or str)

        Returns:
            int: The normalized tissue ID
        """
        if isinstance(tissue_id, int):
            return tissue_id

        # Handle string tissue IDs with prefixes
        if tissue_id.startswith("model_"):
            tissue_id = tissue_id.replace("model_", "")
        if tissue_id.startswith("tissue_"):
            tissue_id = tissue_id.replace("tissue_", "")

        return int(tissue_id)

    def get_record(self, gene_id: str, tissue_id: str | int) -> GeneTissueRecord | None:
        """
        Get the record for a specific gene_id and tissue_id combination.
        This does not download the file (use get_file_path)

        Args:
            gene_id (str): The gene identifier
            tissue_id (str|int): The tissue identifier

        Returns:
            GeneTissueRecord | None: The record if found, None otherwise
        """
        normalized_tissue_id = self._normalize_tissue_id(tissue_id)
        results = self._query({"gene_id": gene_id, "tissue_id": normalized_tissue_id})
        return results[0] if results else None

    def get_file_path(self, gene_id: str, tissue_id: str | int) -> str | None:
        """
        Returns local file_path after downloading from S3 if needed.

        Args:
            gene_id (str): The gene identifier
            tissue_id (str|int): The tissue identifier

        Returns:
            str | None: Local file path if combination exists, None otherwise
        """
        record = self.get_record(gene_id, tissue_id)
        if not record:
            return None
        return self._read_s3_file(record.file_path)

    def exists(self, gene_id: str, tissue_id: str | int) -> bool:
        """
        Check if a combination of gene_id and tissue_id exists in the data.

        Args:
            gene_id (str): The gene identifier
            tissue_id (str|int): The tissue identifier

        Returns:
            bool: True if the combination exists, False otherwise
        """
        normalized_tissue_id = self._normalize_tissue_id(tissue_id)
        results = self._query({"gene_id": gene_id, "tissue_id": normalized_tissue_id})
        return len(results) > 0

    def get_records_for_gene(self, gene_id: str) -> list[GeneTissueRecord]:
        """
        Get all records associated with a specific gene ID.

        Args:
            gene_id (str): The gene identifier

        Returns:
            list[GeneTissueRecord]: List of records associated with the gene
        """
        return self._query({"gene_id": gene_id})

    def get_records_for_tissue(self, tissue_id: str | int) -> list[GeneTissueRecord]:
        """
        Get all records associated with a specific tissue ID.

        Args:
            tissue_id (str|int): The tissue identifier

        Returns:
            list[GeneTissueRecord]: List of records associated with the tissue
        """
        normalized_tissue_id = self._normalize_tissue_id(tissue_id)
        return self._query({"tissue_id": normalized_tissue_id})

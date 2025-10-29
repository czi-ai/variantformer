#!/usr/bin/env -S python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3",
# ]
# ///
import argparse
import collections
import dataclasses
import hashlib
import pathlib
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config


@dataclasses.dataclass
class Artifact:
    remote_uri: str
    local_path: str


@dataclasses.dataclass
class AWSCredentials:
    profile: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None


DEFAULT_DESTINATION = pathlib.Path(__file__).parent.resolve() / "_artifacts"
DEFAULT_BUCKET="czi-variantformer"


ARTIFACTS = [
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/v4_ag/all_genes_gencodeV24.csv",
        local_path="all_genes_ag_gencodeV24.csv",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/v4_pcg/all_genes_gencodeV24.csv",
        local_path="all_genes_v1_pcg_gencodeV24.csv",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/v4_pcg/tokenizer_checkpoint.pth",
        local_path="pretrained_tokenizers_checkpoint.pth",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/data/HG00096.vcf.gz",
        local_path="HG00096.vcf.gz",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/data/HG00096.vcf.gz.tbi",
        local_path="HG00096.vcf.gz.tbi",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/v4_ag/checkpoint.pth",
        local_path="v4_ag_epoch9_checkpoint.pth",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/v4_pcg/checkpoint.pth",
        local_path="v4_pcg_epoch11_checkpoint.pth",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/common/ENCFF234XEZ.bed.gz",
        local_path="ENCFF234XEZ.bed.gz",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/data/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz",
        local_path="GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/gold_data/f9bbc0ba.pq",
        local_path="f9bbc0ba.pq",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/gold_data/924979a7.pq",
        local_path="924979a7.pq",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/gold_data/befd2388.npz",
        local_path="befd2388.npz",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/gold_data/be73e19a.pq",
        local_path="be73e19a.pq",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/common/reference_genomes/data_split/hg38/cres/data_split/hg38_chr19.pkl.gz",
        local_path="reference_genomes/data_split/hg38/cres/data_split/hg38_chr19.pkl.gz",
    ),
    Artifact(
        remote_uri=f"s3://{DEFAULT_BUCKET}/model/common/reference_genomes/data/hg38/genes/data/ENSG00000130203.9_hg38.npz",
        local_path="reference_genomes/data/hg38/genes/data/ENSG00000130203.9_hg38.npz",
    ),
    # Artifact(
    #     remote_uri=f"s3://{DEFAULT_BUCKET}/data/1KG_af_hg38_tables/",
    #     local_path="1KG_af_hg38_tables",
    # ),
]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key/path into (bucket, key)"""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")

    parts = uri.removeprefix("s3://").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _does_local_file_match_etag(local_path: pathlib.Path, etag: str) -> bool:
    """
    Returns True if the local file's md5 hash matches the ETag

    This does not properly handle multi-part uploads, and always returns False for those
    """
    etag = etag.strip('"')

    # multipart upload ETags contains a hyphen
    if "-" in etag:
        # Skip verifying the hash because the process is complicated
        # assume if there's a file there, it's the correct one
        return local_path.is_file()

    # simple single part case, the ETag is the md5 hash
    md5_hash = hashlib.md5()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)

    local_md5 = md5_hash.hexdigest()
    return local_md5 == etag


def _download_file(client, bucket: str, key: str, local_path: pathlib.Path):
    """
    Download a single file from S3 to local path.
    Checks ETag first if file exists and skips download if it matches.
    """
    if local_path.is_file():
        try:
            head_response = client.head_object(Bucket=bucket, Key=key)
            etag = head_response["ETag"]
            if _does_local_file_match_etag(local_path, etag):
                return  # it's already been downloaded
        except Exception:
            pass  # we might still be able to download it even though we couldn't check the etag

    local_path.parent.mkdir(parents=True, exist_ok=True)

    client.download_file(bucket, key, str(local_path))


def download(client, destination: pathlib.Path) -> list[str]:
    """
    Download files from S3 to local.
    Returns list of error messages encountered during download.
    """
    destination.mkdir(parents=True, exist_ok=True)
    errors = []

    for artifact in ARTIFACTS:
        try:
            bucket, key = _parse_s3_uri(artifact.remote_uri)

            is_prefix = artifact.remote_uri.endswith("/")

            if is_prefix:
                # download everything under the prefix
                paginator = client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=bucket, Prefix=key)

                for page in pages:
                    if "Contents" not in page:
                        continue

                    for obj in page["Contents"]:
                        obj_key = obj["Key"]
                        relative_path = obj_key[len(key) :]
                        if not relative_path or relative_path.endswith("/"):
                            continue  # skip directory markers
                        local_path = destination / artifact.local_path / relative_path
                        try:
                            _download_file(client, bucket, obj_key, local_path)
                        except Exception as e:
                            errors.append(
                                f"Error downloading {artifact.remote_uri}{relative_path}: {e}"
                            )
            else:
                # download the single file
                local_path = destination / artifact.local_path
                try:
                    _download_file(client, bucket, key, local_path)
                except Exception as e:
                    errors.append(f"Error downloading {artifact.remote_uri}: {e}")

        except Exception as e:
            errors.append(f"Error processing artifact {artifact.remote_uri}: {e}")

    return errors


def _validate():
    """check that the artifact destinations make sense"""
    local_path_count = collections.Counter(
        [artifact.local_path for artifact in ARTIFACTS]
    )
    multiples = {filename for filename, count in local_path_count.items() if count > 1}
    if multiples:
        raise Exception(
            f"Multiple artifacts will be downloaded to the same local path: {multiples!r}"
        )


def get_s3_client(credentials: AWSCredentials | None = None, is_bucket_public: bool = False):
    if is_bucket_public:
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        return client
        
    credentials = (
        credentials or AWSCredentials()
    )  # simplifies the logic if this is not None

    if credentials.profile:
        # use that profile
        session = boto3.Session(profile_name=credentials.profile)
        client = session.client("s3")
    else:
        # try to use what credentials they provided
        # if they're all none, boto will use its standard way to get them from the environment
        client_kwargs = {}
        if credentials.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = credentials.aws_access_key_id
        if credentials.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = credentials.aws_secret_access_key
        if credentials.aws_session_token:
            client_kwargs["aws_session_token"] = credentials.aws_session_token
        client = boto3.client("s3", **client_kwargs)
    return client


def main(
    destination: pathlib.Path = DEFAULT_DESTINATION,
    credentials: AWSCredentials | None = None,
):
    try:
        _validate()
    except Exception as e:
        print("ABORTING DOWNLOAD", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    client = get_s3_client(credentials, is_bucket_public=True)
    errors = download(client, destination)

    if errors:
        print("\nErrors encountered during download:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download artifacts from S3 to local filesystem"
    )
    parser.add_argument(
        "--destination",
        type=pathlib.Path,
        default=DEFAULT_DESTINATION,
        help=f"Destination directory (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument("--profile", help="AWS profile name to use")
    parser.add_argument(
        "--aws-access-key-id", help="AWS access key ID (if no profile specified)"
    )
    parser.add_argument(
        "--aws-secret-access-key",
        help="AWS secret access key (if no profile specified)",
    )
    parser.add_argument(
        "--aws-session-token",
        help="AWS session token (optional, if no profile specified)",
    )

    args = parser.parse_args()

    # specifying a profile overrides any of these other credentials
    if args.profile and any(
        [
            args.aws_access_key_id,
            args.aws_secret_access_key,
            args.aws_session_token,
        ]
    ):
        parser.error(
            "Cannot specify both --profile and explicit AWS credentials "
            "(--aws-access-key-id, --aws-secret-access-key, --aws-session-token)"
        )

    credentials = AWSCredentials(
        profile=args.profile,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=args.aws_session_token,
    )

    main(destination=args.destination, credentials=credentials)

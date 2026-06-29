import subprocess
import pandas as pd


class ExtractSeqFromBed:
    def __init__(
        self, neighbour_hood: int, ref_fasta: str, upstream_neighbour_hood: int = None
    ):
        self.neighbour_hood = neighbour_hood
        self.ref_fasta = ref_fasta
        self.upstream_neighbour_hood = upstream_neighbour_hood

    def apply_bcftools_consensus(
        self, region, vcf_file, reference_fasta, variant_type: str = None
    ):
        """Apply bcftools consensus to extract mutated sequence for a region."""
        chrom = region.chrom
        start = max(0, int(region.start) - self.neighbour_hood)
        end = int(region.end) + self.neighbour_hood
        region_str = f"{chrom}:{start + 1}-{end}"  # bcftools uses 1-based coordinates

        # Command to extract the reference sequence. Fallback if bcftools consensus fails
        cmd_ref = ["samtools", "faidx", reference_fasta, region_str]
        # If vcf_file is None, return the reference sequence
        if not vcf_file:
            result_ref = subprocess.run(cmd_ref, capture_output=True, text=True)
            if result_ref.returncode != 0:
                print(region_str)
                print(f"\nError running samtools faidx: {result_ref.stderr}")
                return None, 0
            else:
                mutated_seq = "".join(result_ref.stdout.strip().split("\n")[1:])
                return mutated_seq, 0
        # If vcf_file is not None, run bcftools consensus
        # Command to extract the reference sequence and apply mutations
        if variant_type == "SNP":
            bcftools_args = [
                "bcftools",
                "consensus",
                "-H",
                "I",
                "-e",
                'ALT~\"<.*>\" || TYPE!=\"snp\"',
                vcf_file,
            ]
        else:
            bcftools_args = [
                "bcftools",
                "consensus",
                "-H",
                "I",
                "-e",
                'ALT~\"<.*>\"',
                vcf_file,
            ]

        # Use piped commands without shell=True
        samtools_process = subprocess.Popen(
            cmd_ref, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        result = subprocess.run(
            bcftools_args, stdin=samtools_process.stdout, capture_output=True, text=True
        )
        samtools_process.stdout.close()
        samtools_stderr = samtools_process.stderr.read()
        samtools_process.stderr.close()
        samtools_process.wait()

        # If bcftools consensus fails, return the reference sequence
        if result.returncode != 0:
            print(region_str)
            print(f"\nError running bcftools consensus: {result.stdout}")
            print(f"\nError running bcftools consensus: {result.stderr}")
            print("Falling back to ref genome")
            result_ref = subprocess.run(cmd_ref, capture_output=True, text=True)
            if result_ref.returncode != 0:
                print(f"\nError running samtools faidx: {result_ref.stderr}")
                return None, 0
            mutated_seq = "".join(result_ref.stdout.strip().split("\n")[1:])
            return mutated_seq, 0

        mutated_seq = "".join(result.stdout.strip().split("\n")[1:])
        # If bcftools consensus succeeds, return the mutated sequence and the number of mutations
        mutations = None
        if len(result.stderr.split("\n")) >= 2:
            mutations = result.stderr.split("\n")[-2].split()[1]
        else:
            print(f"Less than 2 lines in stderr: {result.stderr}")

        try:
            mutations = int(mutations)
        except (ValueError, TypeError):
            err = result.stderr.split("\n")
            mutations = 0
            print(f"Cannot convert to int: {err}, stdout: {result.stdout}")

        return mutated_seq, mutations

    def process_region(self, args):
        """Process a single region (used as fallback)."""
        region, vcf_file, reference_fasta, variant_type = args
        mutated_seq, mutations = self.apply_bcftools_consensus(
            region, vcf_file, reference_fasta, variant_type=variant_type
        )
        if mutated_seq:
            D = {
                "chrom": region.chrom,
                "start_cre": max(0, region.start - self.neighbour_hood),
                "end_cre": region.end + self.neighbour_hood,
                "sequence": mutated_seq,
                "cCRE": region.cCRE,
            }
            return D, mutations
        return None, None

    def _serial_processing(self, args_list):
        """Fallback (serially processes regions with own bcftools call)."""
        results = []
        for arg in args_list:
            result = self.process_region(arg)
            results.append(result)
        return results
    
    def _region_to_str(self, region):
        """Builds 1-based samtools region string for a given CRE region."""
        chrom = region.chrom
        start = max(0, int(region.start) - self.neighbour_hood)
        end = int(region.end) + self.neighbour_hood
        return f"{chrom}:{start + 1}-{end}"
    
    @staticmethod
    def _bcftools_consensus_args(vcf_file, variant_type):
        if variant_type == "SNP":
            exclude = 'ALT~"<.*>" || TYPE!="snp"'
        else:
            exclude = 'ALT~"<.*>"'
        return ["bcftools", "consensus", "-H", "I", "-e", exclude, vcf_file]
    
    @staticmethod
    def _parse_multifasta(text: str):
        """Parse multi-record FASTA text into ordered list of sequences."""
        sequences = []
        current = None
        for line in text.splitlines():
            if line.startswith(">"):
                if current is not None:
                    sequences.append("".join(current))
                current = []
            elif current is not None:
                current.append(line.strip())
        if current is not None:
            sequences.append("".join(current))
        return sequences
    
    def _extract_sequences_batched(self, region_strs, vcf_file, variant_type):
        """Extract sequences for multiple regions with a single subprocess pipeline.

        Executes single `samtools faidx` command with multiple regions, optionally piped
        through `bcftools consensus` for variant consensus calling.
        More efficient than processing regions individually!
        """
        cmd_ref = ["samtools", "faidx", self.ref_fasta, *region_strs]

        if not vcf_file:
            result = subprocess.run(cmd_ref, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"\nError running batched samtools faidx: {result.stderr}")
                return None
            sequences = self._parse_multifasta(result.stdout)
        else:
            samtools_process = subprocess.Popen(
                cmd_ref, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            result = subprocess.run(
                self._bcftools_consensus_args(vcf_file, variant_type),
                stdin=samtools_process.stdout,
                capture_output=True,
                text=True,
            )
            samtools_process.stdout.close()
            samtools_process.stderr.read()
            samtools_process.stderr.close()
            samtools_process.wait()

            if result.returncode != 0:
                print(f"\nError running batched bcftools consensus: {result.stderr}")
                return None
            sequences = self._parse_multifasta(result.stdout)

        if len(sequences) != len(region_strs):
            print(
                "Batched consensus returned "
                f"{len(sequences)} records for {len(region_strs)} regions; "
                "falling back to per-region extraction."
            )
            return None
        return sequences

    def process_subject(
        self, vcf_file: str, bed_regions: pd.DataFrame, variant_type: str = None
    ):
        """Apply variants to all of a gene's CRE windows in one subprocess call.

        Runs a single `samtools faidx | bcftools consensus` over every CRE region
        rather than spawning a nested pool of one subprocess pair per region. The
        per-record output is identical to calling `bcftools consensus` on each
        region individually.
        """
        regions = [region for _, region in bed_regions.iterrows()]
        region_strs = [self._region_to_str(region) for region in regions]

        sequences = None
        if region_strs:
            sequences = self._extract_sequences_batched(
                region_strs, vcf_file, variant_type
            )

        D = []
        if sequences is not None:
            for region, mutated_seq in zip(regions, sequences):
                if not mutated_seq:
                    continue
                D.append(
                    {
                        "chrom": region.chrom,
                        "start_cre": max(0, region.start - self.neighbour_hood),
                        "end_cre": region.end + self.neighbour_hood,
                        "sequence": mutated_seq,
                        "cCRE": region.cCRE,
                    }
                )
        else:
            # fall back to serial per-region calls
            args_list = [
                (region, vcf_file, self.ref_fasta, variant_type)
                for region in regions
            ]
            for result, _ in self._serial_processing(args_list):
                if result:
                    D.append(result)

        df = pd.DataFrame(D)
        if not df.empty and not df["start_cre"].is_monotonic_increasing:
            df = df.sort_values(by=["chrom", "start_cre"], ascending=True).reset_index(
                drop=True
            )
        return df

    def apply_bcftools_consensus_to_gene(
        self,
        chrom: str,
        strand: str,
        start: int,
        end: int,
        vcf_file: str,
        variant_type: str = None,
    ):
        """
        Apply bcftools consensus to extract mutated sequence for a gene.
        Args:
            chrom: str, chromosome name
            strand: str, strand of the gene
            start: int, start position of the gene
            end: int, end position of the gene
            vcf_file: str, path to the vcf file
        Returns:
            mutated_seq: str, mutated sequence of the gene
        """
        if strand == "-":
            start = max(
                int(start), int(end) - self.neighbour_hood
            )  # the start of the gene is the end location because the strand is negative and 300,000bp downstream
            end = (
                int(end) + self.upstream_neighbour_hood
            )  # the end of the gene is the end location because the strand is negative and 1000 bp upstream
        else:
            start = max(
                0, int(start) - self.upstream_neighbour_hood
            )  # the start of the gene is the start location because the strand is positive and 1000bp upstream
            end = min(
                int(end), int(start) + self.neighbour_hood
            )  # the end of the gene is the end location because the strand is positive and 300,000 bp downstream

        region_str = f"{chrom}:{start + 1}-{end}"  # bcftools uses 1-based coordinates

        cmd_ref = ["samtools", "faidx", self.ref_fasta, region_str]
        # If vcf_file is None, return the reference sequence
        if not vcf_file:
            result_ref = subprocess.run(cmd_ref, capture_output=True, text=True)
            if result_ref.returncode != 0:
                print(region_str)
                print(f"\nError running samtools faidx: {result_ref.stderr}")
                return None
            else:
                mutated_seq = "".join(result_ref.stdout.strip().split("\n")[1:])
                return mutated_seq
        # If vcf_file is not None, run bcftools consensus
        if variant_type == "SNP":
            bcftools_args = [
                "bcftools",
                "consensus",
                "-H",
                "I",
                "-e",
                'ALT~\"<.*>\" || TYPE!=\"snp\"',
                vcf_file,
            ]
        else:
            bcftools_args = [
                "bcftools",
                "consensus",
                "-H",
                "I",
                "-e",
                'ALT~\"<.*>\"',
                vcf_file,
            ]

        # Use piped commands without shell=True
        samtools_process = subprocess.Popen(
            cmd_ref, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        result = subprocess.run(
            bcftools_args, stdin=samtools_process.stdout, capture_output=True, text=True
        )
        samtools_process.stdout.close()
        samtools_stderr = samtools_process.stderr.read()
        samtools_process.stderr.close()
        samtools_process.wait()

        if result.returncode != 0:
            print(region_str)
            print(f"\nError running bcftools consensus: {result.stdout}")
            print(f"\nError running bcftools consensus: {result.stderr}")
            print("Falling back to reference")
            result_ref = subprocess.run(cmd_ref, capture_output=True, text=True)
            if result_ref.returncode != 0:
                raise ValueError(
                    f"Error running bcftools consensus: {result_ref.stderr}"
                )
            mutated_seq = "".join(result_ref.stdout.strip().split("\n")[1:])
            return mutated_seq

        mutations = result.stderr.split("\n")[-2].split()[1]
        # print(f"Applied bcftools consensus for gene region {region_str}, mutations: {mutations}")
        # print('---'*20)
        # Using the mutated sequence
        mutated_seq = "".join(result.stdout.strip().split("\n")[1:])
        return mutated_seq

    def process_gene(self, gene_info, vcf_file, variant_type: str = None):
        """Apply bcftools consensus to extract mutated sequence for a region."""
        chrom = gene_info["chromosome"]
        start = gene_info["start"]
        end = int(gene_info["end"])
        strand = gene_info["strand"]
        mutated_seq = self.apply_bcftools_consensus_to_gene(
            chrom=chrom,
            strand=strand,
            start=start,
            end=end,
            vcf_file=vcf_file,
            variant_type=variant_type,
        )
        return mutated_seq

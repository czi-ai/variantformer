import os
import subprocess
from multiprocessing import Pool
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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
        """Process a single region."""
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

    def _is_nested_parallel_context(self):
        """Check if we're already in a parallel execution context."""
        import threading
        import os

        # Check for PyTorch DataLoader worker context
        try:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                return True
        except ImportError:
            pass

        # Check various other indicators of parallel execution
        indicators = [
            # Check if we're in a joblib worker process
            os.environ.get("JOBLIB_WORKER_ID") is not None,
            # Check PyTorch environment variables
            os.environ.get("PYTORCH_WORKER_ID") is not None,
            # Check if thread name suggests we're in a worker
            "worker" in threading.current_thread().name.lower(),
            # Check if process name suggests we're in a worker
            "worker" in mp.current_process().name.lower(),
            # Check for multiprocessing Pool worker naming convention
            "Pool" in mp.current_process().name,
            # Check for DataLoader worker naming patterns
            "DataLoader" in mp.current_process().name,
        ]

        return any(indicators)

    def _try_joblib_parallel(self, args_list, max_workers):
        """Try joblib.Parallel for multiprocessing."""
        from joblib import Parallel, delayed

        # Always try multiprocessing first - let other methods handle nested context detection
        return Parallel(n_jobs=max_workers, backend="multiprocessing")(
            delayed(self.process_region)(arg) for arg in args_list
        )

    def _try_joblib_loky(self, args_list, max_workers):
        """Try joblib.Parallel with loky backend (good for nested contexts)."""
        from joblib import Parallel, delayed

        return Parallel(n_jobs=max_workers, backend="loky")(
            delayed(self.process_region)(arg) for arg in args_list
        )

    def _try_safe_nested_spawn(self, args_list, max_workers):
        """Try a smaller spawn pool specifically for nested contexts."""
        # Use even fewer workers for safer nested processing
        safe_workers = min(max_workers, 2)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=safe_workers) as pool:
            return pool.map(self.process_region, args_list)

    def _try_joblib_threading(self, args_list, max_workers):
        """Try joblib.Parallel with threading backend specifically."""
        from joblib import Parallel, delayed

        return Parallel(n_jobs=max_workers, backend="threading")(
            delayed(self.process_region)(arg) for arg in args_list
        )

    def _try_optimized_io_threading(self, args_list, max_workers):
        """Optimized threading specifically for I/O-bound bcftools operations."""
        from concurrent.futures import ThreadPoolExecutor

        # Use higher thread count for I/O-bound operations
        io_workers = min(max_workers * 2, len(args_list), 16)

        with ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="BCFTools"
        ) as executor:
            return list(executor.map(self.process_region, args_list))

    def _try_process_pool_executor(self, args_list, max_workers):
        """Try concurrent.futures.ProcessPoolExecutor."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.process_region, args_list))

    def _try_spawn_pool(self, args_list, max_workers):
        """Try multiprocessing Pool with spawn context."""
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max_workers) as pool:
            return pool.map(self.process_region, args_list)

    def _try_forkserver_pool(self, args_list, max_workers):
        """Try multiprocessing Pool with forkserver context."""
        ctx = mp.get_context("forkserver")
        with ctx.Pool(processes=max_workers) as pool:
            return pool.map(self.process_region, args_list)

    def _try_thread_pool_executor(self, args_list, max_workers):
        """Try concurrent.futures.ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.process_region, args_list))

    def _try_async_thread_pool(self, args_list, max_workers):
        """Try async threading approach for better I/O concurrency."""
        import concurrent.futures
        import asyncio

        async def run_async():
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                tasks = [
                    loop.run_in_executor(executor, self.process_region, arg)
                    for arg in args_list
                ]
                return await asyncio.gather(*tasks)

        # Run in new event loop to avoid conflicts
        try:
            return asyncio.run(run_async())
        except RuntimeError:
            # If there's already an event loop, run in thread
            import threading

            result = []
            exception = []

            def run_in_thread():
                try:
                    result.append(asyncio.run(run_async()))
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception[0]
            return result[0]

    def _try_global_spawn_pool(self, args_list, max_workers):
        """Try setting global start method to spawn and using regular Pool."""
        original_start_method = mp.get_start_method()
        try:
            mp.set_start_method("spawn", force=True)
            with Pool(processes=max_workers) as pool:
                return pool.map(self.process_region, args_list)
        finally:
            # Restore original start method
            try:
                mp.set_start_method(original_start_method, force=True)
            except:
                pass  # Ignore errors when restoring

    def _serial_processing(self, args_list):
        """Fallback to serial processing."""
        results = []
        for arg in args_list:
            result = self.process_region(arg)
            results.append(result)
        return results

    def process_subject(
        self, vcf_file: str, bed_regions: pd.DataFrame, variant_type: str = None
    ):
        """Process each subject's VCF file and apply mutations to the reference sequence."""
        ncpus = os.cpu_count()
        args_list = [
            (region, vcf_file, self.ref_fasta, variant_type)
            for _, region in bed_regions.iterrows()
        ]

        # Check if we're in a DataLoader worker context
        is_nested = self._is_nested_parallel_context()

        if is_nested:
            # Adjust worker count for nested context to avoid resource conflicts
            try:
                import torch.utils.data

                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    # Calculate workers per DataLoader worker
                    dataloader_workers = worker_info.num_workers
                    max_workers = max(
                        1, min(ncpus // dataloader_workers, len(bed_regions), 8)
                    )
                else:
                    max_workers = max(1, min(ncpus // 4, len(bed_regions), 4))
            except ImportError:
                max_workers = max(1, min(ncpus // 4, len(bed_regions), 4))

            # For nested context, prioritize threading approaches (processes won't work in daemon context)
            approaches = [
                ("Optimized I/O Threading", self._try_optimized_io_threading),
                ("ThreadPoolExecutor", self._try_thread_pool_executor),
                ("async thread pool", self._try_async_thread_pool),
                ("joblib.Threading", self._try_joblib_threading),
                # These will fail but kept as fallbacks with minimal logging
                ("joblib.loky", self._try_joblib_loky),
            ]
        else:
            # Normal context - use more workers
            max_workers = min(ncpus // 3 + 1, len(bed_regions))
            print(f"Normal context: using {max_workers} workers")

            approaches = [
                ("joblib.Parallel", self._try_joblib_parallel),
                ("ProcessPoolExecutor", self._try_process_pool_executor),
                ("spawn Pool", self._try_spawn_pool),
                ("global spawn Pool", self._try_global_spawn_pool),
                ("forkserver Pool", self._try_forkserver_pool),
                ("ThreadPoolExecutor", self._try_thread_pool_executor),
            ]

        results = None
        for name, method in approaches:
            try:
                results = method(args_list, max_workers)
                break
            except Exception as e:
                # Suppress expected errors in nested contexts
                if is_nested and (
                    "daemonic processes" in str(e) or "multiprocessing" in str(e)
                ):
                    continue  # Expected failure, no logging
                else:
                    print(f"{name} failed: {e}")
                continue

        # If all methods failed, fall back to serial processing
        if results is None:
            print(f"All methods failed, processing {len(bed_regions)} regions serially")
            results = self._serial_processing(args_list)
        D = []
        for result in results:
            if result[0]:
                D.append(result[0])
        df = pd.DataFrame(D)
        if not df["start_cre"].is_monotonic_increasing:
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

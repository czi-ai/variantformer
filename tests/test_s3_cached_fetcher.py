import unittest
import sys
import multiprocessing
import time
from pathlib import Path
from filelock import FileLock
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.assets import S3CachedFetcher 


def fetch_file_worker(args):
    """Worker function for multiprocessing test"""
    worker_id, file_to_fetch = args
    print(f"Worker {worker_id} starting fetch at {time.time()}")
    
    fetcher = S3CachedFetcher()
    start_time = time.time()
    local_file = fetcher.get(file_to_fetch)
    end_time = time.time()
    
    print(f"Worker {worker_id} completed fetch in {end_time - start_time:.2f}s at {end_time}")
    
    # Return results for verification
    return {
        'worker_id': worker_id,
        'local_file': local_file,
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time
    }


def _is_file_locked(lock_path: str) -> bool:
    return FileLock(lock_path).is_locked


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = S3CachedFetcher()
        self.bucket = self.fetcher.bucket

    def test_fetch_1_file(self) -> None:
        file_to_fetch = "model/common/reference_genomes/data/eur/genes/data/ENSG00000111731.12_HG00096.npz"
        local_file = Path(self.fetcher.get(file_to_fetch))
        self.assertTrue(local_file.exists())
        # Lock file persists after release - this is standard behavior for FileLock
        lock_file = Path(str(local_file) + '.lock')
        self.assertTrue(lock_file.exists())
        self.assertFalse(_is_file_locked(str(lock_file)))

    def test_fetch_same_file_multiprocessing(self) -> None:
        """Test fetching the same file with multiple processes in parallel"""
        file_to_fetch = "model/common/reference_genomes/data/eur/genes/data/ENSG00000111731.12_HG00096.npz"
        num_workers = 64
        
        # Clean up any existing files to test the race condition
        test_fetcher = S3CachedFetcher()
        dst_path = Path(test_fetcher.tmp_dir) / file_to_fetch
        if dst_path.exists():
            dst_path.unlink()
        
        print(f"\nStarting multiprocessing test with {num_workers} workers")
        print(f"Target file: {file_to_fetch}")
        
        # Prepare arguments for workers
        worker_args = [(i, file_to_fetch) for i in range(num_workers)]
        
        # Use multiprocessing Pool to run workers in parallel
        start_time = time.time()
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(fetch_file_worker, worker_args)
        end_time = time.time()
        
        print(f"\nAll workers completed in {end_time - start_time:.2f}s")
        
        # Verify all workers got the same file
        local_files = [result['local_file'] for result in results]
        self.assertTrue(all(Path(f).exists() for f in local_files))
        self.assertTrue(all(f == local_files[0] for f in local_files))
        
        # Verify lock file exists (persists after release - standard FileLock behavior)
        lock_file = Path(str(local_files[0]) + '.lock')
        self.assertTrue(lock_file.exists())
        self.assertFalse(_is_file_locked(str(lock_file)))

        # Print timing information
        for result in results:
            print(f"Worker {result['worker_id']}: {result['duration']:.2f}s")
        
        print(f"File downloaded to: {local_files[0]}")



if __name__ == "__main__":
    unittest.main()
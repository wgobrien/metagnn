import subprocess
from pathlib import Path
import shutil
import uuid

from metagnn.utils import get_logger

class MashRunner:
    def __init__(self, binary_path: str = None):
        if binary_path:
            self.binary_path = binary_path
        else:
            self.binary_path = self.check_mash_installed()
        self.logger = get_logger(__name__)

    def check_mash_installed(self):
        mash_path = shutil.which("mash")
        if mash_path is None:
            raise RuntimeError(
                "mash not found. Please install it using Bioconda:\n"
                "    conda install bioconda::mash"
            )
        return mash_path

    def run(
        self,
        reference_path: str,
        query_paths,
        output_path: str,
        kmer_size: int = 14,
        sketch_size: int = 10_000,
        max_pvalue: float = 1.0,
        max_distance: float = 1.0,
        num_threads: int = 1,
        table_output: bool = False,
    ):
        query_paths = [query_paths] if isinstance(query_paths, str) else query_paths
    
        query_list_path = None
        out_dir = Path(output_path).parent / str(uuid.uuid4())
        out_dir.mkdir(parents=True, exist_ok=True)
    
        try:
            if len(query_paths) > 1:
                query_list_path = out_dir / "queries.txt"
                with open(query_list_path, "w") as f:
                    for path in query_paths:
                        f.write(f"{path}\n")
    
            cmd = [self.binary_path, "dist", "-i"]
    
            cmd += [reference_path]
    
            if len(query_paths) == 1:
                cmd += query_paths
            else:
                cmd += ["-l", str(query_list_path)]
    
            cmd += [
                "-k", str(kmer_size),
                "-s", str(sketch_size),
                "-v", str(max_pvalue),
                "-d", str(max_distance),
                "-p", str(num_threads),
            ]
    
            if table_output:
                cmd.append("-t")
    
            self.logger.info(f"Running command: {' '.join(cmd)}")
            with open(output_path, "w") as outfile:
                subprocess.run(cmd, stdout=outfile, check=True)
    
        finally:
            if query_list_path and query_list_path.exists():
                query_list_path.unlink()
            if out_dir.exists():
                shutil.rmtree(out_dir)
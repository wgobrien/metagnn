import subprocess
from pathlib import Path
import shutil
import uuid

from metagnn.utils import get_logger

class LzAniRunner:
    def __init__(self, binary_path: str=None):
        if binary_path:
            self.binary_path = binary_path
        else:
            self.binary_path = self.check_lzani_installed()
        self.logger = get_logger(__name__)
    
    def check_lzani_installed(self):
        lzani_path = shutil.which("lz-ani")
        if lzani_path is None:
            raise RuntimeError(
                "lz-ani not found. Please install it using Bioconda:\n"
                "    conda install -c bioconda lz-ani"
            )
        return lzani_path

    def run(
        self,
        fasta_paths,
        output_path,
        outfmt="standard",
        filter_path=None,
        filter_threshold=None,
        mal=11,
        msl=7,
        mrd=40,
        mqd=40,
        reg=35,
        aw=15,
        am=7,
        ar=3,
        num_threads=1,
        verbose=False,
    ):
        fasta_paths = [fasta_paths] if isinstance(fasta_paths, str) else fasta_paths
    
        out_dir = Path(output_path).parent / str(uuid.uuid4())
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / "ids.txt"
    
        if len(fasta_paths) > 1:
            with open(txt_path, "w") as f:
                for path in fasta_paths:
                    f.write(f"{path}\n")
    
        try:
            cmd = [self.binary_path, "all2all"] 
    
            if len(fasta_paths) == 1:
                cmd += ["--in-fasta", fasta_paths[0]]  # Single file input
            else:
                cmd += ["--in-txt", str(txt_path)]     # Multiple files in text file
    
            cmd += [
                "-o", str(output_path),
            ]
    
            if filter_path and filter_threshold is not None:
                cmd += ["--flt-kmerdb", filter_path, str(filter_threshold)]
    
            cmd += [
                "--mal", str(mal),
                "--msl", str(msl),
                "--mrd", str(mrd),
                "--mqd", str(mqd),
                "--reg", str(reg),
                "--aw", str(aw),
                "--am", str(am),
                "--ar", str(ar),
                "--threads", str(num_threads),
            ]
    
            if verbose:
                cmd.append("--verbose")
    
            self.logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    
        finally:
            if out_dir.exists():
                shutil.rmtree(out_dir)
import os
import requests
from ftplib import FTP

REFSEQ_VIRAL_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/"
OUTPUT_DIR = "refseq_viral_sequences"
GBFF_EXTENSION = ".gbff.gz"
FASTA_EXTENSION = ".fna.gz"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, output_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def get_viral_sequences():
    print("Connecting to the server...")
    response = requests.get(REFSEQ_VIRAL_URL)
    response.raise_for_status()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    gbff_files = [
        a["href"] for a in soup.find_all("a", href=True)
        if GBFF_EXTENSION in a["href"]
    ]

    fasta_files = [
        a["href"] for a in soup.find_all("a", href=True)
        if FASTA_EXTENSION in a["href"]
    ]

    if not gbff_files and not fasta_files:
        print("No GBFF or FASTA files found.")
        return

    for file_list, file_type in [(gbff_files, "GBFF"), (fasta_files, "FASTA")]:
        for file in file_list:
            file_url = REFSEQ_VIRAL_URL + file
            output_path = os.path.join(OUTPUT_DIR, file)

            if not os.path.exists(output_path):
                print(f"Downloading {file} ({file_type})...")
                download_file(file_url, output_path)
            else:
                print(f"{file} ({file_type}) already exists")


if __name__ == "__main__":
    get_viral_sequences()
    print("Done Done!")

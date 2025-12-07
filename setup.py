import os
import bz2
import shutil

import urllib.request

import logging

import nltk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_and_extract_pagerank():
    """
    Download and extract the RageRank file from the AWS S3 bucket.
    The file is stored in the "pagerank" folder.
    """
    logger.info("Downloading and extracting RageRank files...")
    
    url = "https://danker.s3.amazonaws.com/2025-11-05.allwiki.links.rank.bz2"
    folder = "pagerank"
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Download file
    bz2_file = os.path.join(folder, "2025-11-05.allwiki.links.rank.bz2")
    logger.info(f"Downloading {url}...")
    urllib.request.urlretrieve(url, bz2_file)
    
    # Extract bz2 file
    output_file = os.path.join(folder, "2025-11-05.allwiki.links.rank")
    logger.info(f"Extracting {bz2_file}...")
    with bz2.open(bz2_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Delete bz2 file
    os.remove(bz2_file)
    logger.info(f"Deleted {bz2_file}")
    logger.info(f"Extraction complete: {output_file}")


if __name__ == "__main__":
    """
    Download the PageRank file and extract it.
    """
    nltk.download('punkt')
    download_and_extract_pagerank()
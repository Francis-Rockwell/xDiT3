import argparse
import logging
import random
import numpy as np
import torch
from cleanfid import fid
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fid_computation.log')
        ]
    )

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU

def compute_fid_score(ref_path: str, sample_path: str, device: str = "cuda") -> float:
    """
    Compute FID score
    
    Args:
        ref_path: Path to ref images directory
        sample_path: Path to sample images directory
        device: Computing device ('cuda' or 'cpu')
    
    Returns:
        float: FID score
    
    Raises:
        ValueError: If directory does not exist
    """
    # Check if paths exist
    ref_dir = Path(ref_path)
    gen_dir = Path(sample_path)
    
    if not ref_dir.exists():
        raise ValueError(f"ref images directory does not exist: {ref_path}")
    if not gen_dir.exists():
        raise ValueError(f"sample images directory does not exist: {sample_path}")
    
    logging.info(f"ref images directory: {ref_path}")
    logging.info(f"sample images directory: {sample_path}")
    
    try:
        score = fid.compute_fid(
            ref_path,
            sample_path,
            device=device,
            batch_size=1,
            num_workers=1
        )

        return score
        
    except Exception as e:
        logging.error(f"Error occurred during FID computation: {str(e)}")
        raise

def main():
    set_random_seed()

    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Compute FID score')
    parser.add_argument('--ref', type=str, required=True,
                      help='Path to ref images directory')
    parser.add_argument('--sample', type=str, required=True,
                      help='Path to sample images directory')
    parser.add_argument('--device', type=str, default="cuda",
                      choices=['cuda', 'cpu'], help='Computing device')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Compute FID
        score = compute_fid_score(args.ref, args.sample, args.device)
        
        # Output result
        logging.info(f"FID score: {score:.4f}")
        
    except Exception as e:
        logging.error(f"Program execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
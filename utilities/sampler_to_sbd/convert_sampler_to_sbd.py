#!/usr/bin/env python3
"""
Convert Qiskit Sampler JSON output to SBD determinant files.

This script uses qiskit-addon-sqd utility functions to properly convert
Sampler output (count_dict.json) into determinant files compatible with
the SBD solver.

Usage:
    python convert_sampler_to_sbd.py ../count_dict.json --norb 29
    python convert_sampler_to_sbd.py ../count_dict.json --norb 29 --output-dir ./sbd_input
"""

import json
import argparse
from pathlib import Path
import numpy as np

# Use qiskit-addon-sqd functions for proper conversion
from qiskit_addon_sqd.counts import counts_to_arrays, bitstring_matrix_to_integers
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
from qiskit_addon_sqd.configuration_recovery import recover_configurations


def convert_counts_to_sbd_format(
    counts_dict: dict,
    norb: int,
    nelec: tuple[int, int],
    output_dir: str = ".",
    prefix: str = "determinants",
    max_dets: int | None = None,
    rand_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert counts dictionary to SBD determinant files using qiskit-addon-sqd functions.
    
    Args:
        counts_dict: Dictionary mapping bitstrings to counts from Sampler
        norb: Number of spatial orbitals
        nelec: Tuple of (n_alpha, n_beta) number of electrons
        output_dir: Directory to save output files
        prefix: Prefix for output filenames
        max_dets: Maximum number of determinants to subsample (None = use all)
        rand_seed: Random seed for subsampling
        
    Returns:
        Tuple of (alpha_dets, beta_dets) as integer arrays
    """
    n_alpha, n_beta = nelec
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use qiskit-addon-sqd function to convert counts to arrays
    # This returns (bitstring_matrix, probabilities)
    bitstring_matrix_raw, probs_raw = counts_to_arrays(counts_dict)
    
    print(f"Loaded {len(bitstring_matrix_raw)} unique bitstrings")
    print(f"Bitstring length: {bitstring_matrix_raw.shape[1]} bits")
    print(f"Expected: {2 * norb} bits (2 * {norb} orbitals)")
    
    if bitstring_matrix_raw.shape[1] != 2 * norb:
        raise ValueError(
            f"Bitstring length {bitstring_matrix_raw.shape[1]} does not match "
            f"2 * norb = {2 * norb}"
        )
    
    # Use qiskit-addon-sqd configuration recovery to fix noisy bitstrings
    # This corrects bitstrings to have the proper electron count
    print(f"\nApplying configuration recovery (fixing noisy bitstrings)...")
    print(f"  Target: n_alpha={n_alpha}, n_beta={n_beta}")
    
    # For initial occupancies, use uniform distribution
    initial_occupancies = np.ones((2, norb)) * 0.5
    
    bitstring_matrix, probs = recover_configurations(
        bitstring_matrix_raw,
        probs_raw,
        initial_occupancies,
        n_alpha,
        n_beta,
        rand_seed=None
    )
    
    print(f"  Recovered {len(bitstring_matrix)} valid bitstrings")
    
    if len(bitstring_matrix) == 0:
        raise ValueError(
            f"Configuration recovery produced no valid bitstrings. "
            f"Check your nelec values (n_alpha={n_alpha}, n_beta={n_beta})."
        )
    
    # Subsample if max_dets is specified (to cap computational cost)
    if max_dets is not None and len(bitstring_matrix) > max_dets:
        print(f"\nSubsampling to {max_dets} determinants (from {len(bitstring_matrix)})...")
        # subsample() returns a list of batches; we just want one batch
        batches = subsample(bitstring_matrix, probs, max_dets, num_batches=1, rand_seed=rand_seed)
        bitstring_matrix = batches[0]
        # Recalculate probabilities for the subsampled batch (uniform for simplicity)
        probs = np.ones(len(bitstring_matrix)) / len(bitstring_matrix)
        print(f"  Subsampled to {len(bitstring_matrix)} determinants")
    
    # Split bitstrings into alpha (right half) and beta (left half)
    # Convention in qiskit-addon-sqd: [beta_bits | alpha_bits]
    beta_bitstrings = bitstring_matrix[:, :norb]
    alpha_bitstrings = bitstring_matrix[:, norb:]
    
    # Convert bitstring matrices to integers using qiskit-addon-sqd function
    alpha_dets = bitstring_matrix_to_integers(alpha_bitstrings)
    beta_dets = bitstring_matrix_to_integers(beta_bitstrings)
    
    print(f"\nConverted to determinants:")
    print(f"  Alpha determinants: {len(alpha_dets)}")
    print(f"  Beta determinants: {len(beta_dets)}")
    print(f"  Sample alpha det: {alpha_dets[0]} (binary: {format(alpha_dets[0], f'0{norb}b')})")
    print(f"  Sample beta det: {beta_dets[0]} (binary: {format(beta_dets[0], f'0{norb}b')})")
    
    # Get unique determinants for each spin
    unique_alpha, alpha_inverse = np.unique(alpha_dets, return_inverse=True)
    unique_beta, beta_inverse = np.unique(beta_dets, return_inverse=True)
    
    # Calculate marginal counts for sorting
    counts_array = probs * len(bitstring_matrix)
    
    alpha_counts = np.bincount(alpha_inverse, weights=counts_array)
    beta_counts = np.bincount(beta_inverse, weights=counts_array)
    
    # Sort by counts (descending)
    alpha_order = np.argsort(alpha_counts)[::-1]
    beta_order = np.argsort(beta_counts)[::-1]
    
    unique_alpha_sorted = unique_alpha[alpha_order]
    unique_beta_sorted = unique_beta[beta_order]
    
    print(f"\nUnique determinants (after filtering):")
    print(f"  Unique alpha: {len(unique_alpha_sorted)}")
    print(f"  Unique beta: {len(unique_beta_sorted)}")
    
    # Write unique determinants (sorted by marginal probability) in SBD format
    alpha_file = output_path / f"{prefix}_alpha.txt"
    beta_file = output_path / f"{prefix}_beta.txt"
    
    with open(alpha_file, "w") as f:
        for det in unique_alpha_sorted:
            binary_str = format(int(det), f"0{norb}b")
            f.write(binary_str + "\n")
    
    with open(beta_file, "w") as f:
        for det in unique_beta_sorted:
            binary_str = format(int(det), f"0{norb}b")
            f.write(binary_str + "\n")
    
    print(f"\nSaved unique determinants (sorted by marginal probability):")
    print(f"  Alpha: {alpha_file} ({len(unique_alpha_sorted)} determinants)")
    print(f"  Beta:  {beta_file} ({len(unique_beta_sorted)} determinants)")
    
    return alpha_dets, beta_dets


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert Qiskit Sampler JSON to SBD determinant files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to count_dict.json file from Qiskit Sampler"
    )
    parser.add_argument(
        "--norb",
        type=int,
        required=True,
        help="Number of spatial orbitals (bitstring length should be 2*norb)"
    )
    parser.add_argument(
        "--nelec",
        type=int,
        nargs=2,
        metavar=('N_ALPHA', 'N_BETA'),
        required=True,
        help="Number of alpha and beta electrons (e.g., --nelec 5 5 for 10 total electrons)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for determinant files"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="determinants",
        help="Prefix for output filenames"
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        default=None,
        help="Maximum number of determinants to subsample (default: use all). "
             "Useful for capping computational cost in SBD solver."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling"
    )
    
    args = parser.parse_args()
    
    # Load JSON file
    json_path = Path(args.json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        counts_dict = json.load(f)
    
    # Convert to SBD format
    convert_counts_to_sbd_format(
        counts_dict,
        args.norb,
        tuple(args.nelec),
        args.output_dir,
        args.prefix,
        args.max_dets,
        args.seed
    )
    
    print("\n" + "="*70)
    print("Conversion complete!")
    print("="*70)
    print("\nThe generated files can be used with SBD solver:")
    print("  - Use *_alpha.txt and *_beta.txt as --adetfile and --bdetfile")
    print("  - Or use *_alpha_unique.txt and *_beta_unique.txt for unique dets")
    print("\nExample SBD command:")
    print(f"  mpirun -np 4 sbd_diag --fcidump fcidump.txt \\")
    print(f"    --adetfile {args.prefix}_alpha_unique.txt \\")
    print(f"    --bdetfile {args.prefix}_beta_unique.txt")


if __name__ == "__main__":
    main()

# Made with Bob

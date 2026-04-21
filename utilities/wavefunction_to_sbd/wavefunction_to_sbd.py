"""Convert a wavefunction (determinants + coefficients) to SBD restart format.

SBD's --loadname option expects binary restart files, one per MPI rank in the
basis communicator (adet_comm_size * bdet_comm_size files). This script
produces those files from:

  1. Alpha/beta determinant files (text format, one bitstring per line)
  2. A coefficient matrix (dense, row-major, alpha-index varies slowest)

The coefficient matrix can come from:
  - SBD's --dump_matrix_form_wf output (.bin = raw float64, .npy = numpy)
  - PySCF's CI coefficients (saved as .npy)

Usage:
    python wavefunction_to_sbd.py \\
        --adetfile alpha_dets.txt \\
        --coefficients wavefunction.bin \\
        --norb 24 \\
        --output_prefix my_wf_ \\
        --adet_comm_size 2 \\
        --bdet_comm_size 2

    # Then load in SBD:
    mpirun -np 4 python run_sbd_diag.py --loadname my_wf_ \\
        --adet_comm_size 2 --bdet_comm_size 2 ...

File format reference: sbd/include/sbd/chemistry/tpb/restart.h
  - Header: adet_range (uint64), bdet_range (uint64), det_length (uint64)
  - Alpha determinants: adet_range * det_length uint64 values
  - Beta determinants: bdet_range * det_length uint64 values
  - Coefficients: adet_range * bdet_range float64 values (row-major)
"""

import argparse
import struct
from pathlib import Path

import numpy as np


def get_mpi_range(mpi_size: int, mpi_rank: int, total: int) -> tuple[int, int]:
    """Replicate SBD's get_mpi_range: partition [0, total) across mpi_size ranks.

    Returns (begin, end) for the given rank.
    Source: sbd/include/sbd/framework/mpi_utility.h
    """
    i_div = total // mpi_size
    i_res = total % mpi_size

    if mpi_rank < i_res:
        begin = (i_div + 1) * mpi_rank
        end = (i_div + 1) * (mpi_rank + 1)
    else:
        begin = (i_div + 1) * i_res + i_div * (mpi_rank - i_res)
        end = (i_div + 1) * i_res + i_div * (mpi_rank + 1 - i_res)

    return begin, end


def bitstring_to_sbd_words(bitstring: str, bit_length: int) -> list[int]:
    """Convert a text bitstring ("110100...") to SBD's word representation.

    SBD stores determinants as vectors of size_t, each holding bit_length bits.
    Orbital i is stored in word i // bit_length, bit position i % bit_length.
    The text format has the highest orbital on the left (index 0).

    Source: sbd/include/sbd/framework/bit_manipulation.h, from_string()
    """
    norb = len(bitstring)
    num_words = (norb + bit_length - 1) // bit_length
    words = [0] * num_words

    for i in range(norb):
        # bit_manipulation.h line 217: char bit = s[L - 1 - i]
        bit = bitstring[norb - 1 - i]
        if bit == '1':
            p = i % bit_length
            b = i // bit_length
            words[b] |= (1 << p)

    return words


def integer_to_sbd_words(value: int, norb: int, bit_length: int) -> list[int]:
    """Convert an integer determinant to SBD's word representation.

    Bit i of the integer represents orbital i occupancy.
    """
    num_words = (norb + bit_length - 1) // bit_length
    words = [0] * num_words

    for i in range(norb):
        if (value >> i) & 1:
            p = i % bit_length
            b = i // bit_length
            words[b] |= (1 << p)

    return words


def load_determinants_text(filepath: str, bit_length: int) -> tuple[list[list[int]], int]:
    """Load determinants from a text file (one bitstring per line).

    Returns:
        (determinants, norb) where determinants is a list of SBD word vectors.
    """
    dets = []
    norb = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if norb is None:
                norb = len(line)
            dets.append(bitstring_to_sbd_words(line, bit_length))
    return dets, norb


def load_determinants_npy(filepath: str, norb: int, bit_length: int) -> list[list[int]]:
    """Load determinants from a .npy file (array of integers).

    Each integer is a bit-packed determinant where bit i = orbital i.
    """
    arr = np.load(filepath)
    return [integer_to_sbd_words(int(v), norb, bit_length) for v in arr]


def load_coefficients(filepath: str) -> np.ndarray:
    """Load coefficient matrix from .bin (raw float64) or .npy file.

    Returns 1D array of float64 values in row-major order (alpha varies slowest).
    """
    ext = Path(filepath).suffix.lower()
    if ext == '.npy':
        arr = np.load(filepath)
        return arr.astype(np.float64).ravel()
    elif ext == '.bin':
        return np.fromfile(filepath, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported coefficient file format: {ext}")


def write_restart_files(
    output_prefix: str,
    alpha_dets: list[list[int]],
    beta_dets: list[list[int]],
    coefficients: np.ndarray,
    adet_comm_size: int,
    bdet_comm_size: int,
):
    """Write SBD restart files, one per basis communicator rank.

    Source: sbd/include/sbd/chemistry/tpb/restart.h, SaveWavefunction()
    """
    na = len(alpha_dets)
    nb = len(beta_dets)
    det_length = len(alpha_dets[0])

    if coefficients.size != na * nb:
        raise ValueError(
            f"Coefficient matrix size {coefficients.size} doesn't match "
            f"{na} alpha x {nb} beta = {na * nb}"
        )

    # Reshape to 2D for slicing
    coef_matrix = coefficients.reshape(na, nb)

    files_written = []
    for arank in range(adet_comm_size):
        for brank in range(bdet_comm_size):
            a_begin, a_end = get_mpi_range(adet_comm_size, arank, na)
            b_begin, b_end = get_mpi_range(bdet_comm_size, brank, nb)
            adet_range = a_end - a_begin
            bdet_range = b_end - b_begin

            rank = brank + arank * bdet_comm_size
            filename = f"{output_prefix}{rank:06d}"

            with open(filename, 'wb') as f:
                # Header: adet_range, bdet_range, det_length
                f.write(struct.pack('Q', adet_range))
                f.write(struct.pack('Q', bdet_range))
                f.write(struct.pack('Q', det_length))

                # Alpha determinants for this rank's partition
                for i in range(a_begin, a_end):
                    for word in alpha_dets[i]:
                        f.write(struct.pack('Q', word))

                # Beta determinants for this rank's partition
                for i in range(b_begin, b_end):
                    for word in beta_dets[i]:
                        f.write(struct.pack('Q', word))

                # Coefficients: row-major, alpha varies slowest
                for ia in range(a_begin, a_end):
                    row_slice = coef_matrix[ia, b_begin:b_end]
                    f.write(row_slice.tobytes())

            files_written.append(filename)

    return files_written


def main():
    parser = argparse.ArgumentParser(
        description="Convert wavefunction to SBD restart format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--adetfile', required=True,
                        help='Alpha determinants file (.txt or .npy)')
    parser.add_argument('--bdetfile', default='',
                        help='Beta determinants file (.txt or .npy). '
                             'If not given, uses adetfile (symmetric alpha=beta).')
    parser.add_argument('--coefficients', required=True,
                        help='Coefficient matrix file (.bin or .npy). '
                             'Row-major, shape (n_alpha, n_beta).')
    parser.add_argument('--norb', type=int, default=0,
                        help='Number of spatial orbitals (required for .npy det files, '
                             'auto-detected from .txt files)')
    parser.add_argument('--bit_length', type=int, default=20,
                        help='Bits per size_t word (must match SBD --bit_length)')
    parser.add_argument('--output_prefix', required=True,
                        help='Output file prefix (files: prefix000000, prefix000001, ...)')
    parser.add_argument('--adet_comm_size', type=int, default=1,
                        help='Alpha communicator size (must match SBD --adet_comm_size)')
    parser.add_argument('--bdet_comm_size', type=int, default=1,
                        help='Beta communicator size (must match SBD --bdet_comm_size)')

    args = parser.parse_args()

    # Load alpha determinants
    adet_ext = Path(args.adetfile).suffix.lower()
    if adet_ext == '.npy':
        if args.norb == 0:
            parser.error("--norb required when using .npy determinant files")
        alpha_dets = load_determinants_npy(args.adetfile, args.norb, args.bit_length)
        norb = args.norb
    else:
        alpha_dets, norb = load_determinants_text(args.adetfile, args.bit_length)
        if args.norb and args.norb != norb:
            print(f"Warning: --norb={args.norb} doesn't match detected norb={norb} from det file")

    # Load beta determinants
    if args.bdetfile:
        bdet_ext = Path(args.bdetfile).suffix.lower()
        if bdet_ext == '.npy':
            beta_dets = load_determinants_npy(args.bdetfile, norb, args.bit_length)
        else:
            beta_dets, _ = load_determinants_text(args.bdetfile, args.bit_length)
    else:
        beta_dets = alpha_dets  # Symmetric case

    # Load coefficients
    coefficients = load_coefficients(args.coefficients)

    na = len(alpha_dets)
    nb = len(beta_dets)
    det_length = len(alpha_dets[0])

    print(f"Alpha determinants: {na}")
    print(f"Beta determinants:  {nb}")
    print(f"Norb: {norb}, bit_length: {args.bit_length}, det_length: {det_length}")
    print(f"Coefficient matrix: {coefficients.size} values ({na} x {nb})")
    print(f"MPI topology: {args.adet_comm_size} x {args.bdet_comm_size} = "
          f"{args.adet_comm_size * args.bdet_comm_size} ranks")

    files = write_restart_files(
        output_prefix=args.output_prefix,
        alpha_dets=alpha_dets,
        beta_dets=beta_dets,
        coefficients=coefficients,
        adet_comm_size=args.adet_comm_size,
        bdet_comm_size=args.bdet_comm_size,
    )

    print(f"\nWrote {len(files)} restart files:")
    for f in files:
        print(f"  {f}")


if __name__ == '__main__':
    main()

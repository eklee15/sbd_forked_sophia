#!/usr/bin/env python3
"""Round-trip test: SBD save → Python convert → SBD load.

Validates that wavefunction_to_sbd.py produces correct SBD restart files by:
  1. SBD diagonalizes H2O and saves wavefunction (matrix form + restart files)
  2. Python reads the matrix form and determinants
  3. wavefunction_to_sbd.py converts them back to SBD restart format
  4. SBD loads the converted restart file and verifies identical energy

If the round-trip produces the same energy, the conversion is correct.

Usage:
    python test_roundtrip.py

    # With MPI
    mpirun -np 1 python test_roundtrip.py
"""

import os
import tempfile
from pathlib import Path

import numpy as np

from wavefunction_to_sbd import (
    bitstring_to_sbd_words,
    load_coefficients,
    write_restart_files,
)


def load_dets_as_sbd_words(filepath: str, bit_length: int) -> tuple[list[list[int]], int]:
    """Read SBD text determinant file and convert to SBD word-packed format.

    Returns (list_of_word_vectors, norb).
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


def main():
    # ─── Configuration ───────────────────────────────────────────────
    # H2O: 24 spatial orbitals, 10 electrons (5 alpha + 5 beta), singlet
    sbd_data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "h2o"
    fcidump_file = str(sbd_data_dir / "fcidump.txt")
    adetfile = str(sbd_data_dir / "h2o-1em3-alpha.txt")  # 275 determinants
    bit_length = 20  # SBD default; must match across all steps

    tmpdir = tempfile.mkdtemp(prefix="sbd_roundtrip_")
    print(f"Working directory: {tmpdir}")
    print()

    # ─── Step 1: SBD diagonalization ─────────────────────────────────
    # Save wavefunction in two formats:
    #   --dump_matrix_form_wf: raw float64 coefficients (no header), shape (na, nb)
    #   --savename: SBD binary restart files (one per MPI rank)
    print("=" * 70)
    print("Step 1: SBD diagonalization — save wavefunction")
    print("=" * 70)

    import sbd
    sbd.init(device="cpu")
    rank = sbd.get_rank()

    config = sbd.TPB_SBD()
    config.max_it = 300
    config.max_nb = 50
    config.eps = 1e-8
    config.bit_length = bit_length

    matrix_file = os.path.join(tmpdir, "wf.bin")
    sbd_save_prefix = os.path.join(tmpdir, "wf_sbd_")
    config.dump_matrix_form_wf = matrix_file

    results_orig = sbd.tpb_diag_from_files(
        fcidumpfile=fcidump_file,
        adetfile=adetfile,
        sbd_data=config,
        savename=sbd_save_prefix,
    )

    E_orig = results_orig["energy"]
    if rank == 0:
        print(f"  Energy: {E_orig:.10f} Hartree")
        print(f"  SBD restart: {sbd_save_prefix}000000")
        print(f"  Matrix form: {matrix_file}")
        print()

    # ─── Step 2: Python reads and converts back ──────────────────────
    # Read the matrix-form dump + determinant text file, then use
    # wavefunction_to_sbd.py to produce a new set of restart files.
    print("=" * 70)
    print("Step 2: Python convert — wavefunction_to_sbd.py")
    print("=" * 70)

    # Read determinants from the same text file SBD used
    alpha_dets, norb = load_dets_as_sbd_words(adetfile, bit_length)
    beta_dets = alpha_dets  # symmetric (no --bdetfile)
    na = len(alpha_dets)
    nb = len(beta_dets)

    # Read coefficient matrix from the .bin dump
    coef_flat = load_coefficients(matrix_file)

    if rank == 0:
        print(f"  Determinants: {na} alpha x {nb} beta, norb={norb}")
        print(f"  Coefficients: {coef_flat.size} values")

    # Write new restart files via our utility
    py_save_prefix = os.path.join(tmpdir, "wf_python_")
    files_written = write_restart_files(
        output_prefix=py_save_prefix,
        alpha_dets=alpha_dets,
        beta_dets=beta_dets,
        coefficients=coef_flat,
        adet_comm_size=1,
        bdet_comm_size=1,
    )

    if rank == 0:
        print(f"  Python restart: {files_written[0]}")

    # ─── Step 3: Byte-level comparison ───────────────────────────────
    # If our conversion is correct, the files should be byte-identical
    # to what SBD wrote (for single-rank case).
    print()
    print("=" * 70)
    print("Step 3: Byte-level comparison")
    print("=" * 70)

    if rank == 0:
        sbd_file = f"{sbd_save_prefix}000000"
        py_file = f"{py_save_prefix}000000"
        sbd_bytes = open(sbd_file, "rb").read()
        py_bytes = open(py_file, "rb").read()

        if sbd_bytes == py_bytes:
            print(f"  PASS: Files are byte-identical ({len(sbd_bytes):,} bytes)")
        else:
            print(f"  FAIL: Files differ!")
            print(f"    SBD file: {len(sbd_bytes):,} bytes")
            print(f"    Python file: {len(py_bytes):,} bytes")
            # Find first difference
            for i in range(min(len(sbd_bytes), len(py_bytes))):
                if sbd_bytes[i] != py_bytes[i]:
                    print(f"    First diff at byte {i}: SBD=0x{sbd_bytes[i]:02x} Python=0x{py_bytes[i]:02x}")
                    break
        print()

    # ─── Step 4: SBD loads the Python-produced restart file ──────────
    # Final validation: SBD loads our restart file and should converge
    # to the same energy immediately (warm start from exact solution).
    print("=" * 70)
    print("Step 4: SBD loads Python-produced restart file")
    print("=" * 70)

    config2 = sbd.TPB_SBD()
    config2.max_it = 300
    config2.max_nb = 50
    config2.eps = 1e-8
    config2.bit_length = bit_length

    results_reload = sbd.tpb_diag_from_files(
        fcidumpfile=fcidump_file,
        adetfile=adetfile,
        sbd_data=config2,
        loadname=py_save_prefix,
    )

    E_reload = results_reload["energy"]
    if rank == 0:
        print(f"  Energy (reloaded): {E_reload:.10f} Hartree")
        diff = abs(E_reload - E_orig)
        print(f"  Difference: {diff:.2e} Hartree")
        print()

    # ─── Summary ─────────────────────────────────────────────────────
    if rank == 0:
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"  Original energy:  {E_orig:.10f}")
        print(f"  Reloaded energy:  {E_reload:.10f}")
        print(f"  Difference:       {abs(E_reload - E_orig):.2e}")
        if abs(E_reload - E_orig) < 1e-8:
            print("  PASS: Round-trip conversion is correct")
        else:
            print("  FAIL: Energies differ — check bit_length or file format")
        print()
        print(f"  Working directory: {tmpdir}")

    sbd.finalize()


if __name__ == "__main__":
    main()

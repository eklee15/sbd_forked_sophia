"""SQD loop using Fulqrum eigensolver (alternative to qiskit-addon-sqd + SBD).

Fulqrum is a generalized quantum subspace eigensolver (Cython/C++).
This script demonstrates the SQD workflow using Fulqrum directly,
without qiskit-addon-sqd or SBD.

Requires: fulqrum, numpy, scipy

IMPORTANT — RHEL 9 / Fedora / CentOS Stream 9 compilation note:
    Fulqrum's C++ extensions have latent out-of-bounds vector accesses
    that crash on Linux distros with _GLIBCXX_ASSERTIONS enabled by default.
    Rebuild with assertions disabled:

        CXXFLAGS='-O3 -std=c++17 -ffast-math -fopenmp -U_GLIBCXX_ASSERTIONS -DNDEBUG' \
        pip install -e . --no-build-isolation --force-reinstall --no-deps

    macOS and Ubuntu/Debian are not affected.

Usage:
    # H2O with example bitstrings
    python run_sqd_fulqrum.py \
        --fcidump ../../data/h2o/fcidump.txt \
        --counts count_dict_h2o.json

    # N2 with hardware bitstrings
    python run_sqd_fulqrum.py \
        --fcidump /path/to/fcidump.txt \
        --counts /path/to/count_dict.json \
        --norb 29 --nelec_a 5 --nelec_b 5
"""

import argparse
import json
import re
import time
from collections import OrderedDict

import numpy as np
import scipy.sparse.linalg as spla

import fulqrum as fq
from fulqrum.convert.integrals import fcidump_to_fq_fermionic_op
from fulqrum.core.sqd import (
    postselect_by_hamming_right_and_left,
    subsample,
    recover_configurations,
    get_carryover_full_strs,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="SQD loop with Fulqrum eigensolver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fcidump", required=True, help="Path to FCIDUMP file")
    p.add_argument("--counts", required=True,
                   help="Path to count_dict.json (bitstring counts)")

    # System parameters (auto-detected from FCIDUMP if not given)
    p.add_argument("--norb", type=int, default=None,
                   help="Number of orbitals (auto-detected from FCIDUMP)")
    p.add_argument("--nelec_a", type=int, default=None,
                   help="Number of alpha electrons (auto-detected from FCIDUMP)")
    p.add_argument("--nelec_b", type=int, default=None,
                   help="Number of beta electrons (auto-detected from FCIDUMP)")

    # SQD loop parameters
    p.add_argument("--max_iterations", type=int, default=20)
    p.add_argument("--samples_per_batch", type=int, default=800)
    p.add_argument("--num_batches", type=int, default=3)
    p.add_argument("--carryover_threshold", type=float, default=1e-4)
    p.add_argument("--energy_tol", type=float, default=1e-5)
    p.add_argument("--convergence_window", type=int, default=3,
                   help="Stop early if energy change < tol for this many iters")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def parse_fcidump_header(path):
    """Return (norb, nelec_total, ms2) from FCIDUMP header."""
    with open(path) as f:
        header = f.readline()
    norb = int(re.search(r"NORB\s*=\s*(\d+)", header).group(1))
    nelec = int(re.search(r"NELEC\s*=\s*(\d+)", header).group(1))
    ms2 = int(re.search(r"MS2\s*=\s*(\d+)", header).group(1))
    return norb, nelec, ms2


def unique_alpha_beta_combined(bitstrings):
    """Get unique alpha and beta halves from full bitstrings."""
    if not bitstrings:
        return OrderedDict()
    unique_ab = OrderedDict()
    num_spatial_orb = len(bitstrings[0]) // 2
    for bs in bitstrings:
        a = bs[num_spatial_orb:]
        b = bs[:num_spatial_orb]
        unique_ab[a] = 1
        unique_ab[b] = 1
    return unique_ab


def main():
    args = parse_args()
    total_start = time.perf_counter()

    # --- Auto-detect system parameters from FCIDUMP ---
    norb_fcidump, nelec_total, ms2 = parse_fcidump_header(args.fcidump)
    norb = args.norb or norb_fcidump
    nelec_a = args.nelec_a or (nelec_total + ms2) // 2
    nelec_b = args.nelec_b or (nelec_total - ms2) // 2

    # --- Load FCIDUMP ---
    print(f"Loading FCIDUMP from {args.fcidump}")
    t0 = time.perf_counter()
    fermionic_op = fcidump_to_fq_fermionic_op(args.fcidump)
    print(f"  Fermionic operator terms: {fermionic_op.num_terms}")
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    fulqrum_operator = fermionic_op.extended_jw_transformation()
    print(f"  Qubit operator terms (after JW): {fulqrum_operator.num_terms}")
    print(f"  JW transform took {time.perf_counter() - t0:.2f}s")

    # --- Load bitstring samples ---
    print(f"\nLoading bitstring counts from {args.counts}")
    t0 = time.perf_counter()
    with open(args.counts) as f:
        counts = json.load(f)
    print(f"  Unique bitstrings: {len(counts)}")
    print(f"  Bitstring length: {len(next(iter(counts)))}")
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    raw_bitstrings = []
    raw_probs = []
    for bs, cnt in counts.items():
        raw_bitstrings.append(bs)
        raw_probs.append(cnt)
    raw_probs = np.array(raw_probs, dtype=float)
    raw_probs /= np.linalg.norm(raw_probs)

    # --- SQD Loop ---
    print(f"\nStarting SQD loop: max {args.max_iterations} iterations, "
          f"{args.samples_per_batch} half-strings/batch, {args.num_batches} batch(es)")
    print(f"System: {norb} orbitals, {nelec_a}a + {nelec_b}b electrons")
    print(f"Early stop: energy change < {args.energy_tol} for "
          f"{args.convergence_window} consecutive iters\n")

    current_occupancies = None
    carryover_full_strs = []
    energies = []
    converged = False

    for ni in range(args.max_iterations):
        iter_start = time.perf_counter()
        print(f"ITERATION {ni}")

        # Step 1: Postselect or recover configurations
        if current_occupancies is None:
            bitstrings, probs = postselect_by_hamming_right_and_left(
                raw_bitstrings, raw_probs, nelec_a, nelec_b
            )
        else:
            bitstrings, probs = recover_configurations(
                raw_bitstrings,
                raw_probs,
                current_occupancies[0],
                current_occupancies[1],
                nelec_a,
                nelec_b,
                args.seed,
            )

        print(f"  Postselected/recovered bitstrings: {len(bitstrings)}")

        # Step 2: Run multiple batches with different random seeds, keep best
        best_energy = np.inf
        best_occupancies = None
        best_carryover = []

        for bi in range(args.num_batches):
            batch_seed = args.seed + ni * args.num_batches + bi
            batch = subsample(
                bitstrings, probs, min(len(bitstrings), len(bitstrings)), batch_seed
            )

            # Build half-string set: HF state > carryover > recovered
            half_strs_dict = OrderedDict()

            # Hartree-Fock state
            hf_state = "0" * (norb - nelec_a) + "1" * nelec_a
            half_strs_dict[hf_state] = 1

            # Carryover from previous round
            for bs in unique_alpha_beta_combined(carryover_full_strs):
                half_strs_dict[bs] = 1

            # Recovered bitstrings
            for bs in unique_alpha_beta_combined(batch):
                half_strs_dict[bs] = 1

            half_strs = list(half_strs_dict.keys())

            # Truncate to samples_per_batch
            half_strs = half_strs[:args.samples_per_batch]
            half_strs.sort()

            # Step 3: Build subspace and project Hamiltonian
            t0 = time.perf_counter()
            S = fq.Subspace([half_strs, half_strs])
            sub_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            Hsub = fq.SubspaceHamiltonian(fulqrum_operator, S)
            Hsub_csr_linop = Hsub.to_csr_linearoperator_fast(verbose=False)
            proj_time = time.perf_counter() - t0
            mem_mb = Hsub_csr_linop.memory_size / (1024 * 1024)

            # Step 4: Initial guess
            diag_vec = Hsub.diagonal_vector()
            min_idx = np.where(diag_vec == diag_vec.min())[0]
            v0 = args.energy_tol * np.ones(len(S), dtype=Hsub.dtype)
            v0[min_idx] = 1

            # Step 5: Eigensolve
            t0 = time.perf_counter()
            eigvals, eigvecs = spla.eigsh(
                Hsub_csr_linop,
                k=1,
                which="SA",
                tol=args.energy_tol,
                v0=v0,
            )
            solve_time = time.perf_counter() - t0

            batch_energy = eigvals[0]
            print(f"  Batch {bi}: E={batch_energy:.10f}, "
                  f"{len(half_strs)} half-strs, "
                  f"proj={proj_time:.1f}s, solve={solve_time:.3f}s, "
                  f"CSR={mem_mb:.0f}MB")

            if batch_energy < best_energy:
                best_energy = batch_energy
                eigvecs_best = eigvecs.ravel()
                best_S = S
                best_Hsub = Hsub

        # Step 6: Use best batch for occupancies and carryover
        current_occupancies = best_S.get_orbital_occupancies(
            probs=np.abs(eigvecs_best) ** 2, norb=norb
        )

        carryover_full_strs_and_weights = get_carryover_full_strs(
            best_S, np.abs(eigvecs_best), args.carryover_threshold
        )
        carryover_full_strs = [item[0] for item in carryover_full_strs_and_weights]

        energies.append(best_energy)
        iter_time = time.perf_counter() - iter_start
        delta = abs(energies[-1] - energies[-2]) if len(energies) > 1 else float("inf")
        print(f"  Best E: {best_energy:.10f}, delta: {delta:.2e}, "
              f"carryover: {len(carryover_full_strs)}, time: {iter_time:.1f}s\n")

        # Early convergence check
        if len(energies) >= args.convergence_window + 1:
            recent_deltas = [
                abs(energies[-i] - energies[-i - 1])
                for i in range(1, args.convergence_window + 1)
            ]
            if all(d < args.energy_tol for d in recent_deltas):
                print(f"  Converged after {ni + 1} iterations "
                      f"(delta < {args.energy_tol} for "
                      f"{args.convergence_window} consecutive iters)\n")
                converged = True
                break

    # --- Summary ---
    total_time = time.perf_counter() - total_start
    print("=" * 60)
    print("SQD Loop Summary")
    print("=" * 60)
    print(f"System: NORB={norb}, NELEC={nelec_a + nelec_b}, MS2={ms2}")
    print(f"Iterations: {len(energies)}")
    print(f"Samples per batch: {args.samples_per_batch}")
    print(f"Total time: {total_time:.2f}s")
    print()
    for i, e in enumerate(energies):
        print(f"  Iter {i}: E = {e:.10f}")
    print(f"\n  Final electronic energy: {energies[-1]:.10f}")
    print(f"  Converged: {converged}")


if __name__ == "__main__":
    main()

"""
Test SBD solver with N2 molecule using qiskit-addon-sqd workflow.

This test mirrors the test_n2.py from qiskit-addon-dice-solver but uses
the SBD solver instead of DICE.
"""

import os
import sys
from pathlib import Path

import numpy as np
from pyscf import ao2mo, tools
from mpi4py import MPI

# Add parent directory to path to import sbd_solver
sys.path.insert(0, str(Path(__file__).parent.parent))
from sbd_solver import solve_sci_batch

from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from functools import partial


def test_n2_with_sbd():
    """Test N2 molecule diagonalization using SBD solver."""
    
    # Specify molecule properties
    num_orbitals = 16
    num_elec_a = num_elec_b = 5
    spin_sq = 0
    
    # Read in molecule from disk
    # Use the same molecule file as DICE test if available
    test_dir = Path(__file__).parent
    molecule_path = test_dir / "molecules" / "n2_fci.txt"
    
    # If molecule file doesn't exist, try to find it in qiskit-addon-dice-solver
    if not molecule_path.exists():
        dice_test_dir = Path(__file__).parent.parent.parent.parent / "qiskit-addon-dice-solver" / "test"
        molecule_path = dice_test_dir / "molecules" / "n2_fci.txt"
    
    if not molecule_path.exists():
        print(f"ERROR: Molecule file not found at {molecule_path}")
        print("Please ensure n2_fci.txt is available in the test/molecules directory")
        return
    
    print(f"Loading molecule from: {molecule_path}")
    mf_as = tools.fcidump.to_scf(str(molecule_path))
    hcore = mf_as.get_hcore()
    eri = ao2mo.restore(1, mf_as._eri, num_orbitals)
    nuclear_repulsion_energy = mf_as.mol.energy_nuc()
    
    # Create a seed to control randomness throughout this workflow
    rand_seed = np.random.default_rng(42)
    
    # Generate random samples
    print("Generating random bit array...")
    bit_array = generate_bit_array_uniform(10_000, num_orbitals * 2, rand_seed=rand_seed)
    
    # Run SQD
    result_history = []
    
    def callback(results: list[SCIResult]):
        result_history.append(results)
        iteration = len(result_history)
        print(f"Iteration {iteration}")
        for i, result in enumerate(results):
            print(f"\tSubsample {i}")
            print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy:.8f}")
            print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")
    
    # Configure SBD solver
    sbd_config = {
        "method": 0,        # Davidson method
        "eps": 1e-8,        # Convergence tolerance
        "max_it": 100,      # Max iterations
        "max_nb": 50,       # Max basis vectors
        "do_rdm": 0,        # Only compute density
        "carryover_type": 1,
        "threshold": 1e-4,
    }
    
    # Create configured SBD solver
    print("\nConfiguring SBD solver...")
    print(f"SBD config: {sbd_config}")
    sbd_solver = partial(
        solve_sci_batch,
        mpi_comm=MPI.COMM_WORLD,
        sbd_config=sbd_config,
    )
    
    print("\nStarting SQD with SBD solver...")
    print(f"Number of orbitals: {num_orbitals}")
    print(f"Number of electrons: ({num_elec_a}, {num_elec_b})")
    print(f"Samples per batch: 300")
    print(f"Number of batches: 5")
    print(f"Max iterations: 5")
    print()
    
    result = diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        samples_per_batch=300,
        norb=num_orbitals,
        nelec=(num_elec_a, num_elec_b),
        num_batches=5,
        max_iterations=5,
        sci_solver=sbd_solver,
        symmetrize_spin=True,
        callback=callback,
        seed=rand_seed,
    )
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Exact energy:     -109.10288938")
    print(f"Estimated energy: {result.energy + nuclear_repulsion_energy:.8f}")
    print(f"Error:            {abs(result.energy + nuclear_repulsion_energy + 109.10288938):.8e}")
    print()
    
    # Print convergence history
    if result_history:
        print("Convergence History:")
        print("-" * 60)
        for i, results in enumerate(result_history):
            energies = [r.energy + nuclear_repulsion_energy for r in results]
            min_energy = min(energies)
            max_energy = max(energies)
            avg_energy = np.mean(energies)
            print(f"Iteration {i+1}: min={min_energy:.8f}, max={max_energy:.8f}, avg={avg_energy:.8f}")
    
    # Check if result is reasonable (within 1 mHa of exact)
    error = abs(result.energy + nuclear_repulsion_energy + 109.10288938)
    if error < 0.001:
        print("\n✓ Test PASSED: Energy within 1 mHa of exact value")
    else:
        print(f"\n✗ Test FAILED: Energy error {error:.6f} Ha exceeds threshold")
    
    return result


if __name__ == "__main__":
    print("="*60)
    print("Testing SBD Solver with N2 Molecule")
    print("="*60)
    print()
    
    # Check MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"MPI Size: {size}")
        print(f"MPI Rank: {rank}")
        print()
    
    try:
        result = test_n2_with_sbd()
        if rank == 0:
            print("\nTest completed successfully!")
    except Exception as e:
        if rank == 0:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise

# Made with Bob

"""
Test SBD solver with N2 molecule using qiskit-addon-sqd workflow.

This test mirrors the test_n2.py from qiskit-addon-dice-solver but uses
the SBD solver instead of DICE.

Usage:
    # Auto-detect device (default)
    mpirun -np 4 python test_n2_sbd.py
    
    # Force CPU
    mpirun -np 4 python test_n2_sbd.py --device cpu
    
    # Force GPU
    mpirun -np 4 python test_n2_sbd.py --device gpu
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from pyscf import ao2mo, tools
from mpi4py import MPI

# Import from sbd package
from sbd.sbd_solver import solve_sci_batch
from sbd.device_config import DeviceConfig, print_device_info

from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from functools import partial


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test SBD solver with N2 molecule',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Device to use (auto=detect, cpu=force CPU, gpu=force GPU)')
    parser.add_argument('--max-memory-gb', type=int, default=-1,
                       help='Maximum GPU memory in GB (-1=auto, GPU only)')
    return parser.parse_args()


def test_n2_with_sbd(device_config=None):
    """Test N2 molecule diagonalization using SBD solver.
    
    Args:
        device_config: DeviceConfig object for CPU/GPU selection. If None, uses auto-detect.
    """
    
    # Specify molecule properties
    num_orbitals = 16
    num_elec_a = num_elec_b = 5
    spin_sq = 0
    
    # Read in molecule from disk
    # Use the same molecule file as DICE test if available
    test_dir = Path(__file__).parent
    molecule_path = test_dir / "molecules" / "n2_fci.txt"
    
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
    
    # Configure device (CPU/GPU)
    if device_config is None:
        device_config = DeviceConfig.auto()
    
    # Configure SBD solver
    sbd_config = {
        "method": 0,        # Davidson method
        "eps": 1e-8,        # Convergence tolerance
        "max_it": 100,      # Max iterations
        "max_nb": 50,       # Max basis vectors
        "do_rdm": 0,        # Only compute density
        #"carryover_type": 1,
        "threshold": 1e-8,
    }
    
    # Apply device configuration to SBD config
    # Note: This adds GPU-specific parameters if using GPU
    if device_config.use_gpu:
        sbd_config["use_precalculated_dets"] = device_config.use_precalculated_dets
        sbd_config["max_memory_gb_for_determinants"] = device_config.max_memory_gb
    
    # Create configured SBD solver
    print("\nConfiguring SBD solver...")
    print(f"Device: {device_config}")
    print(f"SBD config: {sbd_config}")
    sbd_solver = partial(
        solve_sci_batch,
        mpi_comm=MPI.COMM_WORLD,
        sbd_config=sbd_config,
        device_config=device_config,
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
    # Parse command line arguments
    args = parse_args()
    
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
        
        # Print device information
        print_device_info()
        print()
    
    # Configure device based on command line argument
    if args.device == 'auto':
        device_config = DeviceConfig.auto(max_memory_gb=args.max_memory_gb)
    elif args.device == 'cpu':
        device_config = DeviceConfig.cpu()
    else:  # gpu
        device_config = DeviceConfig.gpu(max_memory_gb=args.max_memory_gb)
    
    if rank == 0:
        print(f"Selected device configuration: {device_config}")
        print()
    
    try:
        result = test_n2_with_sbd(device_config=device_config)
        if rank == 0:
            print("\nTest completed successfully!")
    except Exception as e:
        if rank == 0:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise

# Made with Bob

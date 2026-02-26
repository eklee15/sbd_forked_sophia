"""
Test SBD solver with molecules using qiskit-addon-sqd workflow.

This test uses the SBD solver with the qiskit-addon-sqd framework for
selected configuration interaction calculations.

Usage:
    # N2 molecule with auto-detect device
    mpirun -np 4 python test_n2_sbd.py
    
    # H2O molecule with GPU
    mpirun -np 4 python test_n2_sbd.py --molecule h2o --device gpu
    
    # Custom molecule file
    mpirun -np 4 python test_n2_sbd.py --fcidump /path/to/fcidump.txt --norb 16 --nelec 10
    
    # With MPI configuration
    mpirun -np 8 python test_n2_sbd.py --adet-comm-size 2 --bdet-comm-size 2
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from pyscf import ao2mo, tools
from mpi4py import MPI

from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from functools import partial


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test SBD solver with molecules using qiskit-addon-sqd',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Device selection
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Device to use (auto=detect, cpu=force CPU, gpu=force GPU)')
    parser.add_argument('--max_memory_gb', type=int, default=-1,
                       help='Maximum GPU memory in GB (-1=auto, GPU only)')
    
    # Molecule input
    parser.add_argument('--fcidump', type=str, default=None,
                       help='Path to FCIDUMP file (default: molecules/n2_fci.txt)')
    parser.add_argument('--adetfile', type=str, default=None,
                       help='Path to alpha determinants file (if provided, uses these instead of random samples)')
    parser.add_argument('--bdetfile', type=str, default=None,
                       help='Path to beta determinants file (optional, uses adetfile if not specified)')
    
    # SQD parameters
    parser.add_argument('--samples', type=int, default=10000,
                       help='Total number of random samples to generate')
    parser.add_argument('--samples_per_batch', type=int, default=300,
                       help='Number of samples per batch')
    parser.add_argument('--num_batches', type=int, default=5,
                       help='Number of batches')
    parser.add_argument('--max_iterations', type=int, default=5,
                       help='Maximum SQD iterations')
    
    # SBD solver parameters
    parser.add_argument('--method', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Diagonalization method: 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham')
    parser.add_argument('--eps', type=float, default=1e-8,
                       help='SBD convergence tolerance')
    parser.add_argument('--max_it', type=int, default=100,
                       help='SBD maximum iterations')
    parser.add_argument('--max_nb', type=int, default=50,
                       help='SBD maximum basis vectors')
    parser.add_argument('--do_rdm', type=int, default=1, choices=[0, 1],
                       help='Calculate RDM (0=density only, 1=full RDM)')
    parser.add_argument('--threshold', type=float, default=1e-4,
                       help='Carryover threshold')
    
    # MPI configuration
    parser.add_argument('--adet_comm_size', type=int, default=1,
                       help='Alpha determinant communicator size')
    parser.add_argument('--bdet_comm_size', type=int, default=1,
                       help='Beta determinant communicator size')
    parser.add_argument('--task_comm_size', type=int, default=1,
                       help='Task communicator size')
    
    return parser.parse_args()


def load_determinants_from_file(adetfile, bdetfile, norb):
    """
    Load determinants from text files.
    
    Args:
        adetfile: Path to alpha determinants file
        bdetfile: Path to beta determinants file (optional)
        norb: Number of orbitals
        
    Returns:
        Bit array compatible with qiskit-addon-sqd
    """
    adetfile = Path(adetfile)
    if not adetfile.exists():
        raise FileNotFoundError(f"Alpha determinants file not found: {adetfile}")
    
    # Read alpha determinants
    with open(adetfile, 'r') as f:
        alpha_dets = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Convert binary string to integer
                alpha_dets.append(int(line, 2))
    
    # Read beta determinants (or use alpha if not specified)
    if bdetfile:
        bdetfile = Path(bdetfile)
        if not bdetfile.exists():
            raise FileNotFoundError(f"Beta determinants file not found: {bdetfile}")
        with open(bdetfile, 'r') as f:
            beta_dets = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    beta_dets.append(int(line, 2))
    else:
        # Use same determinants for beta
        beta_dets = alpha_dets.copy()
    
    # Create bit array by combining alpha and beta determinants
    # Format: each row is [alpha_bits | beta_bits] as a single integer
    bit_array = []
    for alpha in alpha_dets:
        for beta in beta_dets:
            # Combine: alpha in lower bits, beta in upper bits
            combined = alpha | (beta << norb)
            bit_array.append(combined)
    
    return np.array(bit_array, dtype=np.int64)


def get_molecule_data(args):
    """Get molecule data from FCIDUMP file."""
    test_dir = Path(__file__).parent
    
    # Use provided fcidump or default to N2
    if args.fcidump:
        fcidump_path = Path(args.fcidump)
    else:
        fcidump_path = test_dir / "molecules" / "n2_fci.txt"
    
    if not fcidump_path.exists():
        raise FileNotFoundError(f"FCIDUMP file not found: {fcidump_path}")
    
    # Read FCIDUMP to get norb and nelec
    print(f"Parsing {fcidump_path}")
    mf_as = tools.fcidump.to_scf(str(fcidump_path))
    
    # Extract molecule info from FCIDUMP
    # Note: mf_as.mol.nao_nr() returns 0 for FCIDUMP files, so we need to read the header
    with open(fcidump_path, 'r') as f:
        first_line = f.readline()
        # Parse NORB and NELEC from header like: &FCI NORB=  16,NELEC=10,MS2=0,
        import re
        norb_match = re.search(r'NORB\s*=\s*(\d+)', first_line)
        nelec_match = re.search(r'NELEC\s*=\s*(\d+)', first_line)
        
        if not norb_match or not nelec_match:
            raise ValueError(f"Could not parse NORB and NELEC from FCIDUMP header: {first_line}")
        
        norb = int(norb_match.group(1))
        nelec_total = int(nelec_match.group(1))
    
    # Assume closed shell or equal alpha/beta for simplicity
    nelec_a = nelec_total // 2
    nelec_b = nelec_total - nelec_a
    
    # Try to guess molecule name from filename
    molecule_name = fcidump_path.stem.upper()
    
    # Known exact energies for validation
    exact_energies = {
        'N2_FCI': -109.10288938,
        'H2O': -76.24,
    }
    exact_energy = exact_energies.get(molecule_name, None)
    
    return {
        'name': molecule_name,
        'fcidump': fcidump_path,
        'norb': norb,
        'nelec': (nelec_a, nelec_b),
        'exact_energy': exact_energy,
        'mf_as': mf_as,  # Return for later use
    }


def test_molecule_with_sbd(molecule_data, args, device_config=None):
    """Test molecule diagonalization using SBD solver.
    
    Args:
        molecule_data: Dictionary with molecule information
        args: Command line arguments
        device_config: DeviceConfig object for CPU/GPU selection. If None, uses auto-detect.
    """
    
    # Extract molecule properties
    molecule_name = molecule_data['name']
    num_orbitals = molecule_data['norb']
    num_elec_a, num_elec_b = molecule_data['nelec']
    exact_energy = molecule_data['exact_energy']
    mf_as = molecule_data['mf_as']
    spin_sq = 0
    
    print(f"Molecule: {molecule_name}")
    print(f"  Orbitals: {num_orbitals}")
    print(f"  Electrons: ({num_elec_a}, {num_elec_b})")
    if exact_energy:
        print(f"  Exact energy: {exact_energy:.8f} Ha")
    print()
    
    hcore = mf_as.get_hcore()
    eri = ao2mo.restore(1, mf_as._eri, num_orbitals)
    nuclear_repulsion_energy = mf_as.mol.energy_nuc()
    
    # Create a seed to control randomness throughout this workflow
    rand_seed = np.random.default_rng(42)
    
    # Generate or load samples
    if args.adetfile:
        # Load determinants from file
        print(f"Loading determinants from: {args.adetfile}")
        bit_array = load_determinants_from_file(args.adetfile, args.bdetfile, num_orbitals)
        print(f"Loaded {len(bit_array)} determinants")
    else:
        # Generate random samples
        print(f"Generating {args.samples} random samples...")
        bit_array = generate_bit_array_uniform(args.samples, num_orbitals * 2, rand_seed=rand_seed)
    
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
    
    # Configure SBD solver from command line arguments
    sbd_config = {
        "method": args.method,
        "eps": args.eps,
        "max_it": args.max_it,
        "max_nb": args.max_nb,
        "do_rdm": args.do_rdm,
        "carryover_type": 1,
        "threshold": args.threshold,
        "adet_comm_size": args.adet_comm_size,
        "bdet_comm_size": args.bdet_comm_size,
        "task_comm_size": args.task_comm_size,
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
    
    print(f"\nStarting SQD with SBD solver for {molecule_name}...")
    print(f"Number of orbitals: {num_orbitals}")
    print(f"Number of electrons: ({num_elec_a}, {num_elec_b})")
    print(f"Samples per batch: {args.samples_per_batch}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Max iterations: {args.max_iterations}")
    print()
    
    result = diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        samples_per_batch=args.samples_per_batch,
        norb=num_orbitals,
        nelec=(num_elec_a, num_elec_b),
        num_batches=args.num_batches,
        max_iterations=args.max_iterations,
        sci_solver=sbd_solver,
        symmetrize_spin=True,
        callback=callback,
        seed=rand_seed,
    )
    
    # Print final results
    print("\n" + "="*60)
    print(f"FINAL RESULTS - {molecule_name}")
    print("="*60)
    if exact_energy is not None:
        print(f"Exact energy:     {exact_energy:.8f}")
        print(f"Estimated energy: {result.energy + nuclear_repulsion_energy:.8f}")
        print(f"Error:            {abs(result.energy + nuclear_repulsion_energy - exact_energy):.8e}")
    else:
        print(f"Estimated energy: {result.energy + nuclear_repulsion_energy:.8f}")
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
    if exact_energy is not None:
        error = abs(result.energy + nuclear_repulsion_energy - exact_energy)
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
    
    # Import sbd modules first
    import sbd
    from sbd.sbd_solver import solve_sci_batch, _backend_info
    from sbd.device_config import DeviceConfig, print_device_info
    
    # Initialize SBD with the selected device
    # This must be done before using any SBD functions
    device_str = args.device if args.device != 'auto' else ('gpu' if DeviceConfig._check_cuda() else 'cpu')
    
    if rank == 0:
        print(f"Initializing SBD with device: {device_str}")
        print()
    
    try:
        sbd.init(device=device_str, comm_backend='mpi')
    except RuntimeError as e:
        if "already called" not in str(e):
            raise
    
    if rank == 0:
        # Print device information
        print_device_info()
        print()
        print(f"SBD Backend Info:")
        print(f"  Selected: {_backend_info['selected'].upper() if _backend_info['selected'] else 'UNKNOWN'}")
        print(f"  CPU Available: {_backend_info['cpu_available']}")
        print(f"  GPU Available: {_backend_info['gpu_available']}")
        print()
    
    # Create device config object (for configuration only, backend already selected)
    if args.device == 'auto':
        device_config = DeviceConfig.auto(max_memory_gb=args.max_memory_gb)
    elif args.device == 'cpu':
        device_config = DeviceConfig.cpu()
    else:  # gpu
        device_config = DeviceConfig.gpu(max_memory_gb=args.max_memory_gb)
    
    if rank == 0:
        print(f"Device configuration: {device_config}")
        print()
    
    try:
        # Get molecule data
        molecule_data = get_molecule_data(args)
        
        result = test_molecule_with_sbd(molecule_data, args, device_config=device_config)
        if rank == 0:
            print("\nTest completed successfully!")
    except Exception as e:
        if rank == 0:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise

# Made with Bob

#!/usr/bin/env python3
"""
Quantum-Classical Hybrid Workflow Example

This demonstrates how to use SBD's MPI communication primitives to implement
a quantum-classical hybrid workflow similar to IBM's Qiskit SQD + SBD demo.

Workflow:
1. Rank 0 performs quantum sampling (simulated here)
2. Rank 0 broadcasts quantum samples to all ranks
3. All ranks convert quantum bitstrings to SBD determinants
4. All ranks perform SBD diagonalization with quantum-derived determinants
5. Results are gathered and analyzed

Usage:
    mpirun -np 8 python quantum_classical_hybrid.py
"""

import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Quantum-classical hybrid workflow with SBD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Compute device')
    parser.add_argument('--fcidump', default='../../data/h2o/fcidump.txt',
                       help='Path to FCIDUMP file')
    parser.add_argument('--num_quantum_samples', type=int, default=10,
                       help='Number of quantum samples to simulate')
    parser.add_argument('--adet_comm_size', type=int, default=2,
                       help='Alpha determinant communicator size')
    parser.add_argument('--bdet_comm_size', type=int, default=2,
                       help='Beta determinant communicator size')
    parser.add_argument('--task_comm_size', type=int, default=2,
                       help='Task communicator size')
    parser.add_argument('--bit_length', type=int, default=20,
                       help='Bit length for determinant representation')
    
    return parser.parse_args()

def simulate_quantum_sampling(num_samples=10):
    """
    Simulate quantum circuit sampling (rank 0 only).
    
    In a real workflow, this would:
    1. Prepare quantum circuit
    2. Submit to quantum hardware
    3. Receive measurement results as bitstrings
    
    For this demo, we simulate realistic quantum measurement results.
    """
    # Simulate quantum measurements for H2O (10 electrons, 20 orbitals)
    # Ground state and nearby excited states
    
    quantum_samples = {
        'bitstrings': [
            '0x1f001f',  # Ground state: orbitals 0-4 occupied (alpha and beta)
            '0x2f001f',  # Single excitation: alpha electron 4→5
            '0x3e001f',  # Single excitation: alpha electron 4→5, different config
            '0x1f003e',  # Single excitation: beta electron 4→5
            '0x2f003e',  # Double excitation
            '0x1f002f',  # Beta excitation
            '0x3d001f',  # Different alpha excitation
            '0x1f003d',  # Different beta excitation
            '0x2e001f',  # Another excitation
            '0x1f002e',  # Another beta excitation
        ][:num_samples],
        'counts': [523, 245, 156, 76, 45, 32, 28, 21, 18, 16][:num_samples],
        'num_shots': 1000,
        'num_orbitals': 10,  # For H2O
        'num_electrons': 10,
    }
    
    return quantum_samples

def bitstring_to_determinant(bitstring, num_orbitals=10):
    """
    Convert quantum measurement bitstring to determinant format.
    
    Args:
        bitstring: Hex string like '0x1f001f'
        num_orbitals: Number of orbitals
    
    Returns:
        (alpha_det, beta_det): Lists of occupied orbital indices
    """
    # Convert hex to binary
    binary = bin(int(bitstring, 16))[2:].zfill(num_orbitals * 2)
    
    # Split into alpha and beta
    alpha_bits = binary[:num_orbitals]
    beta_bits = binary[num_orbitals:]
    
    # Find occupied orbitals
    alpha_det = [i for i, bit in enumerate(alpha_bits) if bit == '1']
    beta_det = [i for i, bit in enumerate(beta_bits) if bit == '1']
    
    return alpha_det, beta_det

def determinant_to_sbd_format(orbital_list, bit_length=20):
    """
    Convert orbital list to SBD bit array format.
    
    Args:
        orbital_list: List of occupied orbital indices, e.g., [0,1,2,3,4]
        bit_length: Total number of orbitals (bits)
    
    Returns:
        List of size_t representing the determinant in SBD format
    
    Example:
        [0,1,2,3,4] with bit_length=20 -> bit array with first 5 bits set
        This matches the format in h2o-1em3-alpha.txt: "000000000000000000011111"
    """
    # SBD uses size_t arrays to represent determinants
    # Each size_t can hold 64 bits (on 64-bit systems)
    bits_per_word = 64
    num_words = (bit_length + bits_per_word - 1) // bits_per_word
    
    # Initialize array of size_t (represented as list of integers)
    det_array = [0] * num_words
    
    # Set bits for occupied orbitals
    for orbital in orbital_list:
        word_idx = orbital // bits_per_word
        bit_idx = orbital % bits_per_word
        det_array[word_idx] |= (1 << bit_idx)
    
    return det_array

def main():
    args = parse_args()
    
    # Import SBD
    import sbd
    
    # Initialize SBD with MPI backend
    sbd.init(device=args.device, comm_backend='mpi')
    
    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    # Verify MPI configuration
    expected_ranks = args.task_comm_size * args.adet_comm_size * args.bdet_comm_size
    if size != expected_ranks:
        if rank == 0:
            print(f"ERROR: MPI ranks ({size}) != task_comm_size × adet_comm_size × bdet_comm_size ({expected_ranks})")
            print(f"Please run with: mpirun -np {expected_ranks} python {sys.argv[0]}")
        sbd.finalize()
        return 1
    
    try:
        if rank == 0:
            print("="*70)
            print("Quantum-Classical Hybrid Workflow with SBD")
            print("="*70)
            print(f"Device: {sbd.get_device().upper()}")
            print(f"MPI ranks: {size}")
            print(f"MPI configuration: {args.task_comm_size} × {args.adet_comm_size} × {args.bdet_comm_size}")
            print("="*70)
        
        # ====================================================================
        # Step 1: Rank 0 performs quantum sampling
        # ====================================================================
        if rank == 0:
            print("\n[Step 1] Rank 0: Performing quantum sampling...")
            quantum_samples = simulate_quantum_sampling(args.num_quantum_samples)
            print(f"  Received {len(quantum_samples['bitstrings'])} unique configurations")
            print(f"  Total shots: {quantum_samples['num_shots']}")
            print(f"  Top 3 configurations:")
            for i in range(min(3, len(quantum_samples['bitstrings']))):
                bitstring = quantum_samples['bitstrings'][i]
                count = quantum_samples['counts'][i]
                prob = count / quantum_samples['num_shots']
                print(f"    {bitstring}: {count} counts ({prob:.1%})")
        else:
            quantum_samples = None
        
        # ====================================================================
        # Step 2: Broadcast quantum samples from rank 0 to all ranks
        # ====================================================================
        if rank == 0:
            print("\n[Step 2] Broadcasting quantum samples to all ranks...")
        
        quantum_samples = sbd.broadcast(quantum_samples, root=0)
        sbd.barrier()
        
        if rank == 0:
            print(f"  ✓ All {size} ranks now have quantum samples")
        
        # ====================================================================
        # Step 3: All ranks convert quantum samples to SBD determinants
        # ====================================================================
        if rank == 0:
            print("\n[Step 3] All ranks converting quantum samples to determinants...")
        
        # Convert bitstrings to determinants
        alpha_dets_list = []
        beta_dets_list = []
        
        for bitstring in quantum_samples['bitstrings']:
            alpha_orbitals, beta_orbitals = bitstring_to_determinant(
                bitstring,
                quantum_samples['num_orbitals']
            )
            
            # Convert to SBD format
            alpha_det_sbd = determinant_to_sbd_format(alpha_orbitals, args.bit_length)
            beta_det_sbd = determinant_to_sbd_format(beta_orbitals, args.bit_length)
            
            alpha_dets_list.append(alpha_det_sbd)
            beta_dets_list.append(beta_det_sbd)
        
        # Remove duplicates (keep unique determinants)
        unique_alpha = []
        unique_beta = []
        seen_alpha = set()
        seen_beta = set()
        
        for alpha_det in alpha_dets_list:
            alpha_tuple = tuple(alpha_det)
            if alpha_tuple not in seen_alpha:
                seen_alpha.add(alpha_tuple)
                unique_alpha.append(alpha_det)
        
        for beta_det in beta_dets_list:
            beta_tuple = tuple(beta_det)
            if beta_tuple not in seen_beta:
                seen_beta.add(beta_tuple)
                unique_beta.append(beta_det)
        
        if rank == 0:
            print(f"  Unique alpha determinants: {len(unique_alpha)}")
            print(f"  Unique beta determinants: {len(unique_beta)}")
            print(f"  Subspace dimension: {len(unique_alpha)} × {len(unique_beta)} = {len(unique_alpha) * len(unique_beta)}")
            print(f"  Example alpha determinant: {unique_alpha[0]}")
        
        # ====================================================================
        # Step 4: All ranks perform SBD diagonalization
        # ====================================================================
        if rank == 0:
            print("\n[Step 4] All ranks performing SBD diagonalization...")
            print("  (Using quantum-derived determinants)")
        
        # Load FCIDUMP
        fcidump = sbd.LoadFCIDump(args.fcidump)
        
        # Configure SBD
        config = sbd.TPB_SBD()
        config.adet_comm_size = args.adet_comm_size
        config.bdet_comm_size = args.bdet_comm_size
        config.task_comm_size = args.task_comm_size
        config.max_it = 100
        config.eps = 1e-3  # 1em3 tolerance
        config.bit_length = args.bit_length
        
        # Call SBD with quantum-derived determinants
        # All ranks have the same determinant lists
        results = sbd.tpb_diag(
            fcidump=fcidump,
            adet=unique_alpha,
            bdet=unique_beta,
            sbd_data=config,
            loadname="",
            savename=""
        )
        
        # ====================================================================
        # Step 5: Gather and analyze results
        # ====================================================================
        if rank == 0:
            print("\n[Step 5] Analyzing results...")
        
        # Compute statistics
        num_alpha = len(unique_alpha)
        num_beta = len(unique_beta)
        subspace_dim = num_alpha * num_beta
        
        # Use all_reduce to verify all ranks have same counts
        total_alpha_check = sbd.all_reduce(num_alpha, op='sum')
        total_beta_check = sbd.all_reduce(num_beta, op='sum')
        
        if rank == 0:
            # Verify consistency
            expected_alpha_sum = num_alpha * size
            expected_beta_sum = num_beta * size
            
            if total_alpha_check == expected_alpha_sum and total_beta_check == expected_beta_sum:
                print(f"  ✓ All ranks have consistent determinant counts")
            else:
                print(f"  ⚠ Warning: Inconsistent determinant counts across ranks")
        
        if rank == 0:
            print("\n" + "="*70)
            print("Results")
            print("="*70)
            print(f"Ground state energy: {results['energy']:.6f} Hartree")
            print(f"Quantum samples used: {len(quantum_samples['bitstrings'])}")
            print(f"Unique alpha determinants: {num_alpha}")
            print(f"Unique beta determinants: {num_beta}")
            print(f"Subspace dimension: {subspace_dim}")
            
            print(f"\nDensity (first 10 elements):")
            density_str = ", ".join([f"{d:.4f}" for d in results['density'][:10]])
            print(f"  [{density_str}, ...]")
            
            print("="*70)
            print("✓ Quantum-classical hybrid workflow completed successfully!")
            print("="*70)
            print("\nKey Points:")
            print("  • Rank 0 simulated quantum sampling")
            print("  • Quantum samples broadcast to all ranks")
            print("  • All ranks converted bitstrings to determinants")
            print("  • All ranks called SBD with same determinant lists")
            print("  • SBD internally distributed work across ranks")
            print("="*70)
        
        return_code = 0
        
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        return_code = 1
    
    finally:
        # Clean up SBD resources
        sbd.finalize()
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())

# Made with Bob

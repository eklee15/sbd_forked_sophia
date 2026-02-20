#!/usr/bin/env python3
"""
H2O calculation with simplified SBD API (no mpi4py needed!)

This example demonstrates the new simplified API where MPI is handled internally.

Usage:
    # CPU backend
    mpirun -np 8 -x OMP_NUM_THREADS=4 python h2o_simplified.py --device cpu
    
    # GPU backend
    mpirun -np 8 python h2o_simplified.py --device gpu
    
    # Auto-detect (default)
    mpirun -np 8 python h2o_simplified.py
"""

import argparse
import sys

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='H2O calculation with simplified SBD API',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Compute device')
    parser.add_argument('--fcidump', default='../../data/h2o/fcidump.txt',
                       help='Path to FCIDUMP file')
    parser.add_argument('--adetfile', default='../../data/h2o/h2o-1em4-alpha.txt',
                       help='Path to alpha determinants file')
    parser.add_argument('--max_it', type=int, default=100,
                       help='Maximum iterations')
    parser.add_argument('--eps', type=float, default=1e-4,
                       help='Convergence tolerance')
    parser.add_argument('--adet_comm_size', type=int, default=2,
                       help='Alpha determinant communicator size')
    parser.add_argument('--bdet_comm_size', type=int, default=2,
                       help='Beta determinant communicator size')
    parser.add_argument('--task_comm_size', type=int, default=2,
                       help='Task communicator size')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Import sbd (no mpi4py import needed!)
    import sbd
    
    # Initialize SBD with device and communication backend
    # This internally initializes MPI
    sbd.init(device=args.device, comm_backend='mpi')
    
    # Get rank info (no mpi4py needed!)
    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    if rank == 0:
        print("="*70)
        print("SBD Simplified API - H2O Calculation")
        print("="*70)
        sbd.print_info()
        print()
    
    # Configure calculation
    config = sbd.TPB_SBD()
    config.max_it = args.max_it
    config.eps = args.eps
    config.method = 0  # Davidson
    config.do_rdm = 0  # Density only
    config.bit_length = 20
    config.adet_comm_size = args.adet_comm_size
    config.bdet_comm_size = args.bdet_comm_size
    config.task_comm_size = args.task_comm_size
    
    if rank == 0:
        print("Configuration:")
        print(f"  Device: {sbd.get_device()}")
        print(f"  Communication: {sbd.get_comm_backend()}")
        print(f"  Method: Davidson")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Tolerance: {config.eps}")
        print(f"  MPI configuration: {args.task_comm_size} × {args.adet_comm_size} × {args.bdet_comm_size} = {size} ranks")
        print(f"\nInput files:")
        print(f"  FCIDUMP: {args.fcidump}")
        print(f"  Alpha dets: {args.adetfile}")
        print()
    
    # Run calculation (no comm parameter needed!)
    try:
        if rank == 0:
            print("Running TPB diagonalization...")
            print()
        
        results = sbd.tpb_diag_from_files(
            fcidumpfile=args.fcidump,
            adetfile=args.adetfile,
            sbd_data=config
        )
        
        if rank == 0:
            print("="*70)
            print("Results")
            print("="*70)
            print(f"Device: {sbd.get_device().upper()}")
            print(f"Ground state energy: {results['energy']:.10f} Hartree")
            print(f"Density (first 10): {results['density'][:10]}")
            print(f"Carryover determinants: {len(results['carryover_adet'])}")
            print("="*70)
            print("\n✓ Calculation completed successfully!")
            print()
        
        return_code = 0
    
    except FileNotFoundError as e:
        if rank == 0:
            print(f"\n✗ Error: {e}")
            print("\nPlease check file paths:")
            print(f"  FCIDUMP: {args.fcidump}")
            print(f"  Alpha dets: {args.adetfile}")
        return_code = 1
    
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Error during calculation: {e}")
            import traceback
            traceback.print_exc()
        return_code = 1
    
    finally:
        # Clean up SBD (does not finalize MPI - that's handled by mpirun)
        sbd.finalize()
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
"""
H2O calculation example with easy CPU/GPU switching

This example demonstrates how to easily switch between CPU and GPU execution
without changing the core calculation code.

Usage:
    # Auto-detect and use GPU if available
    mpirun -np 4 python h2o_cpu_gpu.py
    
    # Force CPU execution
    mpirun -np 4 python h2o_cpu_gpu.py --device cpu
    
    # Force GPU execution with specific memory limit
    mpirun -np 4 python h2o_cpu_gpu.py --device gpu --gpu-memory 16
"""

import os
import argparse
import sys

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='H2O calculation with CPU/GPU support'
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='Device to use (auto=detect, cpu=force CPU, gpu=force GPU)'
    )
    parser.add_argument(
        '--gpu-memory',
        type=int,
        default=-1,
        help='Maximum GPU memory in GB (-1=auto)'
    )
    parser.add_argument(
        '--fcidump',
        default='../../data/h2o/fcidump.txt',
        help='Path to FCIDUMP file'
    )
    parser.add_argument(
        '--adetfile',
        default='../../data/h2o/h2o-1em3-alpha.txt',
        help='Path to determinants file'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=100,
        help='Maximum iterations'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-3,
        help='Convergence tolerance'
    )
    parser.add_argument(
        '--rdm',
        type=int,
        choices=[0, 1],
        default=0,
        help='Calculate RDM (0=no, 1=yes)'
    )
    parser.add_argument(
        '--adet_comm_size',
        type=int,
        default=1
    )
    parser.add_argument(
        '--bdet_comm_size',
        type=int,
        default=1
    )
    parser.add_argument(
        '--task_comm_size',
        type=int,
        default=1
    )
    
    return parser.parse_args()


# Parse arguments BEFORE importing sbd
args = parse_args()
# Set backend BEFORE importing sbd
if args.device != 'auto':
    os.environ['SBD_BACKEND'] = args.device
os.environ['SBD_VERBOSE'] = '1'  # Show backend selection

# NOW import sbd and mpi4py
import sbd
from mpi4py import MPI

def main():
    # Get MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print("SBD Python Bindings - H2O Calculation")
        print(f"Running with {size} MPI processes")
        print("="*70)
        sbd.print_backend_info()
        print()

    # Configure calculation
    config = sbd.TPB_SBD()
    config.max_it = args.max_iter
    config.eps = args.tolerance
    config.method = 0  # Davidson
    config.do_rdm = args.rdm  # Density only
    config.adet_comm_size = args.adet_comm_size
    config.bdet_comm_size = args.bdet_comm_size
    config.task_comm_size = args.task_comm_size
    
    # GPU-specific settings (only used if GPU backend is active)
    if sbd.get_backend() == 'gpu':
        try:
            # These attributes only exist if compiled with -DSBD_THRUST
            config.use_precalculated_dets = True
            config.max_memory_gb_for_determinants = args.gpu_memory
            if rank == 0:
                print(f"GPU Settings:")
                print(f"  Use precalculated dets: True")
                print(f"  Max GPU memory: {args.gpu_memory} GB")
                print()
        except AttributeError:
            if rank == 0:
                print("Note: GPU-specific parameters not available")
                print("(Backend may not be compiled with THRUST support)")
                print()
    
    if rank == 0:
        print("Calculation Parameters:")
        print(f"  Backend: {sbd.get_backend().upper()}")
        print(f"  Method: Davidson")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Tolerance: {config.eps}")
        print(f"  FCIDUMP file: {args.fcidump}")
        print(f"  Alpha dets file: {args.adetfile}")
        print()
    
    # Run calculation
    try:
        if rank == 0:
            print("Running TPB diagonalization...")
            print()
        
        results = sbd.tpb_diag_from_files(
            comm=comm,
            sbd_data=config,
            fcidumpfile=args.fcidump,
            adetfile=args.adetfile
        )
        
        if rank == 0:
            print("="*90)
            print("Results")
            print("="*90)
            print(f"Backend: {sbd.get_backend().upper()}")
            print(f"Ground state energy: {results['energy']:.10f}")
            print(f"Density (first 10): {results['density'][:10]}")
            #print(f"Number of carryover determinants: {len(results['co_adet'])}")
            print(f"Number of carryover determinants: {len(results['carryover_adet'])}")
            print("="*70)
            print()
            
            return 0
    
    except FileNotFoundError as e:
        if rank == 0:
            print(f"\n✗ Error: {e}")
            print("\nThis example requires H2O input files:")
            print(f"  - {args.fcidump}")
            print(f"  - {args.adetfile}")
            print("\nPlease provide these files or use --fcidump and --adetfile options.")
        return 1
    
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Error during calculation: {e}")
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Made with Bob

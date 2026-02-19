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
    """Parse command line arguments for all TPB_SBD parameters"""
    parser = argparse.ArgumentParser(
        description='H2O calculation with CPU/GPU support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Device selection
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Device to use (auto=detect, cpu=force CPU, gpu=force GPU)')
    
    # Input files
    parser.add_argument('--fcidump', default='../../data/h2o/fcidump.txt',
                       help='Path to FCIDUMP file')
    parser.add_argument('--adetfile', default='../../data/h2o/h2o-1em3-alpha.txt',
                       help='Path to alpha determinants file')
    parser.add_argument('--bdetfile', default='',
                       help='Path to beta determinants file (optional, uses adetfile if not specified)')
    parser.add_argument('--loadname', default='',
                       help='Load initial wavefunction from file')
    parser.add_argument('--savename', default='',
                       help='Save final wavefunction to file')
    
    # MPI communicator sizes
    parser.add_argument('--task_comm_size', type=int, default=1,
                       help='Task communicator size')
    parser.add_argument('--adet_comm_size', type=int, default=1,
                       help='Alpha determinant communicator size')
    parser.add_argument('--bdet_comm_size', type=int, default=1,
                       help='Beta determinant communicator size')
    parser.add_argument('--h_comm_size', type=int, default=1,
                       help='Helper communicator size')
    
    # Diagonalization method and convergence
    parser.add_argument('--method', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Diagonalization method: 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham')
    parser.add_argument('--max_it', '--max-iter', type=int, default=100, dest='max_it',
                       help='Maximum number of iterations')
    parser.add_argument('--max_nb', type=int, default=10,
                       help='Maximum number of basis vectors')
    parser.add_argument('--eps', '--tolerance', type=float, default=1e-3, dest='eps',
                       help='Convergence tolerance')
    parser.add_argument('--max_time', type=float, default=1e10,
                       help='Maximum time in seconds')
    
    # Initialization and options
    parser.add_argument('--init', type=int, default=0,
                       help='Initialization method')
    parser.add_argument('--do_shuffle', type=int, default=0, choices=[0, 1],
                       help='Shuffle determinants (0=no, 1=yes)')
    parser.add_argument('--do_rdm', '--rdm', type=int, default=0, choices=[0, 1], dest='do_rdm',
                       help='Calculate RDM (0=density only, 1=full RDM)')
    
    # Carryover determinant selection
    parser.add_argument('--carryover_type', type=int, default=0,
                       help='Carryover determinant selection type')
    parser.add_argument('--ratio', type=float, default=0.0,
                       help='Carryover ratio')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Carryover threshold')
    
    # Determinant representation
    parser.add_argument('--bit_length', type=int, default=20,
                       help='Bit length for determinant representation')
    
    # Output options
    parser.add_argument('--dump_matrix_form_wf', default='',
                       help='Filename to dump wavefunction in matrix form')
    
    # GPU-specific options (only used with GPU backend)
    parser.add_argument('--use_precalculated_dets', type=int, default=1, choices=[0, 1],
                       help='Use precalculated determinants (GPU only)')
    parser.add_argument('--max_memory_gb_for_determinants', '--gpu-memory', type=int, default=-1,
                       dest='max_memory_gb_for_determinants',
                       help='Maximum GPU memory in GB for determinants (-1=auto, GPU only)')
    
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

    # Configure calculation with all parameters from command line
    config = sbd.TPB_SBD()
    
    # MPI communicator sizes
    config.task_comm_size = args.task_comm_size
    config.adet_comm_size = args.adet_comm_size
    config.bdet_comm_size = args.bdet_comm_size
    config.h_comm_size = args.h_comm_size
    
    # Diagonalization method and convergence
    config.method = args.method
    config.max_it = args.max_it
    config.max_nb = args.max_nb
    config.eps = args.eps
    config.max_time = args.max_time
    
    # Initialization and options
    config.init = args.init
    config.do_shuffle = args.do_shuffle
    config.do_rdm = args.do_rdm
    
    # Carryover determinant selection
    config.carryover_type = args.carryover_type
    config.ratio = args.ratio
    config.threshold = args.threshold
    
    # Determinant representation
    config.bit_length = args.bit_length
    
    # Output options
    config.dump_matrix_form_wf = args.dump_matrix_form_wf
    
    # GPU-specific settings (only used if GPU backend is active)
    if sbd.get_backend() == 'gpu':
        try:
            # These attributes only exist if compiled with -DSBD_THRUST
            config.use_precalculated_dets = bool(args.use_precalculated_dets)
            config.max_memory_gb_for_determinants = args.max_memory_gb_for_determinants
            if rank == 0:
                print(f"GPU Settings:")
                print(f"  Use precalculated dets: {config.use_precalculated_dets}")
                print(f"  Max GPU memory: {config.max_memory_gb_for_determinants} GB")
                print()
        except AttributeError:
            if rank == 0:
                print("Note: GPU-specific parameters not available")
                print("(Backend may not be compiled with THRUST support)")
                print()
    
    if rank == 0:
        method_names = {0: 'Davidson', 1: 'Davidson+Ham', 2: 'Lanczos', 3: 'Lanczos+Ham'}
        print("Calculation Parameters:")
        print(f"  Backend: {sbd.get_backend().upper()}")
        print(f"  Method: {method_names.get(config.method, 'Unknown')}")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Max basis vectors: {config.max_nb}")
        print(f"  Tolerance: {config.eps}")
        print(f"  Max time: {config.max_time} s")
        print(f"  Do RDM: {config.do_rdm}")
        print(f"  Bit length: {config.bit_length}")
        print(f"\nMPI Configuration:")
        print(f"  Task comm size: {config.task_comm_size}")
        print(f"  Adet comm size: {config.adet_comm_size}")
        print(f"  Bdet comm size: {config.bdet_comm_size}")
        print(f"  H comm size: {config.h_comm_size}")
        print(f"  Total ranks needed: {config.task_comm_size * config.adet_comm_size * config.bdet_comm_size}")
        print(f"\nInput Files:")
        print(f"  FCIDUMP: {args.fcidump}")
        print(f"  Alpha dets: {args.adetfile}")
        if args.bdetfile:
            print(f"  Beta dets: {args.bdetfile}")
        if args.loadname:
            print(f"  Load wavefunction: {args.loadname}")
        if args.savename:
            print(f"  Save wavefunction: {args.savename}")
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
            adetfile=args.adetfile,
            loadname=args.loadname,
            savename=args.savename
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

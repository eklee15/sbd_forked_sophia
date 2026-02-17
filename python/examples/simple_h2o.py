"""
Simple example of using SBD Python bindings with H2O data

Usage:
    mpirun -np 4 python simple_h2o.py
"""

from mpi4py import MPI
import sbd
import os

def main():
    # Get MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*60)
        print("SBD Python Bindings - Simple H2O Example")
        print(f"Running with {size} MPI processes")
        print("="*60)
    
    # Configure the calculation
    config = sbd.TPB_SBD()
    config.max_it = 100          # Maximum iterations
    config.eps = 1e-6            # Convergence tolerance
    config.do_rdm = 0            # Don't calculate full RDM (faster)
    config.method = 0            # Davidson method
    config.bit_length = 20       # Bit length for determinants
    
    # Paths to data files (adjust as needed)
    fcidump_file = "../../data/h2o/fcidump.txt"
    adet_file = "../../data/h2o/h2o-1em5-alpha.txt"
    
    if rank == 0:
        print("\nConfiguration:")
        print(f"  Method: {config.method} (0=Davidson)")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Tolerance: {config.eps}")
        print(f"  Calculate RDM: {config.do_rdm}")
        print(f"\nInput files:")
        print(f"  FCIDUMP: {fcidump_file}")
        print(f"  Determinants: {adet_file}")
    
    # Run diagonalization using file-based API (most convenient)
    if rank == 0:
        print("\nRunning diagonalization...")
    
    results = sbd.tpb_diag_from_files(
        comm=comm,
        sbd_data=config,
        fcidumpfile=fcidump_file,
        adetfile=adet_file,
        loadname="",           # No initial wavefunction to load
        savename="h2o_wf.dat"  # Save final wavefunction
    )
    
    # Print results (only on rank 0)
    if rank == 0:
        print("\n" + "="*60)
        print("Results:")
        print("="*60)
        print(f"Ground State Energy: {results['energy']:.10f} Hartree")
        print(f"\nOrbital Densities:")
        density = results['density']
        for i in range(len(density)//2):
            total = density[2*i] + density[2*i+1]
            print(f"  Orbital {i}: {total:.6f} (α={density[2*i]:.6f}, β={density[2*i+1]:.6f})")
        
        print(f"\nCarryover Determinants:")
        print(f"  Alpha: {len(results['carryover_adet'])} determinants")
        print(f"  Beta:  {len(results['carryover_bdet'])} determinants")
        
        if len(results['one_p_rdm']) > 0:
            print(f"\nReduced Density Matrices:")
            print(f"  1-RDM calculated: Yes")
            print(f"  2-RDM calculated: Yes")
        else:
            print(f"\nReduced Density Matrices: Not calculated (do_rdm=0)")
        
        print("\n" + "="*60)
        print("Calculation completed successfully!")
        print("="*60)


if __name__ == '__main__':
    main()

# Made with Bob

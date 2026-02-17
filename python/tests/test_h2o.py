"""
Integration test for SBD Python bindings using H2O data
Run with: mpirun -np 4 python test_h2o.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from mpi4py import MPI
    import sbd
    SBD_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    SBD_AVAILABLE = False
    sys.exit(1)


def test_h2o_diagonalization_from_files():
    """Test full diagonalization workflow with H2O data using file-based API"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running H2O test with {size} MPI processes")
    
    # Paths to test data (relative to this file)
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/h2o')
    fcidump_file = os.path.join(data_dir, 'fcidump.txt')
    adet_file = os.path.join(data_dir, 'h2o-1em5-alpha.txt')
    
    # Check if files exist
    if not os.path.exists(fcidump_file):
        if rank == 0:
            print(f"FCIDUMP file not found: {fcidump_file}")
        return False
    
    if not os.path.exists(adet_file):
        if rank == 0:
            print(f"Determinant file not found: {adet_file}")
        return False
    
    # Configure
    config = sbd.TPB_SBD()
    config.max_it = 50
    config.eps = 1e-5
    config.do_rdm = 0  # Don't calculate full RDM for speed
    config.bit_length = 20
    
    if rank == 0:
        print(f"Configuration:")
        print(f"  max_it: {config.max_it}")
        print(f"  eps: {config.eps}")
        print(f"  method: {config.method}")
        print(f"  do_rdm: {config.do_rdm}")
    
    try:
        # Run diagonalization using file-based API
        results = sbd.tpb_diag_from_files(
            comm=comm,
            sbd_data=config,
            fcidumpfile=fcidump_file,
            adetfile=adet_file,
            loadname="",
            savename=""
        )
        
        if rank == 0:
            print(f"\nResults:")
            print(f"  H2O Ground State Energy: {results['energy']:.10f}")
            print(f"  Density vector length: {len(results['density'])}")
            print(f"  Carryover alpha dets: {len(results['carryover_adet'])}")
            print(f"  Carryover beta dets: {len(results['carryover_bdet'])}")
            
            # Basic validation
            assert results['energy'] < 0, "Energy should be negative"
            assert len(results['density']) > 0, "Density should not be empty"
            
            print("\n✓ H2O test (file-based API) PASSED")
        
        return True
        
    except Exception as e:
        if rank == 0:
            print(f"\n✗ H2O test FAILED with error: {e}")
            import traceback
            traceback.print_exc()
        return False


def test_h2o_diagonalization_with_data():
    """Test full diagonalization workflow with H2O data using data structure API"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Testing data structure API")
        print("="*60)
    
    # Paths to test data
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/h2o')
    fcidump_file = os.path.join(data_dir, 'fcidump.txt')
    adet_file = os.path.join(data_dir, 'h2o-1em5-alpha.txt')
    
    # Check if files exist
    if not os.path.exists(fcidump_file) or not os.path.exists(adet_file):
        if rank == 0:
            print("Test data not available, skipping")
        return False
    
    try:
        # Load data on rank 0
        if rank == 0:
            fcidump = sbd.LoadFCIDump(fcidump_file)
            alpha_dets = sbd.LoadAlphaDets(adet_file, bit_length=20, total_bit_length=26)
            print(f"Loaded {len(alpha_dets)} determinants")
        else:
            fcidump = None
            alpha_dets = None
        
        # Broadcast data (in real use, you might want to do this in C++)
        # For now, we'll just use the file-based API which handles this internally
        
        # Configure
        config = sbd.TPB_SBD()
        config.max_it = 50
        config.eps = 1e-5
        config.do_rdm = 0
        config.bit_length = 20
        
        if rank == 0 and fcidump is not None and alpha_dets is not None:
            # Run diagonalization using data structure API
            results = sbd.tpb_diag(
                comm=comm,
                sbd_data=config,
                fcidump=fcidump,
                adet=alpha_dets,
                bdet=alpha_dets,  # Use same for closed shell
                loadname="",
                savename=""
            )
            
            print(f"\nResults (data structure API):")
            print(f"  Energy: {results['energy']:.10f}")
            print(f"  Density vector length: {len(results['density'])}")
            
            print("\n✓ H2O test (data structure API) PASSED")
        
        return True
        
    except Exception as e:
        if rank == 0:
            print(f"\n✗ H2O test (data structure API) FAILED with error: {e}")
            import traceback
            traceback.print_exc()
        return False


if __name__ == '__main__':
    if not SBD_AVAILABLE:
        print("SBD module not available. Please build the module first.")
        sys.exit(1)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60)
        print("SBD Python Bindings - H2O Integration Test")
        print("="*60)
    
    # Test file-based API
    success1 = test_h2o_diagonalization_from_files()
    
    # Test data structure API
    success2 = test_h2o_diagonalization_with_data()
    
    if rank == 0:
        print("\n" + "="*60)
        if success1 and success2:
            print("All tests PASSED ✓")
        else:
            print("Some tests FAILED ✗")
        print("="*60)
    
    sys.exit(0 if (success1 and success2) else 1)

# Made with Bob

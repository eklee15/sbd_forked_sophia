"""
Basic tests for SBD Python bindings
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import sbd
    SBD_AVAILABLE = True
except ImportError:
    SBD_AVAILABLE = False


@pytest.mark.skipif(not SBD_AVAILABLE, reason="SBD module not built")
class TestBasicBindings:
    """Test basic binding functionality"""
    
    def test_import(self):
        """Test that sbd module can be imported"""
        import sbd
        assert sbd is not None
    
    def test_version(self):
        """Test version attribute"""
        import sbd
        assert hasattr(sbd, '__version__')
        assert isinstance(sbd.__version__, str)
    
    def test_tpb_sbd_creation(self):
        """Test TPB_SBD configuration object creation"""
        import sbd
        config = sbd.TPB_SBD()
        assert config is not None
        
        # Test default values
        assert config.method == 0
        assert config.max_it == 1
        assert config.max_nb == 10
        assert config.eps == 1.0e-4
        assert config.bit_length == 20
    
    def test_tpb_sbd_setters(self):
        """Test TPB_SBD configuration setters"""
        import sbd
        config = sbd.TPB_SBD()
        
        # Set values
        config.max_it = 100
        config.eps = 1e-6
        config.do_rdm = 1
        config.method = 2
        
        # Verify values
        assert config.max_it == 100
        assert config.eps == 1e-6
        assert config.do_rdm == 1
        assert config.method == 2
    
    def test_fcidump_creation(self):
        """Test FCIDump object creation"""
        import sbd
        fcidump = sbd.FCIDump()
        assert fcidump is not None
        assert hasattr(fcidump, 'header')
        assert hasattr(fcidump, 'one_electron_integrals')
        assert hasattr(fcidump, 'two_electron_integrals')
    
    def test_functions_exist(self):
        """Test that main functions exist"""
        import sbd
        assert hasattr(sbd, 'LoadFCIDump')
        assert hasattr(sbd, 'LoadAlphaDets')
        assert hasattr(sbd, 'makestring')
        assert hasattr(sbd, 'tpb_diag')
        assert hasattr(sbd, 'tpb_diag_from_files')
    
    def test_load_fcidump_with_file(self):
        """Test LoadFCIDump with actual file if available"""
        import sbd
        
        # Look for test data
        test_data_path = os.path.join(os.path.dirname(__file__), '../../data/h2o/fcidump.txt')
        if os.path.exists(test_data_path):
            fcidump = sbd.LoadFCIDump(test_data_path)
            assert fcidump is not None
            assert len(fcidump.header) > 0
            assert 'NORB' in fcidump.header
            assert 'NELEC' in fcidump.header
        else:
            pytest.skip("Test data not available")
    
    def test_load_alpha_dets_with_file(self):
        """Test LoadAlphaDets with actual file if available"""
        import sbd
        
        # Look for test data
        test_data_path = os.path.join(os.path.dirname(__file__), '../../data/h2o/h2o-1em5-alpha.txt')
        if os.path.exists(test_data_path):
            dets = sbd.LoadAlphaDets(test_data_path, bit_length=20, total_bit_length=26)
            assert dets is not None
            assert isinstance(dets, list)
            assert len(dets) > 0
            assert isinstance(dets[0], list)
        else:
            pytest.skip("Test data not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# Made with Bob

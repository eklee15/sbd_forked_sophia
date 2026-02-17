/**
 * @file python/bindings.cpp
 * @brief Python bindings for SBD TPB diagonalization using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>

#include "sbd/sbd.h"

namespace py = pybind11;

/**
 * Helper function to convert mpi4py communicator to MPI_Comm
 */
MPI_Comm get_mpi_comm(py::object py_comm) {
    PyObject* py_comm_ptr = py_comm.ptr();
    MPI_Comm* comm_ptr = PyMPIComm_Get(py_comm_ptr);
    if (!comm_ptr) {
        throw std::runtime_error("Invalid MPI communicator");
    }
    return *comm_ptr;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for SBD (Selected Basis Diagonalization) library";

    // Initialize mpi4py
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Failed to import mpi4py");
    }

    // ========================================================================
    // Bind FCIDump structure
    // ========================================================================
    py::class_<sbd::FCIDump>(m, "FCIDump", "FCIDUMP data structure")
        .def(py::init<>())
        .def_readwrite("header", &sbd::FCIDump::header,
                      "Header information as dictionary (map<string, string>)")
        .def_readwrite("integrals", &sbd::FCIDump::integrals,
                      "Integral data as list of tuples (value, i, j, k, l)");

    // ========================================================================
    // Bind TPB SBD configuration structure
    // ========================================================================
    py::class_<sbd::tpb::SBD>(m, "TPB_SBD", "Configuration for TPB diagonalization")
        .def(py::init<>())
        .def_readwrite("task_comm_size", &sbd::tpb::SBD::task_comm_size,
                      "Task communicator size")
        .def_readwrite("adet_comm_size", &sbd::tpb::SBD::adet_comm_size,
                      "Alpha determinant communicator size")
        .def_readwrite("bdet_comm_size", &sbd::tpb::SBD::bdet_comm_size,
                      "Beta determinant communicator size")
        .def_readwrite("h_comm_size", &sbd::tpb::SBD::h_comm_size,
                      "Helper communicator size")
        .def_readwrite("method", &sbd::tpb::SBD::method,
                      "Diagonalization method (0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham)")
        .def_readwrite("max_it", &sbd::tpb::SBD::max_it,
                      "Maximum number of iterations")
        .def_readwrite("max_nb", &sbd::tpb::SBD::max_nb,
                      "Maximum number of basis vectors")
        .def_readwrite("eps", &sbd::tpb::SBD::eps,
                      "Convergence tolerance")
        .def_readwrite("max_time", &sbd::tpb::SBD::max_time,
                      "Maximum time in seconds")
        .def_readwrite("init", &sbd::tpb::SBD::init,
                      "Initialization method")
        .def_readwrite("do_shuffle", &sbd::tpb::SBD::do_shuffle,
                      "Shuffle determinants flag")
        .def_readwrite("do_rdm", &sbd::tpb::SBD::do_rdm,
                      "Calculate RDM flag (0=density only, 1=full RDM)")
        .def_readwrite("carryover_type", &sbd::tpb::SBD::carryover_type,
                      "Carryover determinant selection type")
        .def_readwrite("ratio", &sbd::tpb::SBD::ratio,
                      "Carryover ratio")
        .def_readwrite("threshold", &sbd::tpb::SBD::threshold,
                      "Carryover threshold")
        .def_readwrite("bit_length", &sbd::tpb::SBD::bit_length,
                      "Bit length for determinant representation")
        .def_readwrite("dump_matrix_form_wf", &sbd::tpb::SBD::dump_matrix_form_wf,
                      "Filename to dump wavefunction in matrix form")
#ifdef SBD_THRUST
        .def_readwrite("use_precalculated_dets", &sbd::tpb::SBD::use_precalculated_dets,
                      "Use precalculated determinants (THRUST)")
        .def_readwrite("max_memory_gb_for_determinants", &sbd::tpb::SBD::max_memory_gb_for_determinants,
                      "Maximum memory in GB for determinants (THRUST)")
#endif
        ;

    // ========================================================================
    // Utility functions
    // ========================================================================
    
    m.def("LoadFCIDump", &sbd::LoadFCIDump,
          "Load FCIDUMP file and return FCIDump object",
          py::arg("filename"));

    m.def("LoadAlphaDets",
          [](const std::string& filename, size_t bit_length, size_t total_bit_length) {
              std::vector<std::vector<size_t>> dets;
              sbd::LoadAlphaDets(filename, dets, bit_length, total_bit_length);
              return dets;
          },
          "Load alpha determinants from file",
          py::arg("filename"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    m.def("makestring", &sbd::makestring,
          "Convert bitstring to string representation",
          py::arg("config"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    // ========================================================================
    // Main TPB diagonalization function (data structure version)
    // ========================================================================
    
    m.def("tpb_diag",
        [](py::object py_comm,
           const sbd::tpb::SBD& sbd_data,
           const sbd::FCIDump& fcidump,
           const std::vector<std::vector<size_t>>& adet,
           const std::vector<std::vector<size_t>>& bdet,
           const std::string& loadname,
           const std::string& savename) {
            
            // Convert MPI communicator
            MPI_Comm comm = get_mpi_comm(py_comm);
            
            // Output variables
            double energy;
            std::vector<double> density;
            std::vector<std::vector<size_t>> co_adet;
            std::vector<std::vector<size_t>> co_bdet;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;
            
            // Release GIL for long computation
            py::gil_scoped_release release;
            
            // Call C++ function
            sbd::tpb::diag(comm, sbd_data, fcidump, adet, bdet,
                          loadname, savename, energy, density,
                          co_adet, co_bdet, one_p_rdm, two_p_rdm);
            
            // Reacquire GIL for Python object creation
            py::gil_scoped_acquire acquire;
            
            // Return results as dictionary
            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_adet"] = co_adet;
            results["carryover_bdet"] = co_bdet;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;
            
            return results;
        },
        "Perform TPB diagonalization with pre-loaded data structures",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidump"),
        py::arg("adet"),
        py::arg("bdet"),
        py::arg("loadname") = "",
        py::arg("savename") = "");

    // ========================================================================
    // Main TPB diagonalization function (file-based version)
    // ========================================================================
    
    m.def("tpb_diag_from_files",
        [](py::object py_comm,
           const sbd::tpb::SBD& sbd_data,
           const std::string& fcidumpfile,
           const std::string& adetfile,
           const std::string& loadname,
           const std::string& savename) {
            
            // Convert MPI communicator
            MPI_Comm comm = get_mpi_comm(py_comm);
            
            // Output variables
            double energy;
            std::vector<double> density;
            std::vector<std::vector<size_t>> co_adet;
            std::vector<std::vector<size_t>> co_bdet;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;
            
            // Release GIL for long computation
            py::gil_scoped_release release;
            
            // Call file-based C++ function
            sbd::tpb::diag(comm, sbd_data, fcidumpfile, adetfile,
                          loadname, savename, energy, density,
                          co_adet, co_bdet, one_p_rdm, two_p_rdm);
            
            // Reacquire GIL for Python object creation
            py::gil_scoped_acquire acquire;
            
            // Return results as dictionary
            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_adet"] = co_adet;
            results["carryover_bdet"] = co_bdet;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;
            
            return results;
        },
        "Perform TPB diagonalization from files (convenience function)",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidumpfile"),
        py::arg("adetfile"),
        py::arg("loadname") = "",
        py::arg("savename") = "");

    // ========================================================================
    // Version information
    // ========================================================================
    
    m.attr("__version__") = "1.2.0";
}

// Made with Bob

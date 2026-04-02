/**
 * Convert Qiskit Sampler JSON output to SBD determinant files using qiskit-addon-sqd-hpc.
 *
 * This program uses the qiskit-addon-sqd-hpc library to properly convert Sampler output
 * into determinant files compatible with the SBD solver.
 *
 * Dependencies:
 *   - qiskit-addon-sqd-hpc library
 *   - Boost C++ library (tested with Boost 1.85.0)
 *
 * Compile:
 *   export SQD_HPC_PATH=${SQD_HPC_PATH:-../../qiskit-addon-sqd-hpc}
 *   export BOOST_PATH=${BOOST_PATH:-/path/to/boost}
 *   g++ -std=c++17 -O3 -I${SQD_HPC_PATH}/include -I${BOOST_PATH} \
 *       -o convert_sampler_to_sbd_cpp convert_sampler_to_sbd.cpp
 *
 * Usage:
 *   ./convert_sampler_to_sbd_cpp count_dict.json 29 5 5 output_prefix
 *
 *   Arguments:
 *     count_dict.json - Input JSON file from Qiskit Sampler
 *     29              - Number of orbitals
 *     5               - Number of alpha electrons
 *     5               - Number of beta electrons
 *     output_prefix   - Output file prefix (generates prefix_alpha.txt and prefix_beta.txt)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <optional>
#include <unordered_set>
#include <random>
#include <boost/dynamic_bitset.hpp>

// Use qiskit-addon-sqd-hpc library
#include "qiskit/addon/sqd/fermion.hpp"
#include "qiskit/addon/sqd/configuration_recovery.hpp"
#include "qiskit/addon/sqd/bitset_full.hpp"

/**
 * Parse a JSON line containing a bitstring and count.
 * Returns true on success, false on failure.
 */
bool parse_json_line(const std::string& line, std::string& bitstring, int& count) {
    // Find the bitstring (between quotes)
    auto start = line.find('"');
    if (start == std::string::npos) return false;
    start++;
    
    auto end = line.find('"', start);
    if (end == std::string::npos) return false;
    
    bitstring = line.substr(start, end - start);
    
    // Find the count (after the colon)
    auto colon = line.find(':', end);
    if (colon == std::string::npos) return false;
    
    try {
        count = std::stoi(line.substr(colon + 1));
    } catch (...) {
        return false;
    }
    
    return true;
}

/**
 * Convert a binary string to a boost::dynamic_bitset.
 */
boost::dynamic_bitset<> string_to_bitset(const std::string& binary_str) {
    boost::dynamic_bitset<> bs(binary_str.size());
    for (size_t i = 0; i < binary_str.size(); ++i) {
        if (binary_str[i] == '1') {
            bs.set(binary_str.size() - 1 - i);  // Reverse order for bitset
        }
    }
    return bs;
}

/**
 * Convert a boost::dynamic_bitset to a binary string.
 */
std::string bitset_to_string(const boost::dynamic_bitset<>& bs) {
    std::string result;
    result.reserve(bs.size());
    for (size_t i = bs.size(); i > 0; --i) {
        result += bs.test(i - 1) ? '1' : '0';
    }
    return result;
}

/**
 * Process the JSON file and generate determinant files using qiskit-addon-sqd-hpc.
 */
int process_json_file(const std::string& json_file, unsigned int norb,
                     unsigned int n_alpha, unsigned int n_beta,
                     const std::string& output_prefix) {
    std::ifstream fp(json_file);
    if (!fp.is_open()) {
        std::cerr << "Error: Cannot open file " << json_file << std::endl;
        return 1;
    }
    
    std::cout << "Processing " << json_file << "..." << std::endl;
    std::cout << "Expected bitstring length: " << (2 * norb) << " bits (2 * "
              << norb << " orbitals)" << std::endl;
    std::cout << "Target electrons: n_alpha=" << n_alpha << ", n_beta=" << n_beta << std::endl;
    
    // Read entire JSON file (may be single line)
    std::stringstream buffer;
    buffer << fp.rdbuf();
    std::string json_content = buffer.str();
    fp.close();
    
    std::vector<boost::dynamic_bitset<>> bitstrings_raw;
    std::vector<double> probabilities;
    std::string bitstring;
    int count;
    unsigned int expected_length = 2 * norb;
    int total_counts = 0;
    
    // Parse JSON content - find all bitstring:count pairs
    size_t pos = 0;
    while ((pos = json_content.find('"', pos)) != std::string::npos) {
        pos++;  // Skip opening quote
        size_t end_quote = json_content.find('"', pos);
        if (end_quote == std::string::npos) break;
        
        bitstring = json_content.substr(pos, end_quote - pos);
        pos = end_quote + 1;
        
        // Find the colon and count
        size_t colon = json_content.find(':', pos);
        if (colon == std::string::npos) break;
        
        // Find the number after colon
        size_t num_start = colon + 1;
        while (num_start < json_content.length() &&
               (json_content[num_start] == ' ' || json_content[num_start] == '\t')) {
            num_start++;
        }
        
        size_t num_end = num_start;
        while (num_end < json_content.length() &&
               (std::isdigit(json_content[num_end]) || json_content[num_end] == '-')) {
            num_end++;
        }
        
        if (num_end > num_start) {
            try {
                count = std::stoi(json_content.substr(num_start, num_end - num_start));
                
                if (bitstring.length() == expected_length) {
                    bitstrings_raw.push_back(string_to_bitset(bitstring));
                    total_counts += count;
                }
            } catch (...) {
                // Skip invalid entries
            }
        }
        
        pos = num_end;
    }
    
    // Calculate probabilities (second pass through same data)
    pos = 0;
    while ((pos = json_content.find('"', pos)) != std::string::npos) {
        pos++;
        size_t end_quote = json_content.find('"', pos);
        if (end_quote == std::string::npos) break;
        
        bitstring = json_content.substr(pos, end_quote - pos);
        pos = end_quote + 1;
        
        size_t colon = json_content.find(':', pos);
        if (colon == std::string::npos) break;
        
        size_t num_start = colon + 1;
        while (num_start < json_content.length() &&
               (json_content[num_start] == ' ' || json_content[num_start] == '\t')) {
            num_start++;
        }
        
        size_t num_end = num_start;
        while (num_end < json_content.length() &&
               (std::isdigit(json_content[num_end]) || json_content[num_end] == '-')) {
            num_end++;
        }
        
        if (num_end > num_start && bitstring.length() == expected_length) {
            try {
                count = std::stoi(json_content.substr(num_start, num_end - num_start));
                probabilities.push_back(static_cast<double>(count) / total_counts);
            } catch (...) {
                // Skip invalid entries
            }
        }
        
        pos = num_end;
    }
    
    std::cout << "Loaded " << bitstrings_raw.size() << " unique bitstrings" << std::endl;
    
    if (bitstrings_raw.empty()) {
        std::cerr << "Error: No valid bitstrings found" << std::endl;
        return 1;
    }
    
    // Use qiskit-addon-sqd-hpc configuration recovery to fix noisy bitstrings
    std::cout << "\nApplying configuration recovery (fixing noisy bitstrings)..." << std::endl;
    
    // Create initial occupancies (uniform distribution)
    std::array<std::vector<double>, 2> initial_occupancies;
    initial_occupancies[0] = std::vector<double>(norb, 0.5);  // Alpha
    initial_occupancies[1] = std::vector<double>(norb, 0.5);  // Beta
    
    // Create num_elec array
    std::array<std::uint64_t, 2> num_elec = {n_alpha, n_beta};
    
    // Create RNG
    std::mt19937_64 rng(42);  // Fixed seed for reproducibility
    
    auto [bitstrings, probs] = Qiskit::addon::sqd::recover_configurations(
        bitstrings_raw,
        probabilities,
        initial_occupancies,
        num_elec,
        rng
    );
    
    std::cout << "  Recovered " << bitstrings.size() << " valid bitstrings" << std::endl;
    
    if (bitstrings.empty()) {
        std::cerr << "Error: Configuration recovery produced no valid bitstrings" << std::endl;
        return 1;
    }
    
    // Split recovered bitstrings into alpha and beta determinants
    std::vector<boost::dynamic_bitset<>> alpha_dets, beta_dets;
    for (const auto& bs : bitstrings) {
        boost::dynamic_bitset<> alpha_bits(norb);
        boost::dynamic_bitset<> beta_bits(norb);
        
        for (unsigned int j = 0; j < norb; ++j) {
            beta_bits[j] = bs[j];
            alpha_bits[j] = bs[norb + j];
        }
        
        alpha_dets.push_back(alpha_bits);
        beta_dets.push_back(beta_bits);
    }
    
    std::cout << "\nConverted to determinants:" << std::endl;
    std::cout << "  Alpha determinants: " << alpha_dets.size() << std::endl;
    std::cout << "  Beta determinants: " << beta_dets.size() << std::endl;
    
    // Get unique determinants using qiskit-addon-sqd-hpc
    // Note: We pass the determinants as separate vectors, not using symmetrize_spin
    // since we want separate alpha and beta lists
    std::vector<boost::dynamic_bitset<>> unique_alpha_vec, unique_beta_vec;
    std::unordered_set<std::string> seen_alpha, seen_beta;
    
    for (const auto& det : alpha_dets) {
        std::string det_str = bitset_to_string(det);
        if (seen_alpha.insert(det_str).second) {
            unique_alpha_vec.push_back(det);
        }
    }
    
    for (const auto& det : beta_dets) {
        std::string det_str = bitset_to_string(det);
        if (seen_beta.insert(det_str).second) {
            unique_beta_vec.push_back(det);
        }
    }
    
    auto unique_alpha = unique_alpha_vec;
    auto unique_beta = unique_beta_vec;
    
    // Open output files
    std::string alpha_file = output_prefix + "_alpha.txt";
    std::string beta_file = output_prefix + "_beta.txt";
    
    std::ofstream fp_alpha(alpha_file);
    std::ofstream fp_beta(beta_file);
    
    if (!fp_alpha.is_open() || !fp_beta.is_open()) {
        std::cerr << "Error: Cannot create output files" << std::endl;
        return 1;
    }
    
    // Write unique determinants
    for (const auto& det : unique_alpha) {
        fp_alpha << bitset_to_string(det) << "\n";
    }
    
    for (const auto& det : unique_beta) {
        fp_beta << bitset_to_string(det) << "\n";
    }
    
    fp_alpha.close();
    fp_beta.close();
    
    std::cout << "\nUnique determinants (after filtering):" << std::endl;
    std::cout << "  Unique alpha: " << unique_alpha.size() << std::endl;
    std::cout << "  Unique beta: " << unique_beta.size() << std::endl;
    
    std::cout << "\nConversion complete!" << std::endl;
    std::cout << "Saved unique determinants (using qiskit-addon-sqd-hpc):" << std::endl;
    std::cout << "  Alpha: " << alpha_file << " (" << unique_alpha.size() << " determinants)" << std::endl;
    std::cout << "  Beta:  " << beta_file << " (" << unique_beta.size() << " determinants)" << std::endl;
    std::cout << "\nThese files can be used with SBD solver:" << std::endl;
    std::cout << "  mpirun -np 4 sbd_diag --fcidump fcidump.txt \\" << std::endl;
    std::cout << "    --adetfile " << alpha_file << " \\" << std::endl;
    std::cout << "    --bdetfile " << beta_file << std::endl;
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <json_file> <norb> <n_alpha> <n_beta> [output_prefix]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  json_file      Path to count_dict.json from Qiskit Sampler" << std::endl;
        std::cerr << "  norb           Number of spatial orbitals" << std::endl;
        std::cerr << "  n_alpha        Number of alpha electrons" << std::endl;
        std::cerr << "  n_beta         Number of beta electrons" << std::endl;
        std::cerr << "  output_prefix  Prefix for output files (default: determinants)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " ../count_dict.json 29 5 5 sbd_output_cpp/determinants" << std::endl;
        std::cerr << std::endl;
        std::cerr << "This version uses qiskit-addon-sqd-hpc library functions including" << std::endl;
        std::cerr << "configuration recovery to fix noisy bitstrings." << std::endl;
        return 1;
    }
    
    std::string json_file = argv[1];
    unsigned int norb = std::stoul(argv[2]);
    unsigned int n_alpha = std::stoul(argv[3]);
    unsigned int n_beta = std::stoul(argv[4]);
    std::string output_prefix = (argc > 5) ? argv[5] : "determinants";
    
    if (norb == 0 || norb > 128) {
        std::cerr << "Error: norb must be between 1 and 128" << std::endl;
        return 1;
    }
    
    if (n_alpha > norb || n_beta > norb) {
        std::cerr << "Error: n_alpha and n_beta must be <= norb" << std::endl;
        return 1;
    }
    
    return process_json_file(json_file, norb, n_alpha, n_beta, output_prefix);
}

// Made with Bob

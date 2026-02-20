#!/usr/bin/env python3
"""
Test script to verify from_string() conversion correctness.
Compares old Python method vs new C++ from_string().
"""

import sbd

def old_python_method(bitstring, num_orbitals=10, bit_length=20):
    """Old Python implementation for comparison"""
    # Convert hex to integer
    value = int(bitstring, 16)
    
    # Extract alpha and beta parts
    alpha_mask = (1 << num_orbitals) - 1
    beta_mask = alpha_mask << num_orbitals
    
    alpha = value & alpha_mask
    beta = (value & beta_mask) >> num_orbitals
    
    # Convert to binary strings
    alpha_binary = bin(alpha)[2:].zfill(num_orbitals)
    beta_binary = bin(beta)[2:].zfill(num_orbitals)
    
    # Python implementation of from_string
    def python_from_string(binary_str, bit_length, total_bits):
        bits_per_word = 64
        num_words = (total_bits + bits_per_word - 1) // bits_per_word
        result = [0] * num_words
        
        for i, bit in enumerate(reversed(binary_str)):
            if bit == '1':
                word_idx = i // bits_per_word
                bit_idx = i % bits_per_word
                result[word_idx] |= (1 << bit_idx)
        
        return result
    
    alpha_det = python_from_string(alpha_binary, bit_length, num_orbitals)
    beta_det = python_from_string(beta_binary, bit_length, num_orbitals)
    
    return alpha_det, beta_det, alpha_binary, beta_binary

def new_cpp_method(bitstring, num_orbitals=10, bit_length=20):
    """New C++ implementation"""
    # Convert hex to integer
    value = int(bitstring, 16)
    
    # Extract alpha and beta parts
    alpha_mask = (1 << num_orbitals) - 1
    beta_mask = alpha_mask << num_orbitals
    
    alpha = value & alpha_mask
    beta = (value & beta_mask) >> num_orbitals
    
    # Convert to binary strings
    alpha_binary = bin(alpha)[2:].zfill(num_orbitals)
    beta_binary = bin(beta)[2:].zfill(num_orbitals)
    
    # Use C++ from_string
    alpha_det = sbd.from_string(alpha_binary, bit_length, num_orbitals)
    beta_det = sbd.from_string(beta_binary, bit_length, num_orbitals)
    
    return alpha_det, beta_det, alpha_binary, beta_binary

def main():
    sbd.init()
    
    print("="*70)
    print("Testing from_string() Conversion")
    print("="*70)
    
    # Test cases
    test_cases = [
        '0x1f001f',  # Ground state: orbitals 0-4 (both spins)
        '0x2f001f',  # Single excitation
        '0x3e001f',  # Different excitation
    ]
    
    for bitstring in test_cases:
        print(f"\nTest: {bitstring}")
        print("-" * 70)
        
        # Old method
        old_alpha, old_beta, old_alpha_bin, old_beta_bin = old_python_method(bitstring)
        
        # New method
        new_alpha, new_beta, new_alpha_bin, new_beta_bin = new_cpp_method(bitstring)
        
        # Compare
        print(f"Alpha binary: {old_alpha_bin}")
        print(f"  Old Python: {old_alpha}")
        print(f"  New C++:    {new_alpha}")
        print(f"  Match: {old_alpha == new_alpha}")
        
        print(f"Beta binary:  {old_beta_bin}")
        print(f"  Old Python: {old_beta}")
        print(f"  New C++:    {new_beta}")
        print(f"  Match: {old_beta == new_beta}")
        
        if old_alpha != new_alpha or old_beta != new_beta:
            print("  ⚠️  MISMATCH DETECTED!")
        else:
            print("  ✓ All match")
    
    print("\n" + "="*70)
    sbd.finalize()

if __name__ == '__main__':
    main()

# Made with Bob

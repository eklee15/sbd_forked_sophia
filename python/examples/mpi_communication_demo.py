#!/usr/bin/env python3
"""
Simple MPI Communication Demo

This example demonstrates SBD's MPI communication primitives without
running the full diagonalization. It's a lightweight way to test and
understand the communication API.

Usage:
    mpirun -np 4 python mpi_communication_demo.py
"""

import sys

def main():
    # Import SBD
    import sbd
    
    # Initialize SBD with MPI backend
    sbd.init(device='cpu', comm_backend='mpi')
    
    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    print(f"="*70)
    print(f"MPI Communication Demo - Rank {rank}/{size}")
    print(f"="*70)
    
    try:
        # ====================================================================
        # Demo 1: Broadcast
        # ====================================================================
        if rank == 0:
            print("\n[Demo 1] Broadcast: Rank 0 sends data to all ranks")
        
        # Rank 0 has data
        if rank == 0:
            data = {
                'message': 'Hello from rank 0!',
                'numbers': [1, 2, 3, 4, 5],
                'value': 42
            }
            print(f"  Rank 0: Broadcasting {data}")
        else:
            data = None
        
        # Broadcast to all ranks
        data = sbd.broadcast(data, root=0)
        sbd.barrier()
        
        print(f"  Rank {rank}: Received {data}")
        
        # ====================================================================
        # Demo 2: Gather
        # ====================================================================
        if rank == 0:
            print(f"\n[Demo 2] Gather: All ranks send data to rank 0")
        
        # Each rank has local data
        local_data = {
            'rank': rank,
            'value': rank * 10,
            'message': f'Data from rank {rank}'
        }
        
        print(f"  Rank {rank}: Sending {local_data}")
        
        # Gather to rank 0
        all_data = sbd.gather(local_data, root=0)
        sbd.barrier()
        
        if rank == 0:
            print(f"  Rank 0: Received data from all ranks:")
            for i, d in enumerate(all_data):
                print(f"    From rank {i}: {d}")
        
        # ====================================================================
        # Demo 3: All-Gather
        # ====================================================================
        if rank == 0:
            print(f"\n[Demo 3] All-Gather: All ranks exchange data")
        
        # Each rank has local data
        local_value = rank + 100
        
        print(f"  Rank {rank}: My value is {local_value}")
        
        # All-gather (everyone gets everyone's data)
        all_values = sbd.all_gather(local_value)
        sbd.barrier()
        
        print(f"  Rank {rank}: All values are {all_values}")
        
        # ====================================================================
        # Demo 4: Reduce (Sum)
        # ====================================================================
        if rank == 0:
            print(f"\n[Demo 4] Reduce: Sum values from all ranks")
        
        # Each rank contributes a value
        local_count = rank + 1  # 1, 2, 3, 4
        
        print(f"  Rank {rank}: Contributing {local_count}")
        
        # Sum all values to rank 0
        total = sbd.reduce(local_count, op='sum', root=0)
        sbd.barrier()
        
        if rank == 0:
            expected = size * (size + 1) // 2  # Sum of 1 to size
            print(f"  Rank 0: Total sum = {total} (expected: {expected})")
            assert total == expected, "Sum mismatch!"
        
        # ====================================================================
        # Demo 5: All-Reduce (Average)
        # ====================================================================
        if rank == 0:
            print(f"\n[Demo 5] All-Reduce: Compute average across all ranks")
        
        # Each rank has a value
        local_measurement = (rank + 1) * 10.0  # 10, 20, 30, 40
        
        print(f"  Rank {rank}: My measurement is {local_measurement}")
        
        # Compute average (all ranks get result)
        average = sbd.all_reduce(local_measurement, op='avg')
        sbd.barrier()
        
        expected_avg = (size + 1) * 5.0  # Average of 10, 20, 30, 40
        print(f"  Rank {rank}: Average = {average:.1f} (expected: {expected_avg:.1f})")
        
        # ====================================================================
        # Demo 6: Reduce with Different Operations
        # ====================================================================
        if rank == 0:
            print(f"\n[Demo 6] Reduce: Different operations")
        
        local_value = rank + 1
        
        # Sum
        sum_result = sbd.reduce(local_value, op='sum', root=0)
        # Product
        prod_result = sbd.reduce(local_value, op='prod', root=0)
        # Max
        max_result = sbd.reduce(local_value, op='max', root=0)
        # Min
        min_result = sbd.reduce(local_value, op='min', root=0)
        
        if rank == 0:
            print(f"  Values: {list(range(1, size+1))}")
            print(f"  Sum:     {sum_result}")
            print(f"  Product: {prod_result}")
            print(f"  Max:     {max_result}")
            print(f"  Min:     {min_result}")
        
        # ====================================================================
        # Demo 7: Simulated Quantum-Classical Pattern
        # ====================================================================
        if rank == 0:
            print(f"\n[Demo 7] Quantum-Classical Pattern")
            print(f"  (Simulating quantum sampling workflow)")
        
        # Step 1: Rank 0 "samples quantum circuit"
        if rank == 0:
            quantum_samples = {
                'bitstrings': ['0x1f001f', '0x2f001f', '0x3e001f'],
                'counts': [523, 245, 156],
                'num_shots': 1000
            }
            print(f"  Rank 0: Simulated quantum sampling")
            print(f"    Got {len(quantum_samples['bitstrings'])} configurations")
        else:
            quantum_samples = None
        
        # Step 2: Broadcast to all ranks
        quantum_samples = sbd.broadcast(quantum_samples, root=0)
        print(f"  Rank {rank}: Received quantum samples")
        
        # Step 3: Each rank processes subset
        num_samples = len(quantum_samples['bitstrings'])
        samples_per_rank = num_samples // size
        start = rank * samples_per_rank
        end = start + samples_per_rank if rank < size - 1 else num_samples
        
        local_samples = quantum_samples['bitstrings'][start:end]
        local_processed = len(local_samples)
        
        print(f"  Rank {rank}: Processing {local_processed} samples")
        
        # Step 4: Gather results
        all_processed = sbd.gather(local_processed, root=0)
        
        if rank == 0:
            total_processed = sum(all_processed)
            print(f"  Rank 0: Total samples processed = {total_processed}")
            print(f"    Distribution: {all_processed}")
        
        # ====================================================================
        # Summary
        # ====================================================================
        sbd.barrier()
        
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"✓ All MPI communication demos completed successfully!")
            print(f"{'='*70}")
            print(f"\nAvailable Communication Primitives:")
            print(f"  • sbd.broadcast(data, root=0)")
            print(f"  • sbd.gather(data, root=0)")
            print(f"  • sbd.all_gather(data)")
            print(f"  • sbd.reduce(data, op='sum', root=0)")
            print(f"  • sbd.all_reduce(data, op='sum')")
            print(f"  • sbd.barrier()")
            print(f"\nSupported reduce operations: 'sum', 'prod', 'max', 'min', 'avg'")
            print(f"{'='*70}")
        
        return_code = 0
        
    except Exception as e:
        print(f"\n✗ Error on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        return_code = 1
    
    finally:
        # Clean up
        sbd.finalize()
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())

# Made with Bob

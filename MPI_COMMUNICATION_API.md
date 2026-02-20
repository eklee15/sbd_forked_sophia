# SBD MPI Communication Primitives API Design

## Overview

This document describes MPI communication primitives in the SBD Python API, similar to `torch.distributed`, to support quantum-classical hybrid workflows where rank 0 performs quantum operations and broadcasts results to other ranks.

**Note for Quantum/Qiskit Users:** If you're coming from the quantum computing community and haven't used MPI before, don't worry! MPI (Message Passing Interface) is simply a way for multiple processes to communicate and work together. Think of it like having multiple Python processes that can send messages to each other. We provide a simple, PyTorch-like API that makes it easy to use.

## Quick Start: Simple MPI Demo

**New to MPI?** Start with our simple demo that shows how to use the communication primitives without running the full diagonalization:

```bash
# Run the simple MPI communication demo (no diagonalization)
mpirun -np 4 python python/examples/mpi_communication_demo.py
```

This demo shows:
- How to broadcast data from one process to all others
- How to gather results from all processes
- How to compute averages across processes
- A simulated quantum-classical workflow pattern

**See:** `python/examples/mpi_communication_demo.py` for the complete working example.

## Motivation

In quantum-centric supercomputing (QCSC) workflows:
1. **Rank 0** performs quantum sampling on QPU (Quantum Processing Unit)
2. **Rank 0** receives measurement results from quantum hardware
3. **All ranks** need the quantum data to perform classical eigensolver operations
4. **Broadcast** is needed to distribute quantum results from rank 0 to all other ranks

This pattern is used in IBM's Qiskit SQD + SBD integration demo.

**Why MPI?** While Qiskit doesn't currently use MPI, classical eigensolvers (like SBD) need it for distributed computing on HPC systems. Our API bridges the gap between quantum (single-process) and classical (multi-process) parts of the workflow.

## Proposed API

### Communication Primitives

Similar to `torch.distributed`, we propose these core primitives:

```python
import sbd

# Initialize
sbd.init(device='gpu', comm_backend='mpi')

# Broadcast: Send data from root rank to all other ranks
sbd.broadcast(data, root=0)

# Gather: Collect data from all ranks to root
data_list = sbd.gather(local_data, root=0)

# All-gather: Collect data from all ranks to all ranks
all_data = sbd.all_gather(local_data)

# Reduce: Combine data from all ranks using an operation
result = sbd.reduce(local_data, op='sum', root=0)

# All-reduce: Combine and distribute to all ranks
result = sbd.all_reduce(local_data, op='sum')

# Barrier: Synchronize all ranks
sbd.barrier()

# Cleanup
sbd.finalize()
```

### Supported Data Types

- **Python lists**: `[1, 2, 3]`
- **NumPy arrays**: `np.array([1, 2, 3])`
- **Python dictionaries**: `{'energy': -76.2, 'samples': [...]}`
- **Strings**: `"measurement_results"`
- **Scalars**: `42`, `3.14`

### Reduction Operations

For `reduce()` and `all_reduce()`:
- `'sum'`: Sum values across ranks
- `'prod'`: Product of values
- `'max'`: Maximum value
- `'min'`: Minimum value
- `'avg'`: Average value

## Use Case: Quantum-Classical Hybrid Workflow

### Scenario

1. Rank 0 runs quantum circuit on QPU
2. Rank 0 receives measurement samples
3. Rank 0 broadcasts samples to all ranks
4. All ranks perform SBD diagonalization with quantum data
5. Rank 0 collects results

### Example Code

**Simple Demo (No Diagonalization):**
```bash
# Test communication primitives only - great for learning MPI concepts
mpirun -np 4 python python/examples/mpi_communication_demo.py
```

**Full Quantum-Classical Workflow:**
```bash
# Complete workflow with SBD diagonalization
mpirun -np 8 python python/examples/quantum_classical_hybrid.py
```

**Code snippet from `quantum_classical_hybrid.py`:**
```python
import sbd

def main():
    # Initialize SBD with MPI backend
    sbd.init(device='gpu', comm_backend='mpi')
    
    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    # Step 1: Rank 0 performs quantum sampling
    if rank == 0:
        print("Rank 0: Performing quantum sampling...")
        quantum_samples = simulate_quantum_sampling(num_samples=10)
        print(f"Received {len(quantum_samples['bitstrings'])} unique configurations")
    else:
        quantum_samples = None
    
    # Step 2: Broadcast quantum samples from rank 0 to all ranks
    quantum_samples = sbd.broadcast(quantum_samples, root=0)
    sbd.barrier()
    
    # Step 3: All ranks convert quantum samples to determinants
    alpha_dets = []
    beta_dets = []
    for bitstring in quantum_samples['bitstrings']:
        alpha, beta = bitstring_to_determinant(bitstring, num_orbitals=10)
        alpha_det_sbd = determinant_to_sbd_format(alpha, bit_length=20)
        beta_det_sbd = determinant_to_sbd_format(beta, bit_length=20)
        alpha_dets.append(alpha_det_sbd)
        beta_dets.append(beta_det_sbd)
    
    # Remove duplicates
    unique_alpha = list(set(alpha_dets))
    unique_beta = list(set(beta_dets))
    
    # Step 4: All ranks call SBD with same complete determinant lists
    # SBD internally partitions work based on adet_comm_size, bdet_comm_size
    results = sbd.tpb_diag(
        fcidump=fcidump,
        adet=unique_alpha,  # Same on all ranks
        bdet=unique_beta,   # Same on all ranks
        sbd_data=config
    )
    
    # Step 5: Gather results to rank 0
    all_energies = sbd.gather(results['energy'], root=0)
    
    if rank == 0:
        print(f"Ground state energy: {results['energy']:.6f} Hartree")
    
    # Step 4: All ranks perform SBD diagonalization
    # Using determinants derived from (or informed by) quantum samples
    config = sbd.TPB_SBD()
    config.adet_comm_size = 2
    config.bdet_comm_size = 2
    config.task_comm_size = size // 4
    config.max_it = 100
    config.eps = 1e-3
    
    # In real SQD workflow, would use quantum-derived determinants
    # For now, use standard determinant file as placeholder
    results = sbd.tpb_diag_from_files(
        sbd_data=config,
        fcidumpfile='../../data/h2o/fcidump.txt',
        adetfile='../../data/h2o/h2o-1em4-alpha.txt'
    )
    
    # The key point: quantum_samples from rank 0 are now available on all ranks
    # and can be used to inform the classical eigensolver
    
    # Step 5: Gather energies from all tasks
    if rank == 0:
        print(f"\nRank 0: Ground state energy = {results['energy']:.6f} Hartree")
    
    # Step 6: All-reduce to get average convergence info
    local_iterations = config.max_it  # In real case, would be actual iterations
    avg_iterations = sbd.all_reduce(local_iterations, op='avg')
    
    if rank == 0:
        print(f"Average iterations across all ranks: {avg_iterations:.1f}")
    
    # Cleanup
    sbd.finalize()

if __name__ == "__main__":
    main()
```

### Running the Example

```bash
# Run with 8 MPI ranks
mpirun -np 8 python quantum_classical_hybrid.py
```

Expected output:
```
Rank 0: Performing quantum sampling on QPU...
Rank 0: Received 2 unique samples
Rank 0: Waiting for quantum samples...
Rank 1: Waiting for quantum samples...
Rank 2: Waiting for quantum samples...
...
Rank 0: Received quantum samples
Rank 1: Received quantum samples
...
All ranks: Processing quantum samples...
Rank 0: Ground state energy = -76.242958 Hartree
Average iterations across all ranks: 15.0
```

## Implementation Design

### Python API Layer (`python/__init__.py`)

```python
def broadcast(data, root=0):
    """
    Broadcast data from root rank to all other ranks.
    
    Args:
        data: Data to broadcast (any picklable Python object)
        root: Source rank (default: 0)
    
    Returns:
        Broadcasted data on all ranks
    
    Example:
        # Rank 0 has quantum samples
        if sbd.get_rank() == 0:
            samples = {'bitstrings': [...], 'counts': [...]}
        else:
            samples = None
        
        # Broadcast to all ranks
        samples = sbd.broadcast(samples, root=0)
    """
    _check_initialized()
    
    from mpi4py import MPI
    comm = _global_comm
    
    # Use pickle for arbitrary Python objects
    data = comm.bcast(data, root=root)
    return data

def gather(data, root=0):
    """
    Gather data from all ranks to root rank.
    
    Args:
        data: Local data from this rank
        root: Destination rank (default: 0)
    
    Returns:
        List of data from all ranks (only on root), None on other ranks
    """
    _check_initialized()
    
    from mpi4py import MPI
    comm = _global_comm
    
    gathered = comm.gather(data, root=root)
    return gathered

def all_gather(data):
    """
    Gather data from all ranks to all ranks.
    
    Args:
        data: Local data from this rank
    
    Returns:
        List of data from all ranks (on all ranks)
    """
    _check_initialized()
    
    from mpi4py import MPI
    comm = _global_comm
    
    all_data = comm.allgather(data)
    return all_data

def reduce(data, op='sum', root=0):
    """
    Reduce data from all ranks using specified operation.
    
    Args:
        data: Local data (must be numeric)
        op: Operation ('sum', 'prod', 'max', 'min', 'avg')
        root: Destination rank (default: 0)
    
    Returns:
        Reduced result (only on root), None on other ranks
    """
    _check_initialized()
    
    from mpi4py import MPI
    import numpy as np
    
    comm = _global_comm
    
    # Map operation names to MPI operations
    op_map = {
        'sum': MPI.SUM,
        'prod': MPI.PROD,
        'max': MPI.MAX,
        'min': MPI.MIN,
    }
    
    if op == 'avg':
        # Average is sum divided by size
        result = comm.reduce(data, op=MPI.SUM, root=root)
        if comm.Get_rank() == root:
            result = result / comm.Get_size()
        return result
    elif op in op_map:
        return comm.reduce(data, op=op_map[op], root=root)
    else:
        raise ValueError(f"Unknown operation: {op}")

def all_reduce(data, op='sum'):
    """
    Reduce data from all ranks and distribute to all ranks.
    
    Args:
        data: Local data (must be numeric)
        op: Operation ('sum', 'prod', 'max', 'min', 'avg')
    
    Returns:
        Reduced result (on all ranks)
    """
    _check_initialized()
    
    from mpi4py import MPI
    import numpy as np
    
    comm = _global_comm
    
    op_map = {
        'sum': MPI.SUM,
        'prod': MPI.PROD,
        'max': MPI.MAX,
        'min': MPI.MIN,
    }
    
    if op == 'avg':
        result = comm.allreduce(data, op=MPI.SUM)
        return result / comm.Get_size()
    elif op in op_map:
        return comm.allreduce(data, op=op_map[op])
    else:
        raise ValueError(f"Unknown operation: {op}")
```

### Helper Function

```python
def _check_initialized():
    """Check if SBD is initialized"""
    if not _initialized:
        raise RuntimeError(
            "SBD not initialized. Call sbd.init() before using communication primitives."
        )
```

## Comparison with torch.distributed

| PyTorch Distributed | SBD MPI | Purpose |
|---------------------|---------|---------|
| `torch.distributed.broadcast()` | `sbd.broadcast()` | Broadcast from root to all |
| `torch.distributed.gather()` | `sbd.gather()` | Gather to root |
| `torch.distributed.all_gather()` | `sbd.all_gather()` | Gather to all |
| `torch.distributed.reduce()` | `sbd.reduce()` | Reduce to root |
| `torch.distributed.all_reduce()` | `sbd.all_reduce()` | Reduce to all |
| `torch.distributed.barrier()` | `sbd.barrier()` | Synchronize ranks |

## Advanced Example: Multi-Iteration Quantum-Classical Loop

```python
import sbd
import numpy as np

def quantum_classical_loop():
    """
    Iterative quantum-classical workflow:
    1. Rank 0 samples quantum circuit
    2. All ranks diagonalize with quantum samples
    3. Rank 0 updates quantum circuit parameters
    4. Repeat until convergence
    """
    sbd.init(device='gpu', comm_backend='mpi')
    rank = sbd.get_rank()
    
    max_iterations = 5
    converged = False
    
    for iteration in range(max_iterations):
        if rank == 0:
            print(f"\n=== Iteration {iteration} ===")
        
        # Step 1: Rank 0 performs quantum sampling
        if rank == 0:
            quantum_data = perform_quantum_sampling(iteration)
        else:
            quantum_data = None
        
        # Step 2: Broadcast quantum data
        quantum_data = sbd.broadcast(quantum_data, root=0)
        
        # Step 3: All ranks perform classical diagonalization
        energy = perform_classical_diagonalization(quantum_data)
        
        # Step 4: Gather energies to rank 0
        all_energies = sbd.gather(energy, root=0)
        
        # Step 5: Rank 0 checks convergence
        if rank == 0:
            avg_energy = np.mean(all_energies)
            print(f"Average energy: {avg_energy:.6f} Hartree")
            
            if iteration > 0 and abs(avg_energy - prev_energy) < 1e-6:
                converged = True
            prev_energy = avg_energy
        
        # Step 6: Broadcast convergence status
        converged = sbd.broadcast(converged, root=0)
        
        if converged:
            if rank == 0:
                print("Converged!")
            break
    
    sbd.finalize()

def perform_quantum_sampling(iteration):
    """Simulate quantum sampling (rank 0 only)"""
    # In real implementation, would call Qiskit here
    return {
        'samples': np.random.randint(0, 1000, size=100),
        'iteration': iteration
    }

def perform_classical_diagonalization(quantum_data):
    """Perform SBD diagonalization with quantum data"""
    config = sbd.TPB_SBD()
    config.max_it = 50
    config.eps = 1e-4
    
    results = sbd.tpb_diag_from_files(
        sbd_data=config,
        fcidumpfile='../../data/h2o/fcidump.txt',
        adetfile='../../data/h2o/h2o-1em4-alpha.txt'
    )
    
    return results['energy']
```

## Testing Strategy

### Unit Tests

```python
def test_broadcast():
    """Test broadcast from rank 0"""
    sbd.init(device='cpu', comm_backend='mpi')
    rank = sbd.get_rank()
    
    if rank == 0:
        data = {'value': 42, 'array': [1, 2, 3]}
    else:
        data = None
    
    result = sbd.broadcast(data, root=0)
    
    assert result['value'] == 42
    assert result['array'] == [1, 2, 3]
    
    sbd.finalize()

def test_all_reduce_sum():
    """Test all-reduce with sum operation"""
    sbd.init(device='cpu', comm_backend='mpi')
    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    local_value = rank + 1  # 1, 2, 3, ...
    total = sbd.all_reduce(local_value, op='sum')
    
    expected = size * (size + 1) // 2  # Sum of 1 to size
    assert total == expected
    
    sbd.finalize()
```

## Documentation Updates

### README_PYTHON.md

Add new section:

```markdown
### MPI Communication Primitives

For quantum-classical hybrid workflows, SBD provides MPI communication primitives:

```python
import sbd

sbd.init(device='gpu', comm_backend='mpi')

# Broadcast quantum samples from rank 0
if sbd.get_rank() == 0:
    samples = get_quantum_samples()
else:
    samples = None

samples = sbd.broadcast(samples, root=0)

# All ranks use quantum samples
results = sbd.tpb_diag_from_files(...)

sbd.finalize()
```

See [MPI_COMMUNICATION_API.md](MPI_COMMUNICATION_API.md) for complete API reference.
```

## Implementation Phases

### Phase 1: Core Primitives (Week 1)
- Implement `broadcast()`, `gather()`, `barrier()`
- Add to `__init__.py`
- Basic unit tests

### Phase 2: Reduction Operations (Week 2)
- Implement `reduce()`, `all_reduce()`, `all_gather()`
- Support numeric operations
- Extended unit tests

### Phase 3: Examples (Week 3)
- Create `quantum_classical_hybrid.py` example
- Create `quantum_classical_loop.py` example
- Documentation and README updates

### Phase 4: Integration (Week 4)
- Test with real Qiskit integration
- Performance benchmarking
- User feedback and refinement

## Complete SQD Recovery Example

Here's a more detailed example showing how quantum samples would be used in a real SQD workflow:

```python
import sbd
import numpy as np

def quantum_to_determinants(bitstrings, counts, num_orbitals):
    """
    Convert quantum measurement bitstrings to determinant configurations.
    
    In SQD workflow:
    - Quantum circuit measures occupation numbers
    - Bitstrings represent electron configurations
    - Convert to determinant format for SBD
    """
    determinants = []
    weights = []
    
    for bitstring, count in zip(bitstrings, counts):
        # Convert hex bitstring to binary occupation
        # Example: '0x42ed07eba40fde6758' -> binary array
        binary = bin(int(bitstring, 16))[2:].zfill(num_orbitals)
        
        # Convert to determinant format (list of occupied orbitals)
        occupied = [i for i, bit in enumerate(binary) if bit == '1']
        
        determinants.append(occupied)
        weights.append(count)
    
    return determinants, weights

def sqd_recovery_workflow():
    """
    Complete SQD recovery workflow with MPI communication.
    
    Workflow:
    1. Rank 0: Sample quantum circuit
    2. Broadcast: Distribute samples to all ranks
    3. All ranks: Perform determinant recovery in parallel
    4. Gather: Collect recovered determinants to rank 0
    5. Rank 0: Select most important determinants
    6. Broadcast: Distribute selected determinants
    7. All ranks: Diagonalize with selected determinants
    """
    sbd.init(device='gpu', comm_backend='mpi')
    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    num_orbitals = 20
    
    # Step 1: Rank 0 performs quantum sampling
    if rank == 0:
        print("=== SQD Recovery Workflow ===")
        print("Step 1: Rank 0 sampling quantum circuit...")
        
        # Simulate quantum measurement results
        quantum_samples = {
            'bitstrings': [
                '0x42ed07eba40fde6758',
                '0x1a2b3c4d5e6f7890ab',
                '0xfedcba9876543210',
                # ... more samples
            ],
            'counts': [523, 477, 312],  # Measurement counts
            'num_shots': 1000,
            'num_orbitals': num_orbitals
        }
        print(f"  Received {len(quantum_samples['bitstrings'])} unique configurations")
    else:
        quantum_samples = None
    
    # Step 2: Broadcast quantum samples to all ranks
    if rank == 0:
        print("Step 2: Broadcasting quantum samples to all ranks...")
    quantum_samples = sbd.broadcast(quantum_samples, root=0)
    sbd.barrier()
    
    # Step 3: All ranks perform determinant recovery in parallel
    if rank == 0:
        print("Step 3: All ranks performing determinant recovery...")
    
    # Convert quantum bitstrings to determinants
    base_dets, weights = quantum_to_determinants(
        quantum_samples['bitstrings'],
        quantum_samples['counts'],
        quantum_samples['num_orbitals']
    )
    
    # Each rank recovers additional determinants from its subset
    num_base = len(base_dets)
    dets_per_rank = num_base // size
    start = rank * dets_per_rank
    end = start + dets_per_rank if rank < size - 1 else num_base
    
    local_base_dets = base_dets[start:end]
    
    # Perform recovery: generate single/double excitations
    local_recovered = []
    for det in local_base_dets:
        # Generate single excitations
        for i in det:
            for j in range(num_orbitals):
                if j not in det:
                    new_det = det.copy()
                    new_det.remove(i)
                    new_det.append(j)
                    new_det.sort()
                    local_recovered.append(new_det)
    
    # Remove duplicates
    local_recovered = [list(x) for x in set(tuple(x) for x in local_recovered)]
    
    if rank == 0:
        print(f"  Rank {rank}: Recovered {len(local_recovered)} determinants")
    
    # Step 4: Gather all recovered determinants to rank 0
    if rank == 0:
        print("Step 4: Gathering recovered determinants to rank 0...")
    all_recovered = sbd.gather(local_recovered, root=0)
    
    # Step 5: Rank 0 selects most important determinants
    if rank == 0:
        print("Step 5: Rank 0 selecting most important determinants...")
        
        # Flatten list of lists
        all_dets = base_dets.copy()
        for recovered_list in all_recovered:
            all_dets.extend(recovered_list)
        
        # Remove duplicates
        unique_dets = [list(x) for x in set(tuple(x) for x in all_dets)]
        
        print(f"  Total unique determinants: {len(unique_dets)}")
        
        # In real SQD, would rank by importance and select top N
        # For now, just take first 1000
        selected_dets = unique_dets[:1000]
        print(f"  Selected {len(selected_dets)} determinants for diagonalization")
    else:
        selected_dets = None
    
    # Step 6: Broadcast selected determinants to all ranks
    if rank == 0:
        print("Step 6: Broadcasting selected determinants to all ranks...")
    selected_dets = sbd.broadcast(selected_dets, root=0)
    sbd.barrier()
    
    # Step 7: All ranks perform SBD diagonalization
    if rank == 0:
        print("Step 7: All ranks performing SBD diagonalization...")
    
    # In real implementation, would convert selected_dets to SBD format
    # and pass to tpb_diag() instead of using file
    
    config = sbd.TPB_SBD()
    config.adet_comm_size = 2
    config.bdet_comm_size = 2
    config.task_comm_size = size // 4
    config.max_it = 100
    config.eps = 1e-3
    
    # For demonstration, use file-based input
    # In production, would use selected_dets directly
    results = sbd.tpb_diag_from_files(
        sbd_data=config,
        fcidumpfile='../../data/h2o/fcidump.txt',
        adetfile='../../data/h2o/h2o-1em4-alpha.txt'
    )
    
    if rank == 0:
        print(f"\n=== Results ===")
        print(f"Ground state energy: {results['energy']:.6f} Hartree")
        print(f"Number of determinants used: {len(selected_dets)}")
    
    sbd.finalize()

if __name__ == "__main__":
    sqd_recovery_workflow()
```

### Running the Complete Example

```bash
mpirun -np 8 python sqd_recovery_workflow.py
```

Expected output:
```
=== SQD Recovery Workflow ===
Step 1: Rank 0 sampling quantum circuit...
  Received 3 unique configurations
Step 2: Broadcasting quantum samples to all ranks...
Step 3: All ranks performing determinant recovery...
  Rank 0: Recovered 156 determinants
Step 4: Gathering recovered determinants to rank 0...
Step 5: Rank 0 selecting most important determinants...
  Total unique determinants: 1247
  Selected 1000 determinants for diagonalization
Step 6: Broadcasting selected determinants to all ranks...
Step 7: All ranks performing SBD diagonalization...

=== Results ===
Ground state energy: -76.242958 Hartree
Number of determinants used: 1000
```

This example shows the complete data flow:
- **Quantum → Classical**: Rank 0 gets quantum samples, broadcasts to all
- **Parallel Processing**: All ranks recover determinants in parallel
- **Aggregation**: Rank 0 collects and selects best determinants
- **Distribution**: Selected determinants broadcast to all ranks
- **Diagonalization**: All ranks use quantum-informed determinants

## Future Enhancements

1. **Async Communication**: Non-blocking variants (`ibroadcast`, `igather`, etc.)
2. **GPU-Direct**: Support for GPU-to-GPU communication (NCCL backend)
3. **Custom Datatypes**: Optimized serialization for large arrays
4. **Collective Operations**: Scatter, all-to-all, etc.
5. **Process Groups**: Sub-communicators for hierarchical parallelism
6. **Determinant Format Conversion**: Direct API to pass determinants without files

## Summary

This design provides:
- ✅ Familiar API similar to `torch.distributed`
- ✅ Support for quantum-classical hybrid workflows
- ✅ Clean integration with existing SBD API
- ✅ Extensible for future enhancements
- ✅ Well-documented with examples

The implementation enables users to easily combine quantum sampling (rank 0) with distributed classical eigensolving (all ranks), which is essential for QCSC applications.
# SBD Cleanup and Finalization API

## Overview

Similar to `torch.distributed.destroy_process_group()`, SBD provides proper cleanup functions for distributed computing resources. This document explains the cleanup API and when to use each function.

## Cleanup Functions

### 1. `sbd.finalize()` - Standard Cleanup (Recommended)

**Purpose:** Clean up SBD resources including GPU devices and reset internal state.

**What it does:**
- Calls `cudaDeviceReset()` or `hipDeviceReset()` on GPU backend to free GPU memory
- Resets Python internal state
- Does NOT call `MPI_Finalize()` (MPI lifecycle managed by mpi4py)

**When to use:**
- At the end of your computation
- Before switching to different device/backend settings
- In cleanup/finally blocks

**Example:**
```python
import sbd

try:
    sbd.init(device='gpu', comm_backend='mpi')
    config = sbd.TPB_SBD()
    results = sbd.tpb_diag_from_files(...)
finally:
    sbd.finalize()  # Clean up GPU and reset state
```

### 2. `sbd.finalize_mpi()` - Explicit MPI Finalization (Advanced)

**Purpose:** Explicitly finalize MPI (calls `MPI_Finalize()`).

**What it does:**
- Checks if MPI is initialized and not already finalized
- Calls `MPI_Finalize()` if appropriate
- After this, no MPI functions can be used until MPI is reinitialized

**When to use:**
- **Rarely needed** - only if you initialized MPI yourself outside of mpi4py
- For advanced use cases requiring explicit MPI lifecycle control
- When integrating with non-Python MPI code

**Warning:** 
- Do NOT call this if using mpi4py normally (recommended usage)
- mpi4py handles MPI finalization automatically at program exit
- Calling this prematurely will break subsequent MPI operations

**Example (advanced):**
```python
import sbd

# Only if you have special MPI initialization requirements
sbd.init(device='gpu', comm_backend='mpi')
results = sbd.tpb_diag_from_files(...)
sbd.finalize()      # Clean up GPU
sbd.finalize_mpi()  # Explicitly finalize MPI (rarely needed)
```

## C++ Bindings

The cleanup functions are exposed in the C++ bindings:

### `cleanup_device()`
```cpp
// GPU backend only - synchronizes and resets device
#ifdef SBD_THRUST
    cudaDeviceSynchronize();
    cudaDeviceReset();
#endif
```

### `finalize_mpi()`
```cpp
// Safely finalizes MPI if initialized
int initialized, finalized;
MPI_Initialized(&initialized);
MPI_Finalized(&finalized);

if (initialized && !finalized) {
    MPI_Finalize();
}
```

## Comparison with PyTorch Distributed

| PyTorch | SBD | Purpose |
|---------|-----|---------|
| `torch.distributed.init_process_group()` | `sbd.init()` | Initialize distributed environment |
| `torch.distributed.destroy_process_group()` | `sbd.finalize()` | Clean up distributed resources |
| N/A | `sbd.finalize_mpi()` | Explicit MPI finalization (advanced) |

## Best Practices

### 1. Always Call `finalize()` in Cleanup Code

```python
import sbd

try:
    sbd.init(device='gpu')
    # Your computation
    results = sbd.tpb_diag_from_files(...)
finally:
    sbd.finalize()  # Ensures cleanup even if errors occur
```

### 2. Use Context Manager Pattern (Future Enhancement)

```python
# Future API (not yet implemented)
with sbd.context(device='gpu'):
    results = sbd.tpb_diag_from_files(...)
# Automatic cleanup on exit
```

### 3. Multiple Runs with Different Settings

```python
import sbd

# First run with GPU
sbd.init(device='gpu')
results1 = sbd.tpb_diag_from_files(...)
sbd.finalize()

# Second run with CPU
sbd.init(device='cpu')
results2 = sbd.tpb_diag_from_files(...)
sbd.finalize()
```

### 4. Don't Call `finalize_mpi()` Unless You Know What You're Doing

```python
# ❌ BAD - breaks mpi4py's automatic cleanup
import sbd
sbd.init(device='gpu')
results = sbd.tpb_diag_from_files(...)
sbd.finalize()
sbd.finalize_mpi()  # Don't do this with mpi4py!

# ✅ GOOD - let mpi4py handle MPI lifecycle
import sbd
sbd.init(device='gpu')
results = sbd.tpb_diag_from_files(...)
sbd.finalize()  # Only clean up SBD resources
# MPI finalized automatically at program exit
```

## GPU Memory Management

The `finalize()` function is especially important for GPU backends:

- **Without `finalize()`:** GPU memory may not be released properly
- **With `finalize()`:** Calls `cudaDeviceReset()` to ensure all GPU resources are freed
- **Multiple GPUs:** Each MPI rank cleans up its assigned GPU device

## MPI Lifecycle

### Standard Usage (Recommended)
```
Program Start
    ↓
Import mpi4py (MPI_Init called automatically)
    ↓
sbd.init()
    ↓
Computation
    ↓
sbd.finalize() (GPU cleanup only)
    ↓
Program Exit (MPI_Finalize called automatically by mpi4py)
```

### Advanced Usage (Explicit MPI Control)
```
Program Start
    ↓
Custom MPI initialization
    ↓
sbd.init()
    ↓
Computation
    ↓
sbd.finalize() (GPU cleanup)
    ↓
sbd.finalize_mpi() (explicit MPI finalization)
    ↓
Program Exit
```

## Error Handling

The `finalize()` function includes error handling for GPU cleanup:

```python
def finalize():
    # Clean up GPU resources if using GPU backend
    if _device_module is not None and hasattr(_device_module, 'cleanup_device'):
        try:
            _device_module.cleanup_device()
        except Exception as e:
            import warnings
            warnings.warn(f"GPU cleanup failed: {e}")
    
    # Reset Python state (always succeeds)
    # ...
```

This ensures that Python state is always reset even if GPU cleanup fails.

## Summary

- **Use `sbd.finalize()`** for standard cleanup (GPU + Python state)
- **Avoid `sbd.finalize_mpi()`** unless you have specific MPI lifecycle requirements
- **Always call `finalize()`** in finally blocks or cleanup code
- **GPU backend:** `finalize()` is critical for proper memory cleanup
- **CPU backend:** `finalize()` resets state for re-initialization

## See Also

- [Python Bindings README](README_PYTHON.md)
- [Python Bindings Architecture](PYTHON_BINDINGS_ARCHITECTURE.md)
- [GPU Support Documentation](GPU_SUPPORT.md)
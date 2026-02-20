# SBD Python Examples

## H2O Example (h2o_cpu_gpu.py)

Comprehensive example demonstrating TPB (Two-Particle Basis) calculations on H2O molecule with both CPU and GPU backends. All TPB_SBD configuration parameters are accessible via command-line arguments.

### Features
- Full command-line control over all 30+ TPB_SBD parameters
- Backend selection (CPU/GPU/auto)
- Configurable MPI communicator sizes
- Multiple diagonalization methods (Davidson, Lanczos)
- Carryover determinant selection
- RDM calculation options
- Wavefunction save/load support

### Quick Start

**CPU Backend (8 MPI ranks, 4 OpenMP threads each):**
```bash
mpirun -np 8 -x OMP_NUM_THREADS=4 python h2o_cpu_gpu.py \
    --device cpu \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --eps 1e-4
```

**GPU Backend (8 MPI ranks on 8 GPUs):**
```bash
mpirun -np 8 python h2o_cpu_gpu.py \
    --device gpu \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --eps 1e-4
```

### All Command-Line Arguments

Run `python h2o_cpu_gpu.py --help` for complete list. Key options:

#### Device Selection
- `--device {auto,cpu,gpu}` - Backend selection (default: auto)

#### Input Files
- `--fcidump FILE` - FCIDUMP file (default: `../../data/h2o/fcidump.txt`)
- `--adetfile FILE` - Alpha determinants file (required)
- `--bdetfile FILE` - Beta determinants file (optional, uses adetfile if not set)
- `--loadname FILE` - Load initial wavefunction from file
- `--savename FILE` - Save final wavefunction to file

#### MPI Configuration
- `--task_comm_size N` - Task communicator size (default: 1)
- `--adet_comm_size N` - Alpha determinant communicator size (default: 1)
- `--bdet_comm_size N` - Beta determinant communicator size (default: 1)
- `--h_comm_size N` - Helper communicator size (default: 1)

**Note:** Total MPI ranks = `task_comm_size × adet_comm_size × bdet_comm_size`

#### Diagonalization Method
- `--method {0,1,2,3}` - Method selection (default: 0)
  - 0 = Davidson
  - 1 = Davidson + Hamiltonian
  - 2 = Lanczos
  - 3 = Lanczos + Hamiltonian
- `--max_it N` - Maximum iterations (default: 100)
- `--max_nb N` - Maximum basis vectors (default: 10)
- `--eps FLOAT` - Convergence tolerance (default: 1e-3)
- `--max_time SECONDS` - Maximum time limit (default: 1e10)

#### Options
- `--init N` - Initialization method (default: 0)
- `--do_shuffle {0,1}` - Shuffle determinants (default: 0)
- `--do_rdm {0,1}` - Calculate RDM: 0=density only, 1=full RDM (default: 0)
- `--bit_length N` - Bit length for determinants (default: 20)

#### Carryover Determinant Selection
- `--carryover_type N` - Selection type (default: 0)
- `--ratio FLOAT` - Carryover ratio (default: 0.0)
- `--threshold FLOAT` - Carryover threshold (default: 0.0)

#### Output
- `--dump_matrix_form_wf FILE` - Dump wavefunction in matrix form

#### GPU-Specific (only used with GPU backend)
- `--use_precalculated_dets {0,1}` - Use precalculated determinants (default: 1)
- `--max_memory_gb_for_determinants N` - Max GPU memory in GB (integer), -1=auto (default: -1)

### Advanced Usage Examples

**High-accuracy calculation with RDM:**
```bash
mpirun -np 8 python h2o_cpu_gpu.py \
    --device cpu \
    --method 0 \
    --max_it 200 \
    --max_nb 30 \
    --eps 1e-8 \
    --do_rdm 1 \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em6-alpha.txt \
    --savename h2o_wf_1em6.dat
```

**GPU with carryover selection:**
```bash
mpirun -np 8 python h2o_cpu_gpu.py \
    --device gpu \
    --method 0 \
    --eps 1e-5 \
    --carryover_type 1 \
    --ratio 0.1 \
    --threshold 1e-6 \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em5-alpha.txt \
    --max_memory_gb_for_determinants 16
```

**Restart from saved wavefunction:**
```bash
mpirun -np 8 python h2o_cpu_gpu.py \
    --device cpu \
    --loadname h2o_wf_1em6.dat \
    --savename h2o_wf_1em7.dat \
    --eps 1e-9 \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em7-alpha.txt
```

### MPI Configuration Examples

Total ranks must equal product of communicator sizes:

- **8 ranks:** `--task_comm_size 2 --adet_comm_size 2 --bdet_comm_size 2` (2×2×2=8)
- **16 ranks:** `--task_comm_size 2 --adet_comm_size 2 --bdet_comm_size 4` (2×2×4=16)
- **32 ranks:** `--task_comm_size 4 --adet_comm_size 4 --bdet_comm_size 2` (4×4×2=32)

### Available Determinant Files

Located in `../../data/h2o/`:
- `h2o-1em3-alpha.txt` - 10⁻³ threshold (~100 determinants)
- `h2o-1em4-alpha.txt` - 10⁻⁴ threshold (~1,000 determinants)
- `h2o-1em5-alpha.txt` - 10⁻⁵ threshold (~10,000 determinants)
- `h2o-1em6-alpha.txt` - 10⁻⁶ threshold (~100,000 determinants)
- `h2o-1em7-alpha.txt` - 10⁻⁷ threshold (~1,000,000 determinants)
- `h2o-1em8-alpha.txt` - 10⁻⁸ threshold (~10,000,000 determinants)

### Expected Results

Ground state energy for H2O: approximately **-76.236 Hartree**

Convergence depends on:
- Determinant threshold (smaller = more accurate)
- Convergence tolerance (`--eps`)
- Number of iterations (`--max_it`)

### Performance Tips

**CPU Backend:**
- Set `OMP_NUM_THREADS` to cores per MPI rank
- Balance MPI ranks vs threads based on memory
- Example: 8 ranks × 4 threads = 32 cores

**GPU Backend:**
- One MPI rank per GPU is optimal
- Set `OMP_NUM_THREADS=1` (GPU does the work)
- Each rank automatically assigned to GPU: `rank % num_gpus`
- Example: 8 ranks on 8 GPUs = 1 rank per GPU

**MPI Decomposition:**
- Larger problems benefit from more parallelization
- Balance `adet_comm_size`, `bdet_comm_size`, `task_comm_size`
- Start with equal sizes and adjust based on profiling

### Resource Cleanup

All examples properly clean up resources using `sbd.finalize()`:

```python
try:
    sbd.init(device='gpu')
    results = sbd.tpb_diag_from_files(...)
finally:
    sbd.finalize()  # Cleans up GPU memory and resets state
```

**Important:** Always call `sbd.finalize()` to:
- Free GPU memory (calls `cudaDeviceReset()` on GPU backend)
- Reset internal state for re-initialization
- Ensure proper cleanup similar to `torch.distributed.destroy_process_group()`

Note: `finalize()` does NOT call `MPI_Finalize()` - that's handled automatically by mpi4py.

### Notes

- **TPB Method Only:** This example uses Two-Particle Basis (TPB) diagonalization for quantum chemistry
- **GPU Requirements:** NVIDIA HPC SDK and CUDA-capable GPUs
- **Automatic GPU Assignment:** Each MPI rank assigned to GPU automatically
- **CUDA-aware MPI:** Recommended for best GPU performance
- **Memory:** Larger determinant files require more memory per rank
- **Cleanup:** Always call `sbd.finalize()` in finally blocks for proper resource cleanup

### Troubleshooting

**"Total ranks mismatch":**
- Ensure `mpirun -np N` matches `task_comm_size × adet_comm_size × bdet_comm_size`

**"File not found":**
- Check paths to FCIDUMP and determinant files
- Use absolute paths or correct relative paths

**GPU out of memory:**
- Reduce `--max_memory_gb_for_determinants`
- Use smaller determinant file
- Increase number of MPI ranks to distribute memory

**Slow convergence:**
- Increase `--max_it`
- Adjust `--max_nb` (more basis vectors)
- Try different `--method`
- Use better initial guess with `--loadname`

For more details, see the main Python bindings documentation: `../../README_PYTHON.md`
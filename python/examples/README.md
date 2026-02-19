# SBD Python Examples

## H2O Example (h2o_cpu_gpu.py)

Comprehensive example demonstrating SBD calculations on H2O molecule with both CPU and GPU backends.

### Features
- Command-line argument parsing for all parameters
- Backend selection (CPU/GPU)
- Configurable MPI communicator sizes
- Multiple convergence tolerance options
- Support for different determinant files

### Usage

**CPU Backend (8 MPI ranks, 4 OpenMP threads each):**
```bash
$MPI_HOME/bin/mpirun --allow-run-as-root -np 8 -x OMP_NUM_THREADS=4 \
    python h2o_cpu_gpu.py \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --tolerance 1e-4 \
    --device cpu
```

**GPU Backend (8 MPI ranks on 8 GPUs):**
```bash
$MPI_HOME/bin/mpirun --allow-run-as-root -np 8 -x OMP_NUM_THREADS=4 \
    python h2o_cpu_gpu.py \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --adetfile ../../data/h2o/h2o-1em4-alpha.txt \
    --tolerance 1e-4 \
    --device gpu
```

### Command-Line Arguments

- `--device`: Backend selection (`cpu` or `gpu`, default: `cpu`)
- `--adetfile`: Path to alpha determinants file
- `--fcidump`: Path to FCIDUMP file (default: `../../data/h2o/fcidump.txt`)
- `--tolerance`: Convergence tolerance (default: `1e-6`)
- `--max_iterations`: Maximum iterations (default: `100`)
- `--adet_comm_size`: Alpha determinant communicator size (default: `1`)
- `--bdet_comm_size`: Beta determinant communicator size (default: `1`)
- `--task_comm_size`: Task communicator size (default: `1`)

### MPI Configuration

Total MPI ranks must equal: `adet_comm_size × bdet_comm_size × task_comm_size`

Examples:
- 8 ranks: `2 × 2 × 2`
- 16 ranks: `2 × 2 × 4` or `4 × 2 × 2`
- 32 ranks: `4 × 4 × 2` or `2 × 4 × 4`

### Available Determinant Files

Located in `../../data/h2o/`:
- `h2o-1em3-alpha.txt` - 10⁻³ threshold
- `h2o-1em4-alpha.txt` - 10⁻⁴ threshold
- `h2o-1em5-alpha.txt` - 10⁻⁵ threshold
- `h2o-1em6-alpha.txt` - 10⁻⁶ threshold
- `h2o-1em7-alpha.txt` - 10⁻⁷ threshold
- `h2o-1em8-alpha.txt` - 10⁻⁸ threshold

### Expected Results

Ground state energy for H2O: approximately **-76.236 Hartree**

### Notes

- GPU backend requires NVIDIA HPC SDK and CUDA-capable GPUs
- Each MPI rank is automatically assigned to a GPU (rank % num_gpus)
- CPU backend uses OpenMP parallelization (set via `OMP_NUM_THREADS`)
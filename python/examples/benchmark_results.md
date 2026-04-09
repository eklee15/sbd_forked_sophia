# SQD + SBD Benchmark Results

## QPU Sampling Time (ibm_fez)

72-qubit LUCJ circuit for [4Fe-4S] (36 orbitals, 54 electrons).
Transpiled with optimization_level=3, auto layout.

| Shots       | QPU Time | Per-shot | Wall Time |
|------------:|---------:|---------:|----------:|
| 1,000       |     3.0s |  3.0 ms  |    93.6s  |
| 10,000      |     5.0s |  0.5 ms  |   273.7s  |
| 1,000,000   |   303.0s |  0.3 ms  |   381.7s  |

- Wall time = transpilation + queue wait + QPU execution
- Per-shot cost converges to ~0.3 ms; small-shot runs are dominated by fixed overhead (~2-3s)
- Date: 2026-04-09, backend: ibm_fez (156 qubits, Heron)

## Classical SQD Runs (SBD solver)

### Server: Intel Sapphire Rapids + 8×NVIDIA H100

| Component | Spec |
|-----------|------|
| CPU | 2× Intel Xeon (Sapphire Rapids), 40 cores/socket, 80 cores total (160 threads with HT) |
| Memory | 1.7 TB DDR5 |
| GPU | 8× NVIDIA H100 80GB HBM3 (640 GB total GPU memory) |

All runs: 8 MPI ranks, 20 OMP threads/rank, adet_comm_size=4, bdet_comm_size=2, do_rdm=0.

### Client system (NORB=29, NELEC=10, MS2=0)

29 spatial orbitals, 10 electrons (5α, 5β), 58 qubits.
Input: 1M hardware bitstrings (all unique).

Doubling `samples_per_batch` from 10K to 20K increases the subspace dimension
from ~1.4M to ~3.7M. The larger subspace captures more of the Hilbert space,
yielding a lower (better) energy. Convergence across SQD iterations is also
faster — the 20K run reaches -103.593 by iteration 2, while the 10K run needs
5 iterations to reach -103.593.

| samples_per_batch | Subspace dim | Energy (Ha)  | Time    | GPU mem | GPU util |
|------------------:|-------------:|-------------:|--------:|--------:|---------:|
| 10,000            |    1,352,569 | -103.5927851 |  739.7s |  16.0 GB |     22% |
| 20,000            |    3,724,900 | -103.5934994 | 1563.1s |  41.6 GB |     45% |
| 50,000            |          OOM |            — |       — |  >79.6 GB |       — |

Resource details:

| Metric | Aggregation | 10K run | 20K run |
|--------|-------------|------:|------:|
| Wall time | — | 739.7s (~12 min) | 1563.1s (~26 min) |
| CPU cores allocated | sum across ranks | 160 / 160 | 160 / 160 |
| CPU utilization | sum across ranks | 112% (1% of allocated) | 114% (1% of allocated) |
| CPU peak RSS | max across ranks | 2.2 GB | 4.3 GB |
| CPU memory used | sum across ranks on node | 15.1 / 1763.3 GB | 28.9 / 1763.3 GB |
| GPU memory | max across ranks | 16.0 / 79.6 GB | 41.6 / 79.6 GB |
| GPU utilization | avg across ranks | 22% | 45% |

At 50K samples_per_batch, the subspace exceeds 80 GB per-GPU memory.
Scaling beyond 20K would require more GPUs or larger bdet_comm_size to
distribute the subspace across more ranks.

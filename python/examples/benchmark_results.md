# SQD + SBD Benchmark Results

## QPU Sampling Time (ibm_fez)

72-qubit LUCJ circuit for [4Fe-4S] (36 orbitals, 54 electrons).
Transpiled with optimization_level=3, auto layout.

| Shots       | QPU Time | Wall Time |
|------------:|---------:|----------:|
| 1,000       |     3.0s |    93.6s  |
| 10,000      |     5.0s |   273.7s  |
| 1,000,000   |   303.0s |   381.7s  |

- Wall time = transpilation + queue wait + QPU execution
- QPU time scales ~linearly with shots
- Date: 2026-04-09, backend: ibm_fez (156 qubits, Heron)

## Classical SBD Runs (8×H100 GPUs)

<!-- Add rows as new runs are completed -->

| System | samples_per_batch | num_batches | SQD iters | Subspace dim | Energy (Ha) | Time | Notes |
|--------|------------------:|------------:|----------:|-------------:|------------:|-----:|-------|
| [4Fe-4S] | 300 | 1 | 5 | 4,356 | -325.8409 | 509s | 1M hw shots, do_rdm=0 |

Resource usage (8×H100, 20 OMP threads/rank):
- CPU peak RSS: 1.6 GB (rank 0)
- GPU memory: 1.0 / 79.6 GB
- GPU utilization: ~1% (subspace too small)

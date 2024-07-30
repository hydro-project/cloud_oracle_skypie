# Setup

## SkyPIE dependencies
Install Python SkyPIE packages.
Linux on x86:
```
python3 -m pip install --force-reinstall \
    ./dependencies_compiled/skypie_baselines-0.2-cp37-abi3-manylinux_2_34_x86_64.whl \
    ./dependencies_compiled/skypie_proto_messages-0.1-cp37-abi3-manylinux_2_34_x86_64.whl
```
MacOSx on Arm:
```
python3 -m pip install --force-reinstall \
    ./dependencies_compiled/skypie_baselines-0.2-cp37-abi3-macosx_11_0_arm64.whl \
    ./dependencies_compiled/skypie_proto_messages-0.1-cp37-abi3-macosx_11_0_arm64
```

## SkyPIE Oracle

Initial installation of SkyPIE and all its external dependencies:
```
python3 -m pip install .
```


Update of SkyPIE when the files have changed but not the package version:
```
python3 -m pip install --force-reinstall --no-deps .
```

# Usage

## Querying the SkyPIE Oracle (online optimization)
See [examples](./examples/simple_skypie_example.py) how to use the oracle API or the ILP baseline.

## Precomputing SkyPIE Oracles

- There are default oracles in [examples/oracles](./examples/oracles/).
- Further oracles for other cloud deployments, SLOs,... can be precomputed as explained in the [SkyPIE precomputer repo](https://github.com/hydro-project/cloud_oracle_precomputer).

# Notes

- For compiled oracle queries on Mac/M1 pytorch a very recent version like nightly seems to be required.
- The final rounding of binary variables in the ILP has been changed to a threshold of >0.5 (from >0.0)

# Rerun Experiments

## Online Optimization Performance

- Precomputation inside Docker: `python -m deploy --experiment sigmod_scaling --redundancy-elimination-workers=60 [--output-dir=<root result directory>/precomputation_scaling]`
- Experiment "Online Optimization Time" (Fig. 7 a-b): `python3 -m skypie scaling [directory of precomputed oracle]`
- Experiment "Online Optimization Time by Batch Size" (Fig. 7 c): `python3 -m skypie query_batching [directory of precomputed oracle]`
- Experiment "Online Optimization Accuracy" (Fig. 7 d): **Extract results from scaling experiment?**

## Precomputation Performance

- Precomputation inside Docker (Fig. 8a): _reuse sigmod scaling_
- Precomputation batching for Azure-AWS f=3, inside Docker (Fig 8 b-c): `python -m deploy --experiment sigmod_batch_size --redundancy-elimination-workers=60 [--output-dir=<root result directory>/precomputation_batching]`
- Query time under precomputation batching (Fig 8 d): `python3 -m skypie precomputation_batching [directory of precomputed batched oracles]`

## Real Trace

- [Download trace files](doi.org/10.5281/zenodo.13129407)
- Execute precomputation inside Docker container: `python -m deploy --experiment sigmod_real_trace --redundancy-elimination-workers=60 [--output-dir=<root result directory>/precomputation_real_trace]`
- Execute experiment: `python3 -m skypie real_trace [directory of precomputed oracle] --trace_dir [directory of downloaded trace]`
    - Overwrite CUDA device with `--torchDeviceRayShooting cuda:X` or skip cuda with `--torchDeviceRayShooting cpu`
    - Overwrite output directory with `--output [directory]`
- Result location default, relative to working directory: `results/real_trace/real_trace_result.pandas.pickle`

# TODOs
- Compare performance of compile versus interpreted querying

### Bugs
(No known bugs)

### General
- Briefly explain the oracle query API
- Upload processed ubuntu trace
- Cleanup experiments
- Publish SkyPIE packages

# Publication

[Link to paper](https://dl.acm.org/doi/10.1145/3639310)
```
@article{10.1145/3639310,
    author = {Bang, Tiemo and Douglas, Chris and Crooks, Natacha and Hellerstein, Joseph M.},
    title = {SkyPIE: A Fast \& Accurate Oracle for Object Placement},
    year = {2024},
    issue_date = {February 2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {2},
    number = {1},
    url = {https://doi.org/10.1145/3639310},
    doi = {10.1145/3639310},
    journal = {Proc. ACM Manag. Data},
    month = {mar},
    articleno = {55},
    numpages = {27},
    keywords = {cloud oracle, data placement, exact, object placement, offline}
}
```

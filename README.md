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

# TODOs

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

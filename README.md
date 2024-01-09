# Setup

## SkyPIE dependencies
Install Python SkyPIE packages.
Linux on x86:
```
python3 -m pip install --force-reinstall \
    ./dependencies_compiled/sky_pie_baselines-0.1.0-cp37-abi3-manylinux_2_34_x86_64.whl \
    ./dependencies_compiled/sky_pie_precomputer_proto_messages-0.0.0-cp37-abi3-manylinux_2_34_x86_64.whl
```
MacOSx on Arm:
```
python3 -m pip install --force-reinstall \
    ./dependencies_compiled/sky_pie_baselines-0.1.0-cp37-abi3-macosx_11_0_arm64.whl \
    ./dependencies_compiled/sky_pie_precomputer_proto_messages-0.0.0-cp37-abi3-macosx_11_0_arm64.whl
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

# TODOs

### Bugs
- ILP/Oracle scale of costs are off, despite correct and identical placement policies.
- ILP rounds read choices too conservatively

### General
- Briefly explain the oracle query API
- Publish SkyPIE packages
- Upload processed ubuntu trace
- Cleanup experiments
- Verify that precomputation still works with the cleaned-up redundancy elimination
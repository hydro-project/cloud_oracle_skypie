import os
import psutil
import numpy as np
import pandas as pd
from ast import literal_eval
from typing import List

from skypie.oracle import Oracle
from skypie.util.my_dataclasses import OptimizerType
from skypie.experiments.benchmarking import benchmarkQuerying

def query_trace(*, accessSetRegionMapping: str, inputFileName: str, scenarioName: str, addOptimizersFromInput: bool, optimizer: List[OptimizerType], noWorkloads: int, no_warmup: bool, translateOptSchemes: bool, output_file: str, skipWorkloadResults: bool, batchSizes: List[int], implArgs=dict(), verbose=0, filterTimestampByDate=None, inputWorkloads: List[str], sizeScale=1, ingressScale=1, egressScale=1, rescale=1, output=None, exp_args=dict(), skip_loaded_optimizers=False):
    """
    Query the oracle for the optimum of the given access set workload file.

    Args:
        accessSetRegionMapping (str): The mapping of region name to index in the arrays of the access set.
        inputFileName (str): The name of the input file containing the oracle.
        scenarioName (str): The name of the scenario, can be "default".
        addOptimizersFromInput (bool): Whether to add optimizers from the input.
        optimizer (List[OptimizerType]): The list of optimizers to explicitly benchmark, e.g., "ILP".
        noWorkloads (int): The number of workloads to take from the workload trace file.
        no_warmup (bool): Whether to skip warmup.
        translateOptSchemes (bool): Whether to translate optimization schemes.
        output_file (str): The output file.
        skipWorkloadResults (bool): Whether to skip workload results.
        batchSizes (List[int]): The list of batch sizes to benchmark.
        implArgs (dict, optional): Detailed implementation arguments forwarded to the oracle. Defaults to an empty dictionary.
        verbose (int, optional): The verbosity level. Defaults to 0.
        filterTimestampByDate (str, optional): The timestamp filter. Defaults to None.
        inputWorkloads (List[str]): The list of input files for the workload trace to benchmark.
        sizeScale (int, optional): Factor to rescale object size in the trace to normalize units if necessary. Defaults to 1.
        ingressScale (int, optional): Factor to rescale ingress in the trace to normalize units if necessary. Defaults to 1.
        egressScale (int, optional): Factor to rescale egress in the trace to normalize units if necessary. Defaults to 1.
        rescale (int, optional): General factor to rescale all workload features, e.g., to prevent over/underflows. Defaults to 1.
        output (str, optional): The output path. Defaults to None.
        exp_args (dict, optional): The experiment arguments. Defaults to an empty dictionary.
        skip_loaded_optimizers (bool, optional): Whether to skip loaded optimizers. Defaults to False.

    Yields:
        r: The result of the benchmark querying.
    """
    
    benchmark_args = dict(scenarioName=scenarioName, optimizers=optimizer, batchSizes=batchSizes, no_warmup=no_warmup, translateOptSchemes=translateOptSchemes, noWorkloadResults=skipWorkloadResults, exp_args=exp_args)
    
    print ("Querying the oracle for the optimum of the given access set workload file.")

    oracle, optimizer = Oracle.setup_oracle(inputFileName=inputFileName, scenarioName=scenarioName, addOptimizersFromInput=addOptimizersFromInput, optimizer=optimizer, implArgs=implArgs, verbose=verbose, skip_loaded_optimizers=skip_loaded_optimizers)

    # Mapping of region name to index in the arrays of the access set
    access_set_region_mapping = { a.split(":")[0]: int(a.split(":")[1]) for a in accessSetRegionMapping}
    # Mapping of the region name to the index in the workload
    regions = oracle.get_application_regions()
    
    for inputWorkloadFile in inputWorkloads:
        print(f"Loading workload file {inputWorkloadFile}")

        if inputWorkloadFile.endswith(".pickle") or inputWorkloadFile.endswith(".pickle.zip"):
            workloadTrace = pd.read_pickle(inputWorkloadFile)
        elif inputWorkloadFile.endswith(".parquet"):
            import polars as ps
            workloadTrace = ps.read_parquet(inputWorkloadFile).to_pandas()
        elif inputWorkloadFile.endswith(".csv"):
            workloadTrace = pd.read_csv(inputWorkloadFile, sep=";", converters={"get": literal_eval, "put": literal_eval, "ingress": literal_eval, "egress": literal_eval})
        else:
            raise ValueError(f"Unknown workload file format {inputWorkloadFile}")
        
        print(f"Loaded workloads. RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

        if filterTimestampByDate:
            filterTimeStamp = pd.to_datetime(filterTimestampByDate, format='%Y-%m-%d_%H:%M:%S')
            if workloadTrace["timestamp"].dtype == int:
                filterTimeStamp = filterTimeStamp.value
            workloadTrace = workloadTrace[workloadTrace["timestamp"] <= filterTimeStamp]

            print(f"Trace records after filtering by date (<= {filterTimestampByDate}): {workloadTrace.shape[0]}")

        isSingleRegionTrace = ("get" in workloadTrace.columns and \
                "put" in workloadTrace.columns and \
                "egress" in workloadTrace.columns and \
                "ingress" in workloadTrace.columns and \
                "size" in workloadTrace.columns and \
                (workloadTrace["get"].dtype == np.int64 or workloadTrace["get"].dtype == np.float64))

        if isSingleRegionTrace:
            assert len(access_set_region_mapping) == 1, f"Provide a multi-region workload trace or a mapping of the access set with multiple regions. isSingleRegionTrace={isSingleRegionTrace} access_set_region_mapping={access_set_region_mapping}"

        if isSingleRegionTrace:
            r = list(access_set_region_mapping.keys())[0]

        workloads = []
        for w in workloadTrace.itertuples():
            get = np.zeros(oracle.no_apps)
            put = np.zeros(oracle.no_apps)
            ingress = np.zeros(oracle.no_apps)
            egress = np.zeros(oracle.no_apps)

            if isSingleRegionTrace:
                size = w.size
                get[regions[r]] = w.get
                put[regions[r]] = w.put
                ingress[regions[r]] = w.ingress * ingressScale
                egress[regions[r]] = w.egress * egressScale
            else:
                size = w.size
                for (r, access_set_i) in access_set_region_mapping.items():
                    get[regions[r]] = w.get[access_set_i]
                    put[regions[r]] = w.put[access_set_i]
                    ingress[regions[r]] = w.ingress[access_set_i] * ingressScale
                    egress[regions[r]] = w.egress[access_set_i] * egressScale
            # Only add workload if there is at least one access
            if np.max(get) > 0 or np.max(put) > 0 or np.max(ingress) > 0 or np.max(egress) > 0:
                workloads.append(oracle.create_workload(size=size*sizeScale, put=put, get=get, ingress=ingress, egress=egress, rescale=rescale))

            if len(workloads) >= noWorkloads:
                break

        print(f"Processed workloads. RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

        if output:
            output_file = output + f"_{inputWorkloadFile.split('/')[-1].split('.')[0]}_accessSet_{len(accessSetRegionMapping)}"

        print(f"Processing {len(workloads)} workloads of file {inputWorkloadFile}")
        for r in benchmarkQuerying(oracle, workloads=workloads, output=output_file, **benchmark_args):
            yield r

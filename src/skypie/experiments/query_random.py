import os
import sys
import psutil
import numpy as np
from typing import List, Tuple

from skypie.oracle import Oracle
from skypie.util.my_dataclasses import OptimizerType
from skypie.experiments.benchmarking import benchmarkQuerying

def query_random(*, inputFileName: str, scenarioName: str, addOptimizersFromInput: bool, optimizer: List[OptimizerType], workloadSeed: int, workloadRange: Tuple[float,float], noWorkloads: int, queryStepSize: int, querySkip: int, no_warmup: bool, translateOptSchemes: bool, output_file: str, skipWorkloadResults: bool, batchSizes: List[int], implArgs=dict(), exp_args=dict(), verbose=0):
    """
    Randomly query the oracle for the optimum for synthetic workloads.

    Args:
        inputFileName (str): The input file name.
        scenarioName (str): The scenario name.
        addOptimizersFromInput (bool): Whether to add optimizers from input.
        optimizer (List[OptimizerType]): The list of optimizers.
        workloadSeed (int): The seed for generating random workloads.
        workloadRange (Tuple[float, float]): The range for generating random workloads.
        noWorkloads (int): The number of workloads to generate.
        queryStepSize (int): The step size for querying workloads.
        querySkip (int): The number of workloads to skip before querying.
        no_warmup (bool): Whether to skip warm-up.
        translateOptSchemes (bool): Whether to translate optimization schemes.
        output_file (str): The output file name.
        skipWorkloadResults (bool): Whether to skip workload results.
        batchSizes (List[int]): The list of batch sizes.
        implArgs (dict, optional): Additional implementation arguments. Defaults to an empty dictionary.
        exp_args (dict, optional): Additional experiment arguments. Defaults to an empty dictionary.
        verbose (int, optional): The verbosity level. Defaults to 0.

    Returns:
        The result of the benchmark querying.
    """
    
    print ("Querying the oracle for the optimum for synthetic workloads")

    oracle, optimizer = Oracle.setup_oracle(inputFileName=inputFileName, scenarioName=scenarioName, addOptimizersFromInput=addOptimizersFromInput, optimizer=optimizer, implArgs=implArgs, verbose=verbose)

    # Randomly generate workloads
    noApps = oracle.no_apps
    rnd = np.random.default_rng(seed=workloadSeed)
    # Generate random size, put and get
    size = 1 + 2*noApps
    workloads = rnd.random(size=(noWorkloads, size))
    # Shift into given range
    workloads = (workloads * (workloadRange[1] - workloadRange[0])) + workloadRange[0]
    # Instantiate workloads
    workloads = [oracle.create_workload(size=w[0], put=w[1:1+noApps], get=w[1+noApps:1+2*noApps]) for w in workloads]
    # Convert back to numpy array
    #workloadsNumpy = np.array([w.equation for w in workloads])

    print(f"Loaded workloads. RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

    # workloads=workloads[queryStep:queryStep+queryStepSize],
    # workloads=workloads[querySkip:],
    benchmark_args = dict(
        scenarioName=scenarioName, optimizers=optimizer, batchSizes=batchSizes, no_warmup=no_warmup, translateOptSchemes=translateOptSchemes, output=output_file, noWorkloadResults=skipWorkloadResults,
        exp_args=exp_args
    )

    if queryStepSize != 0:
        for queryStep in range(querySkip, len(workloads), queryStepSize):
            print(f"Querying for {queryStep}:{queryStep+queryStepSize} workloads")
            sys.stdout.flush()
            res = benchmarkQuerying(oracle, workloads=workloads[queryStep:queryStep+queryStepSize], **benchmark_args)
    else:
        res = benchmarkQuerying(oracle, workloads=workloads[querySkip:], **benchmark_args)

    return res
import numpy as np
import pandas as pd
import psutil
import os
import sys
import dataclasses
from typing import List

from skypie.util.my_dataclasses import Workload, OracleType, Timer

def benchmarkQuerying(oracle: "Oracle", *, workloads: List[Workload], scenarioName: str, batchSizes, no_warmup, optimizers, translateOptSchemes, output, noWorkloadResults = False, exp_args=dict()):
    """
    Benchmark the querying performance of an Oracle for the given parameters.

    Args:
        oracle (Oracle): The Oracle object used for querying.
        workloads (List[Workload]): List of workloads to query.
        scenarioName (str): Name of the scenario to benchmark.
        batchSizes: List of batch sizes to benchmark.
        no_warmup: Flag indicating whether to skip the warm-up phase.
        optimizers: List of optimizers to benchmark.
        translateOptSchemes: Flag indicating whether to translate optimization schemes.
        output: Output file path to save the benchmark results.
        noWorkloadResults (bool, optional): Flag indicating whether to exclude result of each workload. Defaults to False.
        exp_args (dict, optional): Additional experimental arguments to add to the result output. Defaults to an empty dictionary.

    Returns:
        List[dict]: List of benchmark records containing the querying results.
    """
        
    # XXX: Assuming all workloads are scaled the same...
    rescale = workloads[0].rescale
    
    # Convert back to numpy array
    workloadsNumpy = np.array([w.equation for w in workloads])

    # Results: (optimizer, batchSize) -> (time, cost per workload)
    records = []
    for optimizer in optimizers:
        print(f"Loading optimizer {optimizer}. RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")
        oracle.prepare_scenario(scenarioName=scenarioName, optimizerType=optimizer)
        print(f"Loaded optimizer {optimizer}. RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

        if optimizer.implementation == OracleType.PYTORCH:
            workloadsQuery = workloadsNumpy
        else:
            workloadsQuery = workloads

        for batchSize in batchSizes:
            if batchSize > len(workloads):
                if oracle.verbose > 0:
                    print(f"Skipping batch size {batchSize} as it is larger than the number of workloads.")
                continue            

            print(f"Querying for {len(workloads)} workloads with batch size {batchSize} using optimizer {optimizer}")
            sys.stdout.flush()

            # Warmup
            if not no_warmup:
                print("Warming up...")
                numberOfWarmupQueries = 100 if not optimizer.implementation == OracleType.MOSEK else 5
                for _ in range(numberOfWarmupQueries):
                    oracle.query(workloadsQuery[:1], timer=None)

            # Query
            timer=Timer(verbose=oracle.verbose)
            resAll=list()
            timeTotalAll=list()
            timeComputeAll=list()
            for pos in range(0, len(workloads), batchSize):
                end = min(pos+batchSize, len(workloads))
                if oracle.verbose > 1:
                    print(f"Querying for {pos}:{end} workloads")
                timeTotal = timer.getTotalTime()
                timeCompute = timer.getComputationTime()
                res = oracle.query(workloadsQuery[pos:end], timer=timer, translateOptSchemes=translateOptSchemes, rescale=rescale)
                #print(f"Time for batch: {timer.getTotalTime() - timeTotal}")
                if not noWorkloadResults:
                    timeTotal = timer.getTotalTime() - timeTotal
                    timeCompute = timer.getComputationTime() - timeCompute
                    resAll.extend(res)
                    timeTotalAll.append(timeTotal)
                    timeComputeAll.append(timeCompute)
                else:
                    if len(resAll) <= 0:
                        resAll.append(list([0,0]))

                    resAll[0][0] += sum([optValue for optValue, _ in res])

            values = [optValue for optValue, _ in resAll]
            valueSum = sum(values)
            optSchemes = [optScheme if not dataclasses.is_dataclass(optScheme) else optScheme.asdict() for _, optScheme in resAll]
            if translateOptSchemes:
                optSchemes = [{"Workload": w, "OptScheme": s} for w, s in zip(workloads, optSchemes)]

            
            if oracle.verbose > 2 and optimizer.implementation == OracleType.PYTORCH:
                valuesAssert = oracle.compute_optimal_value(optimalSchemeIndex=optSchemes, optimizerType = None, workload=workloads)
                if not np.allclose(values, valuesAssert):
                    for v, a in zip(values, valuesAssert):
                        if not np.isclose(v, a):
                            print(f"Error: {v} != {a}")

            # ["Optimizer", "BatchSize", "Time Computation", "Time Total", "Cost per workload"]
            record = {"Batch Size": batchSize, "Time Computation (ns)": timer.getComputationTime(), "Time Total (ns)": timer.getTotalTime(), "Time Computation per workload (ns)": timeComputeAll, "Time Total per workload (ns)": timeTotalAll, "Optimal value per workload": values, "Optimal scheme per workload": optSchemes, "Optimal value": valueSum}
            record.update(optimizer.get_parameters())
            record.update(oracle.get_precomputation_times(optimizerType=optimizer))
            record.update( {f"exp_{k}":v for k,v in exp_args.items()} )
            records.append(record)

            if optimizer.implementation == OracleType.MOSEK:
                print("Mosek does not support batching queries at the moment. Skipping the rest of the batch sizes...")
                break

        # Save results after finishing each optimizer
        resultDf = pd.DataFrame(records)
        resultDf["Number of Workloads"] = len(workloads)
        
        if output is not None:
            if output.endswith(".pickle"):
                resultDf.to_pickle(output)
            else:
                resultDf.to_json(output)
        else:
            print(resultDf)

    return records
import skypie_proto_messages as m
import os
import json
import psutil

from skypie.util.my_dataclasses import OptimizerType, OracleType, MosekOptimizerType

def nth_key(data, n: int):
    return list(data.keys())[n]

def get_key(data, key: str):
    if key.startswith("relative"):
        # Get n-th item
        pos = int(key.split("=")[1])
        key = list(data.keys())[pos]
    return data[key]

def is_decision(decision):
    return isinstance(decision, m.Decision)

def get_precomputation_times(*, scenario, optimizerType):
    res = dict()

    # Enumeraiton time of scenario
    res["Enumeration Time (ns)"] = scenario.enumerator_time_ns

    # Time for computing the optimal repliction schemes/partitions with current optimizer
    if optimizerType.name in scenario.optimal_partitions_by_optimizer:
        optimizer_stats  = scenario.optimal_partitions_by_optimizer[optimizerType.name]
        res["Partition Time (ns)"] = optimizer_stats.partitioner_time_ns
        res["Partition Time - Compute Only (ns)"] = optimizer_stats.partitioner_computation_time_ns
        #if "partitioner_wall_time_ns" in optimizer_stats:
        #    res["Partition Wall Time (ns)"] = optimizer_stats["partitioner_wall_time_ns"]
    else:
        res["Partition Time (ns)"] = scenario.partitioner_time_ns
        res["Partition Time - Compute Only (ns)"] = res["Partition Time (ns)"]
        #if "partitioner_wall_time_ns" in scenario:
        #    res["Partition Wall Time (ns)"] = scenario["partitioner_wall_time_ns"]

    return res

def load_cost_matrix(decisions):
    return [d.cost_wl_halfplane[:-1] for d in decisions]

def load_proto(file_name: str):
    # Load wrapper
    return m.load_wrapper(file_name)

def select_scenario(scenario_data, scenario_path: str):
    scenario = None
    paths = scenario_path.split("/")

    if paths[0] == "tier_advise" and paths[1] == "replication_factor":
        tier_adivse_data = scenario_data.tier_advise
        
        replication_factor_data = get_key(tier_adivse_data.replication_factor, paths[2])
        run = get_key(replication_factor_data.runs, paths[3])
        
        scenario = run
    else:
        raise Exception(f"Invalid scenario path: {scenario_path}")
    
    return scenario

def load_candidates(*, scenario, path, threads=1, compact=True):
    if len(scenario.candidate_partitions) <= 0:
        print("WARN: No candidates to load")
        return scenario.candidate_partitions
    
    print(f"Loading candidates from {scenario.candidate_partitions}")
    
    candidates = m.load_decisions_parallel([os.path.join(path, f) for f in scenario.candidate_partitions], threads, compact)

    return candidates

def load_cost_matrix_from_files(*, scenario, optimizer_name: str, path: str, optimal_partition_files = None, threads=1):
    if not optimizer_name in scenario.optimal_partitions_by_optimizer:
        return {}
    
    if optimal_partition_files is None:
        optimal_partition_files = scenario.optimal_partitions_by_optimizer[optimizer_name].optimal_partitions
    #costs = m.load_decision_costs_parallel([os.path.join(path, f) for f in optimal_partition_files], threads)
    #costs = m.load_decision_costs_numpy([os.path.join(path, f) for f in optimal_partition_files])
    costs = m.load_decision_costs_numpy_parallel([os.path.join(path, f) for f in optimal_partition_files])

    print(f"Cost matrix dimensions: {costs.shape}")

    return costs

def load_optimal_partitions(*, scenario, optimizer_name: str, path: str, optimal_partition_files = None, threads=1, compact = True):
    if not optimizer_name in scenario.optimal_partitions_by_optimizer:
        return {}
    
    if optimal_partition_files is None:
        optimal_partition_files = scenario.optimal_partitions_by_optimizer[optimizer_name].optimal_partitions

    
    if compact:
        optimal_partition_files_compact = optimal_partition_files[0:1]
    else:
        optimal_partition_files_compact = optimal_partition_files
    print(f"Loading optimal partitions from {optimal_partition_files_compact}")
    replicationSchemes = m.load_decisions_parallel([os.path.join(path, f) for f in optimal_partition_files_compact], max(1,threads), compact)
    print(f"Loaded {len(replicationSchemes)} optimal partitions, RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

    # Load mapping of application regions to indexes from optimal partitions
    if len(replicationSchemes[0].replication_scheme.app_assignments) > 0:
        applicationRegions = { assignment.app:index for index, assignment in enumerate(replicationSchemes[0].replication_scheme.app_assignments) }

    if compact:
        print(f"Loading cost matrix from {optimal_partition_files}")
        replicationSchemes = load_cost_matrix_from_files(scenario=scenario, optimizer_name=optimizer_name, path=path, optimal_partition_files=optimal_partition_files, threads=threads)
        print(f"Loaded cost matrix, RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

    return {"optimal_partitions": replicationSchemes, "application_regions": applicationRegions}

def load_scenario(*, loaded_file, scenario_path: str, path: str, do_load_candidates: bool = True, threads=1, compact: bool = True):
    scenario = select_scenario(loaded_file, scenario_path)

    candidates = []
    if do_load_candidates:
        candidates = load_candidates(scenario=scenario, path=path, threads=threads, compact=compact)

    object_stores_considered = scenario.object_stores_considered

    if len(candidates) > 0:
        application_regions = {a.app: i for i, a in enumerate(candidates[0].replication_scheme.app_assignments)}
    else:
        application_regions = dict()

    no_apps = scenario.no_app_regions

    return {"scenario": scenario, "candidates": candidates, "object_stores_considered": object_stores_considered, "application_regions": application_regions, "no_apps": no_apps}

def discoverOptimizerTypes(*, loaded_file, scenario_path: str) -> "List[OptimizerType]":
        scenario = select_scenario(loaded_file, scenario_path)
        precomputationOptimizers = list()
        
        print(scenario.optimal_partitions_by_optimizer.keys())
        for optimizer in scenario.optimal_partitions_by_optimizer.keys():
            # Lrs does not provide details about the optimizer type, so we have to create one
            if "Lrs" in optimizer:
                precomputationOptimizers.append(OptimizerType(implementation=OracleType.PYTORCH, type=optimizer))
            else:
                # Load optimizer type from precomputed data
                optimizerData = scenario.optimal_partitions_by_optimizer[optimizer].optimizer_type
                
                if isinstance(optimizerData, str) and "type" in optimizerData and "implementation" in optimizerData:
                    optimizerData = json.loads(optimizerData)
                    
                if isinstance(optimizerData, dict):
                    optimizerData["type"] = MosekOptimizerType.from_str(optimizerData["type"])

                    implementation = optimizerData["implementation"]
                    implementation = {
                        1: OracleType.PYTORCH.value
                    }.get(implementation, implementation)
                    optimizerData["implementation"] = OracleType(implementation)
                    optimizerData = OptimizerType(**optimizerData)
                    #scenario.optimal_partitions_by_optimizer[optimizer].optimizer_type = optimizerData
                    
                precomputationOptimizers.append(optimizerData)

        return precomputationOptimizers

if __name__ == "__main__":
    file_name = "/home/vscode/sky-pie-precomputer/experiments/experiment-2023-08-25-16-48-02/stats.bin"
    scenario_path = "tier_advise/replication_factor/relative=0/relative=0"
    data = load_proto(file_name)
    optimizers = discoverOptimizerTypes(loaded_file=data, scenario_path=scenario_path)
    res = load_scenario(loaded_file=data, scenario_path=scenario_path, path=os.path.dirname(file_name))
    scenario = res["scenario"]
    candidates = res["candidates"]

    num_redundancy_workers = 60
    scenario.candidate_partitions = [f"candidates_{i}.proto.bin" for i in range(num_redundancy_workers)]
    # Path of stats file
    file_path = os.path.dirname(file_name)
    candidates = load_candidates(scenario=scenario, path=file_path, compact=True)

    # Load optimal partitions
    optimizer_name = "MosekOptimizerType.InteriorPoint_Clarkson_iter0_dsize1000"
    # Injecting optimal partitions
    optimal_partition_files = [f"optimal_{i}.proto.bin" for i in range(num_redundancy_workers)]
    res = load_optimal_partitions(scenario=scenario, optimizer_name=optimizer_name, path=file_path, optimal_partition_files=optimal_partition_files)
    print(scenario)

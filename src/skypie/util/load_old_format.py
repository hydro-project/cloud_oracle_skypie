import json
import os
from typing import List

from skypie.util.util import deltaDecode
from skypie.util.my_dataclasses import ReplicationScheme, OptimizerType, OracleType, MosekOptimizerType

def get_precomputation_times(*, scenario, optimizerType):
    res = dict()

    # Enumeraiton time of scenario
    res["Enumeration Time (ns)"] = scenario["enumerator_time_ns"]

    # Time for computing the optimal repliction schemes/partitions with current optimizer
    if "optimal_partitions_by_optimizer" in scenario and optimizerType.name in scenario["optimal_partitions_by_optimizer"]:
        res["Partition Time (ns)"] = scenario["optimal_partitions_by_optimizer"][optimizerType.name]["partitioner_time_ns"]
        res["Partition Time - Compute Only (ns)"] = scenario["optimal_partitions_by_optimizer"][optimizerType.name]["partitioner_computation_time_ns"]
        if "partitioner_wall_time_ns" in scenario["optimal_partitions_by_optimizer"][optimizerType.name]:
            res["Partition Wall Time (ns)"] = scenario["optimal_partitions_by_optimizer"][optimizerType.name]["partitioner_wall_time_ns"]
    else:
        res["Partition Time (ns)"] = scenario["partitioner_time_ns"]
        res["Partition Time - Compute Only (ns)"] = res["Partition Time (ns)"]
        if "partitioner_wall_time_ns" in scenario:
            res["Partition Wall Time (ns)"] = scenario["partitioner_wall_time_ns"]

    res["Partition Time (ns)"] = max(0, int(res["Partition Time (ns)"]))
    res["Partition Time - Compute Only (ns)"] = max(0, int(res["Partition Time - Compute Only (ns)"]))

    if not "Partition Wall Time (ns)" in res:
        res["Partition Wall Time (ns)"] = res["Partition Time (ns)"]

    return res

def load_scenario(*, loaded_file, scenario_path: str, do_load_candidates: bool = True):
    scenario = loaded_file
    for key in scenario_path.split("/"):
        if "relative" in key:
            pos = int(key.split("=")[1])
            key = list(scenario.keys())[pos]
        scenario = scenario[key]
    if "candidate_partitions" in scenario and len(scenario["candidate_partitions"]) > 0 and do_load_candidates:
        candidateSchemes = [ ReplicationScheme(p) for p in scenario["candidate_partitions"]]
        applicationRegions = candidateSchemes[0].applictionRegionMapping
    else:
        candidateSchemes = []
        applicationRegions = []

    if "object_stores_considered" in scenario:
        object_stores_considered = scenario["object_stores_considered"]
    else:
        object_stores_considered = None

    no_apps = int(scenario["no_app_regions"])

    return {"scenario": scenario, "candidates": candidateSchemes, "object_stores_considered": object_stores_considered, "application_regions": applicationRegions, "no_apps": no_apps}

def load_optimal_partitions(*, scenario, optimizer_name: str, path: str):
    replicationSchemes = []

    if "optimal_partitions_by_optimizer" in scenario and \
        optimizer_name in scenario["optimal_partitions_by_optimizer"] and \
        ("optimal_partition_ids" in scenario["optimal_partitions_by_optimizer"][optimizer_name] and len(scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partition_ids"]) > 0 or \
        "optimal_partition_ids_delta" in scenario["optimal_partitions_by_optimizer"][optimizer_name] and len(scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partition_ids_delta"]) > 0 or \
        "optimal_partitions" in scenario["optimal_partitions_by_optimizer"][optimizer_name] and len(scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partitions"]) > 0 
        ):

        if "optimal_partitions" in scenario["optimal_partitions_by_optimizer"][optimizer_name]:

            temp = scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partitions"]
            # If temp is a string, then this is a file to be loaded.
            if isinstance(temp, str) and temp.endswith(".jsonl"):
                # Reuse loading code from below
                temp = [temp]
            # If temp is a list of strings, then this is a list of files to be loaded
            if isinstance(temp, list) and len(temp) > 0 and isinstance(temp[0], str) and temp[0].endswith(".jsonl"):
                # Load optimal partitions from referenced file, line by line
                # Assuming path relative to input file!
                temp_dir = path
                for file in temp:
                    print(f"Loading external policies: {file}")
                    temp_full_path = os.path.join(temp_dir, file)
                    with open(temp_full_path, "r") as f:
                        replicationSchemes = [ReplicationScheme(json.loads(line)) for line in f]
            else:
                # Load optimal partitions
                replicationSchemes = [ ReplicationScheme(s) for s in temp]
        else:
            if "optimal_partition_ids_delta" in scenario["optimal_partitions_by_optimizer"][optimizer_name]:
                # Decompress delta encoded optimal partition ids
                optimal_partition_ids = deltaDecode(scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partition_ids_delta"])
            else:
                # Take raw optimal partition ids
                optimal_partition_ids = scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partition_ids"]
            
            """ # Load incidence information of this optimizer
            if incidenceOptimizerType and "incidence_by_optimizer" in scenario["optimal_partitions_by_optimizer"][optimizer_name]["incidence_by_optimizer"]:
            #scenario["optimal_partitions_by_optimizer"][self.optimizer_name]["incidence_by_optimizer"][name]
                incidence = scenario["optimal_partitions_by_optimizer"][optimizer_name]["incidence_by_optimizer"][incidenceoptimizer_name]['incident_partitions_id']
                incidenceOptimizerType = incidenceOptimizerType
            else:
                incidence = dict()
                incidenceOptimizerType = None """


            # Restore optimal partitions from candidate partitions
            replicationSchemes = [ ReplicationScheme(scenario["candidate_partitions"][pID]) for pID in optimal_partition_ids]
            print("Restored optimal partitions by id")

    #elif optimizerType.type != "None" and optimizerType.type != "ILP" and "optimal_partitions_by_optimizer" in scenario and optimizer_name in scenario["optimal_partitions_by_optimizer"] and len(scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partitions"]) > 0:
    #    self.__replicationSchemes = [ ReplicationScheme(p) for p in scenario["optimal_partitions_by_optimizer"][optimizer_name]["optimal_partitions"]]
    #elif "optimal_partitions" in scenario and len(scenario["optimal_partitions"]) > 0:
    #    self.__replicationSchemes = [ ReplicationScheme(p) for p in scenario["optimal_partitions"]]
    #    print("Falling back to default optimal partitions by unknown optimizer.")
    else:
        raise Exception(f"No optimal partitions found for optimizer {optimizer_name}.")

    # Load application regions from optimal partitions
    if len(replicationSchemes[0].assignments) > 1:
        applicationRegions = { app:index for index, app in enumerate(replicationSchemes[0].assignments) }

    # Load mapping of application regions to indexes
    if len(applicationRegions) == 0 and "applictionRegionMapping" in scenario:
        applicationRegions = scenario["applictionRegionMapping"]

    return {"optimal_partitions": replicationSchemes, "application_regions": applicationRegions}

def discoverOptimizerTypes(*, loaded_file, scenario_path: str) -> "List[OptimizerType]":
        scenario = loaded_file
        precomputationOptimizers = list()
        for key in scenario_path.split("/"):
            if "relative" in key:
                pos = int(key.split("=")[1])
                key = list(scenario.keys())[pos]
            scenario = scenario[key]
        
        if "optimal_partitions_by_optimizer" in scenario:
            print(scenario["optimal_partitions_by_optimizer"].keys())
            for optimizer in scenario["optimal_partitions_by_optimizer"]:
                # Lrs does not provide details about the optimizer type, so we have to create one
                if "Lrs" in optimizer:
                    precomputationOptimizers.append(OptimizerType(implementation=OracleType.PYTORCH, type=optimizer))
                else:
                    # Load optimizer type from precomputed data
                    optimizerData = scenario["optimal_partitions_by_optimizer"][optimizer]["optimizer_type"]
                    
                    if isinstance(optimizerData, str) and "type" in optimizerData and "implementation" in optimizerData:
                        optimizerData = json.loads(optimizerData)
                        
                    if isinstance(optimizerData, dict):
                        optimizerData["type"] = MosekOptimizerType.from_str(optimizerData["type"])
                        optimizerData["implementation"] = OracleType(optimizerData["implementation"])
                        optimizerData = scenario["optimal_partitions_by_optimizer"][optimizer]["optimizer_type"] = OptimizerType(**optimizerData)
                        
                    precomputationOptimizers.append(optimizerData)

        return precomputationOptimizers
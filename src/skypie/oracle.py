from dataclasses import dataclass, field
from typing import *
import json
import numpy as np
import pandas as pd
import bz2
import pickle
import os
import psutil

from skypie.util.my_dataclasses import *
from skypie.oracle_impls.oracle_ilp import OracleImplMosek
from skypie.oracle_impls.oracle_pytorch import OracleImplPyTorch
from skypie.oracle_impls.oracle_kmeans import OracleImplKmeans
from skypie.oracle_impls.oracle_profit_based import OracleImplProfit
from skypie.util.load_old_format import load_scenario as load_scenario_old_format
from skypie.util.load_old_format import load_optimal_partitions as load_optimal_partitions_old_format
from skypie.util.load_old_format import discoverOptimizerTypes as discoverOptimizerTypes_old_format
from skypie.util.load_old_format import get_precomputation_times as get_precomputation_times_old_format
from skypie.util.load_proto import load_scenario as load_scenario_proto
from skypie.util.load_proto import load_optimal_partitions as load_optimal_partitions_proto
from skypie.util.load_proto import discoverOptimizerTypes as discoverOptimizerTypes_proto
from skypie.util.load_proto import load_proto
from skypie.util.load_proto import get_precomputation_times as get_precomputation_times_proto

from skypie.util.util import *
from skypie.oracle_impls.oracle_interface import OracleInterface

@dataclass
class Oracle:
    AVAILABLE_ORACLE_IMPLEMENTATIONS: ClassVar[Dict[OracleType, OracleInterface]] = {
        OracleType.SKYPIE: OracleImplPyTorch,
        OracleType.PYTORCH: OracleImplPyTorch,
        OracleType.MOSEK: OracleImplMosek,
        OracleType.KMEANS: OracleImplKmeans,
        OracleType.PROFIT: OracleImplProfit
    }
    
    inputFileName: str
    scenarioName: str = field(init=False, default=None)
    verbose: int = -1
    optimizerType: "OptimizerType" = field(init=False, default=None) #= field(default=OptimizerType(type="None"))
    no_apps = -1
    __instance: OracleInterface = field(init=False, default=None)
    __precomputedData: Dict[str, Any] = field(init=False, default=None)
    __scenario: Any = field(init=False, default=None)
    __replicationSchemes: List[ReplicationScheme] = field(init=False, default_factory=list)
    # Name of the appliction region and its index for the workload
    __applicationRegions: Dict[str, int] = field(init=False, default_factory=list)
    __objectStoresConsidered: List[str] = field(init=False, default_factory=list)
    __precomputationOptimizers: List[OptimizerType] = field(init=False, default_factory=list)
    __old_format: bool = field(init=False, default=False)

    __baseline_optimizer_types: ClassVar[List[OracleType]] = [OracleType.MOSEK, OracleType.KMEANS, OracleType.PROFIT]
    __default_scenario: ClassVar[str] = "tier_advise/replication_factor/relative=0/relative=0"


    def __post_init__(self):
        self.__old_format = not "proto" in self.inputFileName.split(".")

        if self.__old_format:
            readMode = "rb" if "pickle" in self.inputFileName else "r"
            if self.inputFileName.endswith("bz2"):
                f = bz2.BZ2File(self.inputFileName, readMode)
            else:
                f = open(self.inputFileName, readMode)
            if "pickle" in self.inputFileName:
                self.__precomputedData = pickle.load(f)
            else:
                self.__precomputedData = json.load(f, cls=EnhancedJSONDecoder)
            f.close()
        else:
            self.__precomputedData = load_proto(self.inputFileName)
        
        self.optimizerType = None
        self.__scenario = None
        self.__candidateSchemes = None
        self.__applicationRegions = dict()
        self.__objectStoresConsidered = list()

        #if self.scenarioName:
        #   self.prepare_scenario(scenarioName=self.scenarioName, optimizerType=self.optimizerType)

    def __hasOptimalPartitions(self) -> bool:
        return len(self.__replicationSchemes) > 0

    def __discoverOptimizerTypes(self, scenarioName: str) -> "List[OptimizerType]":
        """
        Discover optimizer types from the loaded precomputed data and the given scenario name.
        """
        
        if self.__old_format:
            return discoverOptimizerTypes_old_format(loaded_file=self.__precomputedData, scenario_path=scenarioName)
        else:
            return discoverOptimizerTypes_proto(loaded_file=self.__precomputedData, scenario_path=scenarioName)

    def prepare_scenario(self, scenario_name: str, optimizer_type: "OptimizerType", repair_input_data: bool = False, skip_instance = False, compact: bool = True):
        """
        This function prepares the oracle for a specific scenario and optimizer from the input file.

        # Arguments
        - scenario_name: The name of the scenario to be used when several oracles have been precomputed in the same input file, or "default" to use the default scenario.
        - optimizer_type: The type of optimizer to be used for querying the oracle. Must be one of the values defined in the OracleType enum and must be supported by the loaded precomputed data.
        - repair_input_data: Whether to repair the input data by removing invalid schemes. Defaults to False.
        - skip_instance: Whether to skip the instantiation of the oracle implementation. Defaults to False.
        - compact: Whether to use the compact representation of the precomputed data. Defaults to True. The compact representation eases benchmarking but is insufficient for regular querying.

        # Notes
        FIXME: Assignments are not correctly parsed!
        """

        if scenario_name == "default":
            scenario_name = self.__default_scenario

        # Instantiate OracleImplementation if implementation or implementationArgs changed
        if not skip_instance and ((self.optimizerType is None) or (optimizer_type.implementation != self.optimizerType.implementation or self.optimizerType.implementationArgs != optimizer_type.implementationArgs)):
            print("Switching OracleImplementation: ", optimizer_type.implementation)

            if optimizer_type.implementation in self.AVAILABLE_ORACLE_IMPLEMENTATIONS:
                self.__instance = self.AVAILABLE_ORACLE_IMPLEMENTATIONS[optimizer_type.implementation](**optimizer_type.implementationArgs)
            else:
                raise Exception(f"Unknown OracleType: {optimizer_type.implementation}")

            self.optimizerType = optimizer_type

        do_load_candidates = optimizer_type.type == "Candidates"

        if self.scenarioName != scenario_name or not self.__candidateSchemes and do_load_candidates:
            self.scenarioName = scenario_name
            self.__scenario = self.__precomputedData
            
            if self.__old_format:
                res = load_scenario_old_format(loaded_file=self.__precomputedData, scenario_path=scenario_name, do_load_candidates=do_load_candidates)
            else:
                res = load_scenario_proto(loaded_file=self.__precomputedData, scenario_path=scenario_name, path=os.path.dirname(self.inputFileName), do_load_candidates=do_load_candidates, threads=optimizer_type.implementationArgs.get("threads", 10), compact=compact)

            self.__scenario = res["scenario"]
            self.__candidateSchemes = res["candidates"]
            self.__objectStoresConsidered = res["object_stores_considered"]
            self.__applicationRegions = res["application_regions"]
            
            self.no_apps = res["no_apps"]
            assert self.no_apps == len(self.__applicationRegions) or len(self.__applicationRegions) == 0, f"Number of application regions in scenario {self.no_apps} does not match number of application regions in optimal partitions {len(self.__applicationRegions)}"

            # Discover optimizer types for this scenario
            self.__precomputationOptimizers = self.__discoverOptimizerTypes(scenario_name)

        # Load optimal partitions
        if optimizer_type.type == "Candidates":
            # Use candidate partitions to query optima. The result is still optimal, but the computation takes longer.
            if len(self.__candidateSchemes) > 0:
                self.__replicationSchemes = self.__candidateSchemes
            else:
                raise Exception("No candidate partitions found in scenario")
        elif optimizer_type.type == "ILP" or optimizer_type.type == OracleType.KMEANS.value or optimizer_type.type == OracleType.PROFIT.value:
            pass
        else:
            if self.verbose > 0:
                print(f"Loading optimal partitions for optimizer {optimizer_type.name}...")
                print(f"RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")
            if self.__old_format:
                res = load_optimal_partitions_old_format(scenario=self.__scenario, optimizer_name=optimizer_type.name, path=os.path.dirname(self.inputFileName))
            else:
                res = load_optimal_partitions_proto(scenario=self.__scenario, optimizer_name=optimizer_type.name, path=os.path.dirname(self.inputFileName), threads=optimizer_type.implementationArgs.get("threads", 10), compact=compact)

            if len(res) > 0:
                self.__replicationSchemes = res["optimal_partitions"]
                self.__applicationRegions = res["application_regions"]
            else:
                print(f"WARN: No optimal partitions found for optimizer {optimizer_type.name} in scenario {scenario_name}")

            if self.verbose > 0:
                print(f"Loaded optimal partitions for optimizer {optimizer_type.name}.")
                print(f"RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

        if repair_input_data:
            self.repairInputData()

        # Mosek does not need the precomputed optimal partitions
        if not skip_instance and (self.__hasOptimalPartitions() or optimizer_type.implementation in self.__baseline_optimizer_types):
            self.__instance.prepare_schemes(oracle=self)
        else: 
            print(f"WARNING: No optimal partitions found in scenario for optimizer {optimizer_type}. The optimal partitons have to be computed from the candidate partitions or the candidate partitions can be used directly.")


    def query_directed_drift(self, w: "Workload", drift: "Workload", timer: Timer = None, translateOptSchemes: bool = False, rescale: np.float64 = np.float64(1)) -> "Tuple[float, List[float], int]":
        """
        Queries the oracle for the next optimal placement decision when the given workload drifts in the given drift direction.

        # Arguments:
        - w: The current workload.
        - drift: The drift of the workload, by how much each workload parameter changes per unit of time.
        - timer: The timer to use for measuring the overhead and computation time, by default None and no timing is performed.
        - translateOptSchemes: Whether to return the full placement decision rather than just their ID. Defaults to False.
        - rescale: A scaling factor to apply to the query costs. Defaults to 1.

        # Returns:
        A tuple containing the distance, the point (workload+cost) of the next optimal decision, and the next optimal decision (translated or ID).

        # Raises:
        - NotImplementedError: If the oracle implementation does not support directed drift queries, since drift queries are optionally supported.

        # Note:
        It does not support batching.
        """
        
        distance, point, index = self.__instance.query_directed_drift(w=w, drift=drift, timer=timer)

        if translateOptSchemes:
            index = self.__replicationSchemes[index] if isinstance(index, (int, np.number)) and index < len(self.__replicationSchemes) else index

        # Rescale costs for scaled workloads
        if rescale != 1.0:
            distance = distance/rescale
            point[-1] = point[-1]/rescale

        return distance, point, index

    def query(self, w: "Workload|List[Workload]", timer: Timer = None, translateOptSchemes: bool = False, rescale: np.float64 = np.float64(1)) -> "List[Tuple[float,int|ReplicationScheme]]":
        """
        Queries the oracle for the optimal placement decision for the given workload(s).

        Args:
            w: A single workload or a list of workloads to query the oracle for.
            timer: A timer object to measure the query time. Defaults to None.
            translateOptSchemes: Whether to return the full placement decision(s) rather than just their ID(s). Defaults to False.
            rescale: A scaling factor to apply to the query costs. Defaults to 1.

        Returns:
            A list of tuples, where each tuple contains the query cost and the optimal placement decision for a given workload.
            If `translateOptSchemes` is True, the detailed placement decisions are returned instead of their IDs.
        """

        res = self.__instance.query(w, timer=timer)

        if translateOptSchemes:

            if self.optimizerType.implementation == OracleType.SKYPIE or self.optimizerType.implementation == OracleType.CANDIDATES:
                res = [ (r[0], self.__replicationSchemes[r[1]] if isinstance(r[1], (int, np.number)) and r[1] < len(self.__replicationSchemes) else r[1]) for r in res ]

        # Rescale costs for scaled workloads
        if rescale != 1.0:
            res = [ (r[0]/rescale,r[1]) for r in res ]

        return res

    def get_schemes(self) -> List[ReplicationScheme]:
        return self.__replicationSchemes
    
    def get_instance(self) -> OracleInterface:
        return self.__instance

    def get_candidate_schemes(self) -> List[ReplicationScheme]:
        return self.__candidateSchemes

    def get_application_regions(self) -> Dict[str, int]:
        return self.__applicationRegions

    def get_object_stores_considered(self) -> bool:
        return self.__objectStoresConsidered

    def get_scenario(self) -> Dict:
        return self.__scenario

    def get_min_replication_factor(self) -> int:
        if self.__old_format:
            return int(self.__scenario["min_replication_factor"])
        else:
            return int(self.__scenario.min_replication_factor)
    
    def get_max_replication_factor(self) -> int:
        if self.__old_format:
            return int(self.__scenario["max_replication_factor"])
        else:
            return int(self.__scenario.max_replication_factor)
    
    def get_precomputation_optimizers(self) -> List[OptimizerType]:
        return self.__precomputationOptimizers

    def save(self, output: str):
        """
        Saves the precomputed data to a file, in JSON or binary pickle format depending on the provided file ending.
        """

        if output is None or output == "":
            json.dumps(self.__precomputedData, cls=EnhancedJSONEncoder, indent=4)
        else:
            print("Writing out to " + output)
            if output.endswith(".pickle"):
                with open(output, "wb") as f:
                    pickle.dump(self.__precomputedData, f)
            else:
                with open(output, "w") as f:
                    json.dump(self.__precomputedData, f, cls=EnhancedJSONEncoder, indent=4)

        # restore the optimal partitions of all optimizers

    def get_precomputation_times(self,*, optimizerType: "OptimizerType|None" = None):

        if optimizerType is None:
            optimizerType = self.optimizerType

        if self.__old_format:
            res = get_precomputation_times_old_format(scenario=self.__scenario, optimizerType=optimizerType)
        else:
            res = get_precomputation_times_proto(scenario=self.__scenario, optimizerType=optimizerType)

        res["Partition Time (ns)"] = max(0, int(res["Partition Time (ns)"]))
        res["Partition Time - Compute Only (ns)"] = max(0, int(res["Partition Time - Compute Only (ns)"]))

        if not "Partition Wall Time (ns)" in res:
            res["Partition Wall Time (ns)"] = res["Partition Time (ns)"]

        return res
        
    def create_workload(self, *, size: np.float64, put: "np.ndarray", get: "np.ndarray", ingress: "np.ndarray|None" = None, egress: "np.ndarray|None" = None, rescale: np.float64 = np.float64(1.0)) -> Workload:
        """
        Creates a new workload object assuming that the region to index mapping was done correctly.
        See create_workload_by_region_name for a more convenient way to create workloads.
        """
        # XXX: Instantiate workload according to application region order in schemes, i.e., must use their correct index of get/ingress/egress.
        if put.shape[0] != self.no_apps:
            raise ValueError("Number of puts does not match number of applications")
        if get.shape[0] != self.no_apps:
            raise ValueError("Number of gets does not match number of applications")

        return Workload(size=size, put=put, get=get, ingress=ingress, egress=egress, rescale=rescale)
    
    def create_workload_by_region_name(self, *, size: np.float64, put: Dict[str,float], get: Dict[str,float], ingress: "Dict[str,float]|None" = None, egress: "Dict[str,float]|None" = None, rescale: np.float64 = np.float64(1.0)) -> Workload:
        """
        Creates a new workload object based on the specified workload parameters of the named application cloud regions.

        # Arguments
            - size (np.float64): The object size of the workload.
            - put (Dict[str,float]): A dictionary mapping region names to the number of PUT requests for each region.
            - get (Dict[str,float]): A dictionary mapping region names to the number of GET requests for each region.
            - ingress (Dict[str,float]|None): A dictionary mapping the number of ingress __bytes send from applications to object stores__ per cloud region, or None to automatically derive from size and puts.
            - egress (Dict[str,float]|None): A dictionary mapping the number of egress __bytes send from object stores to applications__ per cloud region, or None to automatically derive from size and gets.
            - rescale (np.float64): A scaling factor for the workload (default is 1.0).

        # Returns:
            Workload: A new workload object based on the specified parameters.

        # Raises:
            ValueError: If a region specified in the PUT, GET, ingress, or egress dictionaries is not found in the application regions.

        # Notes:
        - Be sure to set ingress and egress appropriately for the specified put/get values for each region. Missing or incorrect network traffic leads to wrong results.
        """

        def translate(items, type):
            target = np.zeros(self.no_apps)
            for region, count in items.items():
                if region in self.__applicationRegions:
                    target[self.__applicationRegions[region]] = count
                else:
                    raise ValueError(f"Region {region} of workload {type} spec. not found in application regions!\n" + "Application regions: " + str(self.__applicationRegions))
            return target

        put_translated = translate(put, "put")
        get_translated = translate(get, "get")
        ingress_translated = translate(ingress, "ingress") if ingress is not None else None
        egress_translated = translate(egress, "egress") if egress is not None else None

        return Workload(size=size, put=put_translated, get=get_translated, ingress=ingress_translated, egress=egress_translated, rescale=rescale)

    @staticmethod
    def get_workload_file_format() -> str:
        return "The format of a workload file is: [\n{'size': <object size>,\n'put':{<region name 1>: <puts of region 1>, <region name 2>: <puts of region 2>, ...},\nget:{<region name 1>: <gets of region 1>,...}, ingress:{<region name 1>: <ingress volume of region 1>, ...}, egress: {<region name 1>: <egress volume of region 1>}},\n{...},...] "
    
    def __compute_optimal_value(self, *, optimizerType: "OptimizerType|None" = None, optimalSchemeIndex: "int|List[int]", workload: "Workload|List[Workload]") -> float:

        if isinstance(optimalSchemeIndex, int):
            optimalSchemeIndex = [optimalSchemeIndex]

        if isinstance(workload, Workload):
            workload = [workload]

        if optimizerType is not None and optimizerType.type != "None":
            print("Changing optimizer type to " + optimizerType.name)
            self.prepare_scenario(scenario_name=self.scenarioName, optimizer_type=optimizerType)

        # Compute optimal value
        if optimizerType is None or self.optimizerType.implementation != OracleType.MOSEK:
            res = [self.__replicationSchemes[index].compute_cost(w) for index, w in zip(optimalSchemeIndex, workload)]
        else:
            # TODO: Use ILP to compute optimal value
            raise NotImplementedError("TODO")

        if self.verbose > 2:
            print("Optimal value:", res)

        return res

    def statistics(self, *, optimizers, output, scenarioName):
        """
        Extract statistics from the precomputed data, either printing them to stdout or saving them to a file.
        """
        records = []
        for optimizer in optimizers:
            self.prepare_scenario(scenario_name=scenarioName, optimizer_type=optimizer, skip_instance=True)

            record = {
                "no_schemes": len(self.get_schemes()),
                "no_candidates": len(self.get_candidate_schemes()),
                } | optimizer.get_parameters() | self.get_precomputation_times()
            
            records.append(record)

        # Save results after finishing each optimizer
        resultDf = pd.DataFrame(records)
        
        if output is not None:
            if output.endswith(".pickle"):
                resultDf.to_pickle(output)
            else:
                resultDf.to_json(output)
            print(f"Saved statistics to {output}")
        else:
            print(resultDf)

    @staticmethod
    def setup_oracle(*, inputFileName: str, scenarioName: str, addOptimizersFromInput: bool = False, optimizer: List[OracleType] = list(), implArgs=dict(), verbose=0, skip_loaded_optimizers=True):
        """
        Helper function to set up a new oracle instance.

        # Arguments:
        - inputFileName: The name of the input file containing the precomputed oracle data.
        - scenarioName: The name of the scenario to be used when several oracles have been precomputed in the same input file, or "default" to use the default scenario.
        - addOptimizersFromInput: Whether to add optimizers from the precomputed data to the optimizer list. Defaults to False.
        - optimizer: A list of optimizers to be used for querying the oracle. Can be a list of elements from AVAILABLE_ORACLE_IMPLEMENTATIONS, defaults to an empty list then addOptimizersFromInput has to be set True.
        - implArgs: A dictionary of implementation-specific arguments to be passed to the oracle implementations of the optimizers. Defaults to an empty dictionary. An empty dictionary means that the default values of the implementation will be used. The default arguments can be retrieved using the get_default_oracle_impl_args function, then be modified and passed to this function.
        - verbose: The level of verbosity for logging of the oracle. 0 means no logging, 1 means basic logging, and 2 means detailed logging. Defaults to 0.
        - skip_loaded_optimizers: Whether to skip optimizers that have been loaded from the precomputed data. Defaults to True. It means that the optimizers from the input file are used for loading but are not used for the actual querying.

        # Returns:
        A tuple containing the oracle instance and the list of optimizers to be used for querying the oracle.
        """

        oracle = Oracle(inputFileName=inputFileName, verbose=verbose)

        if scenarioName == "default":
            scenarioName = Oracle.__default_scenario

        # Translate optimizer names:
        additional_impl_args = {}
        translated_optimizers = []
        for i, o in enumerate(optimizer):
            if o == "ILP":
                implementation = OracleType.MOSEK
            elif o == "LegoStore":
                implementation = OracleType.MOSEK
                additional_impl_args["access_cost_heuristic"] = True
            elif o == "Candidates":
                implementation = OracleType.PYTORCH
            elif o == OracleType.KMEANS.value:
                implementation = OracleType.KMEANS
            elif o == OracleType.PROFIT.value:
                implementation = OracleType.PROFIT
            elif o == OracleType.PYTORCH.value:
                addOptimizersFromInput = True
                skip_loaded_optimizers = False
                continue
            else:
                raise ValueError(f"Unknown optimizer type {o}")
                
            # We need to load the additional information from the precomputed data
            if implementation != OracleType.PYTORCH:
                addOptimizersFromInput = True
                skip_loaded_optimizers = skip_loaded_optimizers and True

            o = OptimizerType(type=o, useClarkson=None, useGPU=None, implementation=implementation, implementationArgs=setImplementationArgs(implementation=implementation, args=implArgs, additional_impl_args=additional_impl_args))
            if o.implementation !=  OracleType.NONE:
                translated_optimizers.append(o)

        # Discover optimizers from precomputation
        additional_optimizers = []
        if addOptimizersFromInput:
            additional_optimizers.extend(oracle.__discoverOptimizerTypes(scenarioName))

            for o in additional_optimizers:
                o.implementationArgs = setImplementationArgs(implementation=o.implementation, args=implArgs)

            print("Discovered optimizers from precomputation: ", additional_optimizers)

        # XXX: Initialization needed here?
        oracle.prepare_scenario(scenario_name=scenarioName, optimizer_type=additional_optimizers[-1], skip_instance=True)

        if not skip_loaded_optimizers:
            translated_optimizers.extend(additional_optimizers)

        return oracle, translated_optimizers

if __name__ == "__main__":
    from skypie.__main__ import __main__
    __main__()
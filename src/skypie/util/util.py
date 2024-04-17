from typing import List, Any, Dict, ClassVar
import numpy as np
import torch
from ast import literal_eval
import pkg_resources
import os

from skypie.util.my_dataclasses import *
#from .oracle import Oracle

@dataclass(frozen=True)
class Package_Resources:
    networkPriceFileName: str = field(init=False, default_factory=lambda: pkg_resources.resource_filename(__name__.split(".")[0], "data/network_cost_v2.csv"))
    storagePriceFileName: str = field(init=False, default_factory=lambda: pkg_resources.resource_filename(__name__.split(".")[0], "data/storage_pricing.csv"))
    network_latency_files: str = field(init=False, default_factory=lambda: {size: pkg_resources.resource_filename(__name__.split(".")[0], f"data/latency_{size}.csv") for size in [10485760,20971520,41943040,83886080,167772160,335544320]})
    oracle_defaults: Dict[str,os.PathLike] = field(init=False, default_factory=lambda: Package_Resources.__init_default_oracles())

    @dataclass
    class Oracle_Defaults_Key:
        replication_factor: int
        region_selector: str
        replication_factor_max: int = None
        latency_slo: float = float("inf")
        object_store_selector: str = ""

        def __post_init__(self):
            if self.replication_factor_max is None:
                self.replication_factor_max = self.replication_factor

        def __hash__(self):
            return hash((self.replication_factor, self.region_selector, self.replication_factor_max, self.latency_slo, self.object_store_selector))
        
    @classmethod
    def __init_default_oracles(cls) -> Dict[Oracle_Defaults_Key, str]:
        # Discover default oracles from experiment files in package files
        experiment_file = "experiment.json"
        oracle_dir_pkg = pkg_resources.resource_filename(__name__.split(".")[0], "data/oracles")

        oracle_defaults = dict()
        for root, _dirs, files in os.walk(oracle_dir_pkg):
            if experiment_file in files:
                with open(os.path.join(root, experiment_file), "r") as f:
                    experiment_args = json.load(f)

                    replication_factor = experiment_args.get("replication_factor", 1)
                    replication_factor_max = experiment_args.get("replication_factor_max", replication_factor)
                    latency_slo = experiment_args.get("latency_slo", float("inf"))
                    if latency_slo is None:
                        latency_slo = float("inf")
                    region_selector = experiment_args.get("region_selector", "")
                    object_store_selector = experiment_args.get("object_store_selector", "")

                    key = Package_Resources.Oracle_Defaults_Key(replication_factor=replication_factor, replication_factor_max=replication_factor_max, latency_slo=latency_slo, region_selector=region_selector, object_store_selector=object_store_selector)

                    oracle_defaults[key] = os.path.abspath(root)

        return oracle_defaults
        

    def get_default_oracle(self, *, min_replication_factor: int, region_selector: str, max_replication_factor: int = None, latency_slo: float = float("inf"), object_store_selector: str = "") -> str:
            """
            Retrieves the default oracle based on the specified scenario parameters.

            # Arguments
            - min_replication_factor (int): The minimum replication factor.
            - region_selector (str): The region selector, e.g., "aws", "gcp", "azure" for all regions of AWS/GCP/Azure, respectively, or "aws-us-east-1" for a specific region.
            - max_replication_factor (int, optional): The maximum replication factor. Defaults to min_replication_factor.
            - latency_slo (float, optional): The latency SLO (Service Level Objective). Defaults to infinity.
            - object_store_selector (str, optional): The object store selector, e.g., "General Purpose" for S3 Standard. Defaults to "".

            # Returns
            - str: The path to an available default oracle.

            # Raises
            - KeyError: If there is no default oracle for the specified scenario.
            """
            key = self.Oracle_Defaults_Key(replication_factor=min_replication_factor, replication_factor_max=max_replication_factor, latency_slo=latency_slo, region_selector=region_selector, object_store_selector=object_store_selector)
            return self.oracle_defaults[key]
    
    def get_all_default_oracles(self) -> Dict[Oracle_Defaults_Key, str]:
        """
        Retrieves all available default oracles.

        Returns:
            Dict[Oracle_Defaults_Key, str]: A dictionary with the scenario details as the key and the path to the according the default oracle.
        """
        return self.oracle_defaults

PACKAGE_RESOURCES = Package_Resources()

def deltaEncode(x: "List[int]"):
    """
    Delta encode a list of integers.
    """
    last = 0
    for i in x:
        yield i - last
        last = i

def deltaDecode(x: "List[int]"):
    """
    Delta decode a list of integers.
    """
    last = 0
    for i in x:
        last += i
        yield last

# https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries
def findDiff(d1, d2, path=""):
    result = []
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                result.extend(findDiff(d1[k],d2[k], "%s -> %s" % (path, k) if path else k))
            if d1[k] != d2[k]:
                result.extend([ "%s: " % path, " - %s : %s" % (k, d1[k]) , " + %s : %s" % (k, d2[k])])
                #print("\n".join(result))
        else:
            #print ("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))
            result.append("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))
    return result

def createOptimizer(*, optimizer: str, args: Dict[str, Any]) -> List[OptimizerType]:
    optimizers = []
    for useClarkson in args["useClarkson"]:
        useClarkson = useClarkson == "True"
        for useGPU in args["useGPU"]:
            implementation = OracleType.PYTORCH
            additional_impl_args = {}

            useGPU = useGPU == "True"
            if "lrs" == optimizer:
                implementation = OracleType.PYTORCH
            elif "LegoStore" == optimizer:
                implementation = OracleType.MOSEK
                additional_impl_args["access_cost_heuristic"] = True
            elif "ILP" == optimizer:
                implementation = OracleType.MOSEK
                # Mosek does not support GPU or Clarkson!
                if useGPU or useClarkson:
                    continue
            elif optimizer == OracleType.KMEANS.value:
                implementation = OracleType.KMEANS
            elif optimizer == OracleType.PROFIT.value:
                implementation = OracleType.PROFIT
            elif "PrimalSimplex" == optimizer:
                optimizer = MosekOptimizerType.PrimalSimplex
                implementation = OracleType.PYTORCH
            elif "InteriorPoint" == optimizer:
                optimizer = MosekOptimizerType.InteriorPoint
                implementation = OracleType.PYTORCH
            elif "Free" == optimizer:
                optimizer = MosekOptimizerType.Free
                implementation = OracleType.PYTORCH

            optimizers.append(OptimizerType(type=optimizer, useClarkson=useClarkson, useGPU=useGPU, implementation=implementation, implementationArgs=setImplementationArgs(implementation=implementation, args=args, additional_impl_args=additional_impl_args)))

    return optimizers

def setImplementationArgs(*, implementation: "OracleType", args: Dict[str, Any], additional_impl_args : Dict[str, Any] = {}) -> Dict[str, Any]:
    implementationArgs = {}
    if implementation == OracleType.PYTORCH:
        precision = args.get("precision", "float64")
        dataType = torch.float64
        if precision == "float32":
            dataType = torch.float32
        elif precision == "float16":
            dataType = torch.float16
        elif precision == "bfloat16":
            dataType = torch.bfloat16
        implementationArgs["data_type"] = dataType

        if "torchDeviceRayShooting" in args:
            implementationArgs["device_query"] = args["torchDeviceRayShooting"]
            #implementationArgs["device_check"] = args["torchDeviceRayShooting"]
        if "device_query" in args:
            implementationArgs["device_query"] = args["torchDeviceRayShooting"]
        
        if implementationArgs.get("device_query", "PLACEHOLDER") == "mps":
            implementationArgs["data_type"] = torch.float32

    elif implementation == OracleType.MOSEK:

        implementationArgs["networkPriceFileName"] = args.get("networkPriceFile", PACKAGE_RESOURCES.networkPriceFileName)
        implementationArgs["storagePriceFileName"] = args.get("storagePriceFile", PACKAGE_RESOURCES.storagePriceFileName)

        if 'noStrictReplication' in args:
            implementationArgs["strictReplication"]=not args['noStrictReplication']
        if 'minReplicationFactor' in args:
            implementationArgs["minReplicationFactor"] = args["minReplicationFactor"]

        implementationArgs["network_latency_file"] = args.get("network_latency_file", None)
        implementationArgs["latency_slo"] = args.get("latency_slo", None)

        if "ignore_considered_scenario" in args:
            implementationArgs["ignore_considered_scenario"] = args["ignore_considered_scenario"]
        
        if "region_selector" in args:
            implementationArgs["region_selector"] = args["region_selector"]
        if "object_store_selector" in args:
            implementationArgs["object_store_selector"] = args["object_store_selector"]

    elif implementation == OracleType.KMEANS:
        if not "networkPriceFile" in args or not "storagePriceFile" in args:
            raise ValueError("Network and storage price files must be provided for Kmeans optimizer.")
        implementationArgs["minReplicationFactor"] = args["minReplicationFactor"]
        implementationArgs["networkPriceFileName"] = args["networkPriceFile"]
        implementationArgs["storagePriceFileName"] = args["storagePriceFile"]
        implementationArgs["strictReplication"]=not args['noStrictReplication']
        if "max_iterations" in args:
            implementationArgs["max_iterations"] = args["max_iterations"]
        if "threshold" in args:
            implementationArgs["threshold"] = args["threshold"]
            
    elif implementation == OracleType.PROFIT:
        if not "networkPriceFile" in args or not "storagePriceFile" in args:
            raise ValueError("Network and storage price files must be provided for Profit-based optimizer.")
        implementationArgs["networkPriceFileName"] = args["networkPriceFile"]
        implementationArgs["storagePriceFileName"] = args["storagePriceFile"]

    implementationArgs["threads"] = args.get("threads", 0)

    implementationArgs.update(additional_impl_args)

    return implementationArgs

def compactifyForPrecomputation(precomputed_data):
    """
    Compactify in-place the precomputed data to save space.
    Only keep costWLPlane of the replication schemes.
    """

    scenario = precomputed_data["tier_advise"]["replication_factor"]
    for s in scenario.values():
        for data in s.values():
            c0 = data["candidate_partitions"][0]
            if isinstance(c0, dict):
                data["applictionRegionMapping"] = ReplicationScheme(c0).applictionRegionMapping
            elif isinstance(c0, ReplicationScheme):
                data["applictionRegionMapping"] = c0.applictionRegionMapping
                
            for candidate in data["candidate_partitions"]:
                deleteProperties = ["replicationScheme", "costWLHalfplaneRational", "inequalities"]
                for prop in deleteProperties:
                    if prop in candidate:
                        del candidate[prop]

    return precomputed_data

from dataclasses import dataclass, field, is_dataclass, asdict
from typing import *
import numpy as np
from enum import Enum
import time
import pandas as pd
import io
import sys
import json
import torch

from sky_pie_baselines import PyLoader

# TODO: Set dataclass frozen=True after testing

@dataclass(init=False, frozen=False)
class Workload:
    #Fixed costs: ignored
    size: np.float64 = -1
    put: np.float64 # Aggregate put
    get: "np.ndarray" = None
    ingress: "np.ndarray" = None #field(init=False)
    egress: "np.ndarray" = None #field(init=False)
    equation: "np.ndarray" = None
    put_raw: "np.ndarray" = field(init=False, default_factory=lambda: np.array([], dtype=np.float64))
    rescale: np.float64 = field(default_factory=lambda: np.float64(1.0))

    def __init__(self, *, equation: "np.ndarray" = None, size: "np.float64" = None, put: "np.ndarray" = None, get: "np.ndarray" = None, ingress: "np.ndarray|None" = None, egress: "np.ndarray|None" = None, rescale: "np.float64" = np.float64(1.0)) -> None:
        
        self.rescale = rescale
        
        if equation is not None:
            self.equation = equation
            no_apps = (len(equation) - 3) // 3
        else:
            no_apps = get.shape[0]

            assert(size is not None)
            assert(put is not None)
            assert(get is not None)
            assert(put.shape == get.shape)

            self.put_raw = put

            self.equation = np.zeros(shape=(3 + no_apps*3,), dtype=np.float64)
            start=0
            end=1
            # For corrrect dot product with fixed costs
            self.equation[start:end] = 1
            start=end
            end=start+1
            self.equation[start] = size
            start=end
            end+=1
            # Put is aggregate workload, as it is a single cost
            self.equation[start] = np.sum(put)
            start=end
            end+=no_apps
            self.equation[start:end] = get
            # Ingress is the product of puts and size
            start=end
            end+=no_apps
            # Use given ingress otherwise compute by put accesses and size
            if ingress is not None:
                assert(ingress.shape[0] == no_apps)
                self.equation[start:end] = ingress
            else:   
                self.equation[start:end] = put * size
            # Egress is the product of gets and size
            start=end
            end+=no_apps
            if egress is not None:
                assert(egress.shape[0] == no_apps)
                self.equation[start:end] = egress
            else:
                self.equation[start:end] = get * size

        self.equation = self.equation * self.rescale
        self.put_raw = self.put_raw * self.rescale

        start=1
        end=2
        self.size = self.equation[start]
        start=end
        end+=1
        self.put = self.equation[start]
        start=end
        end+=no_apps
        self.get = self.equation[start:end]
        start=end
        end+=no_apps
        self.ingress = self.equation[start:end]
        start=end
        end+=no_apps
        self.egress = self.equation[start:end]

@dataclass(init=False, frozen=False)
class Cost:
    fixedCost: float
    storage: float
    put: float
    get: "np.ndarray" = field(default_factory=lambda: np.array([], dtype=np.float64))
    ingress: "np.ndarray" = field(default_factory=lambda: np.array([], dtype=np.float64))
    egress: "np.ndarray" = field(default_factory=lambda: np.array([], dtype=np.float64))
    equation: "np.ndarray" = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __init__(self, jsonObject: Dict[str, Any]):
        self.fixedCost = np.float64(jsonObject["fixedCost"])
        self.storage = np.float64(jsonObject["storage"])
        self.put = np.float64(jsonObject["put"])
        self.get = np.array(jsonObject["get"], dtype=np.float64)
        self.ingress = np.array(jsonObject["ingress"], dtype=np.float64)
        self.egress = np.array(jsonObject["egress"], dtype=np.float64)
        self.equation = np.array(jsonObject["equation"], dtype=np.float64)

        assert self.get.shape[0] == self.ingress.shape[0]
        assert self.ingress.shape[0] == self.egress.shape[0]
        assert len(self.equation.shape) == 1 and len(self.equation) == 3 + self.get.shape[0] + self.ingress.shape[0] + self.egress.shape[0]

    def compute_cost(self, workload: Workload) -> float:
        # Fixed cost + workload costs
        assert(len(self.equation) == len(workload.equation))
        return self.equation[0] + np.dot(self.equation[1:], workload.equation[1:])

@dataclass(init=False, frozen=False)
class ReplicationScheme:
    name: str
    # Choice of object stores
    objectStores: Dict[str, Any]
    # Assignments of application regions to object stores
    assignments: Dict[Any, Set[Any]]
    # Mapping of application regions to indexes of get, ingress, and egress costs
    applictionRegionMapping: Dict[str, int]
    # Cost of repliction scheme
    cost: Cost
    # CostWorkloadHalfplane
    costWorkloadHalfplane: "np.ndarray" = field(default_factory=lambda: np.array([]))
    # Inequalities of workload partition, as single matrix
    inequalities: "np.ndarray" = field(default_factory=lambda: np.array([[]]))
    # Incidence to other replication schemes
    #incidentPartitions: "np.ndarray" = field(default_factory=lambda: np.array([]))
    # Incidence to other replication schemes by optimizer type
    #incidentPartitionsByOptimizer: Dict[str, Dict] = field(default_factory=lambda: {})

    def __init__(self, jsonObject: Dict[str, Any]):
        if "replicationScheme" in jsonObject:
            self.name = jsonObject["replicationScheme"].get("name", "")
            self.objectStores = jsonObject["replicationScheme"]["objectStores"]
            objectStore = jsonObject["replicationScheme"]["appAssignments"][0]["objectStore"]
            if isinstance(objectStore, str):
                self.assignments = {v["app"]: set([v["objectStore"]]) for v in jsonObject["replicationScheme"]["appAssignments"]}
            else:
                # Assuming it is some iterable otherwise
                self.assignments = {v["app"]: set(v["objectStore"]) for v in jsonObject["replicationScheme"]["appAssignments"]}
            self.applictionRegionMapping = { v["app"]: index for index, v in enumerate(jsonObject["replicationScheme"]["appAssignments"])}
            self.cost = Cost(jsonObject["replicationScheme"]["cost"]) if "cost" in jsonObject["replicationScheme"] else None
        else:
            self.name = ""
            self.objectStores = {}
            self.assignments = {}
            self.applictionRegionMapping = {}
            self.cost = None
        self.costWorkloadHalfplane = np.array(jsonObject["costWLHalfplane"], dtype=np.float64)
        if "inequalities" in jsonObject:
            self.inequalities = np.array(jsonObject["inequalities"], dtype=np.float64)
        else:
            self.inequalities = np.array([], dtype=np.float64)
        #if "incidentPartitions" in jsonObject:
        #    self.incidentPartitions = np.array(jsonObject.get("incidentPartitions", []), dtype=np.int64)
        #else:
        #    self.incidentPartitions = np.array([], dtype=np.int64)

        if self.inequalities.shape[0] > 0:
            assert len(self.inequalities.shape) == 2 and len(self.cost.equation) == self.inequalities.shape[1]

        #assert self.inequalities.shape[0] == self.incidentPartitions.shape[0]

    def getWorkloadMapped(self, size: float, put: Dict[str, float], get: Dict[str, float], ingress: Dict[str, float], egress: Dict[str, float]) -> Workload:
        mappedPut = np.zeros(len(self.assignments), dtype=np.float64)
        mappedGet = np.zeros_like(mappedPut)
        mappedIngress = np.zeros_like(mappedPut)
        mappedEgress = np.zeros_like(mappedPut)

        for app, value in put.items():
            mappedPut[self.applictionRegionMapping[app]] = value

        for app, value in get.items():
            mappedGet[self.applictionRegionMapping[app]] = value

        for app, value in ingress.items():
            mappedIngress[self.applictionRegionMapping[app]] = value

        for app, value in egress.items():
            mappedEgress[self.applictionRegionMapping[app]] = value

        return Workload(size=size, put=mappedPut, get=mappedGet, ingress=mappedIngress, egress=mappedEgress)
    
    def getWorkload(self, equation: "np.ndarray", numApplictionRegions: int) -> Workload:
        assert numApplictionRegions == len(self.applictionRegionMapping)
        return Workload(equation=equation)

    def compute_cost(self, workload:Workload) -> float:
        return self.cost.compute_cost(workload)


@dataclass
class Timing:
    name: str = ""
    __startTime: int = field(default_factory=lambda: time.time_ns())
    __duration: int = 0
    verbose: int = 0

    def __iadd__(self, other: "Timing") -> "Timing":
        self.__duration += other.__duration
        return self

    def start(self):
        self.__duration = 0
        self.__startTime = time.time_ns()

    def cont(self):
        self.__startTime = time.time_ns()

    def stop(self):
        now = time.time_ns()
        self.__duration += now - self.__startTime
        if self.verbose > 0:
            print(f"{self.name} = {self.__duration} ns")
        return self.__duration
    
    def getDuration(self):
        return self.__duration

@dataclass
class Timer:
    verbose: int = 0
    __computationTime: Timing = field(default_factory=lambda: Timing(name="Computation"))
    __overheadTime: Timing = field(default_factory=lambda: Timing(name="With overhead"))

    def __post_init__(self):
        self.__computationTime.verbose = self.verbose
        self.__overheadTime.verbose = self.verbose

    def __iadd__(self, other: "Timer"):
        self.__computationTime += other.__computationTime
        self.__overheadTime += other.__overheadTime
        return self

    def startComputation(self):
        self.__computationTime.start()

    def continueComputation(self):
        self.__computationTime.cont()

    def stopComputation(self):
        return self.__computationTime.stop()

    def startOverhead(self):
        self.__overheadTime.start()

    def continueOverhead(self):
        self.__overheadTime.cont()

    def stopOverhead(self):
        return self.__overheadTime.stop()

    def cont(self):
        self.continueComputation()
        self.continueOverhead()

    def stop(self):
        self.stopComputation()
        self.stopOverhead()
        return self.getTotalTime()

    def getComputationTime(self):
        return self.__computationTime.getDuration()
    
    def getOverheadTime(self):
        return self.__overheadTime.getDuration()
    
    def getTotalTime(self):
        return self.getOverheadTime()

class OracleType(Enum):
    NONE=0
    SKYPIE = "SkyPIE"
    PYTORCH = "SkyPIE"
    MOSEK = "ILP"
    ILP = "ILP"
    KMEANS = "Kmeans"
    PROFIT = "Profit-based"
    CANDIDATES = "Candidates"

class NormalizationType(Enum):
    No = 0
    log10All = 1
    log10ColumnWise = 2
    standardScoreColumnWise = 3
    Mosek = 4 # Mosek's scaling

class MosekOptimizerType(str, Enum):
    Free = "free"
    InteriorPoint = "intpnt"
    PrimalSimplex = "primalSimplex"    
    # "free", "intpnt", "conic", "primalSimplex", "dualSimplex", "freeSimplex", "mixedInt"

    @staticmethod
    def from_str(label):
        if label in ('Free', 'free'):
            return MosekOptimizerType.Free
        elif label in ('InteriorPoint', 'intpnt'):
            return MosekOptimizerType.InteriorPoint
        elif label in ('PrimalSimplex', 'primalSimplex'):
            return MosekOptimizerType.PrimalSimplex
        else:
            return label

@dataclass
class OptimizerType:
    type: "MosekOptimizerType|str"
    useClarkson: bool = False
    useGPU: bool = False
    name: str = "" #field(init=False)
    implementation: OracleType = OracleType.NONE
    implementationArgs: Dict[str, Any] = field(default_factory=dict)
    iteration: "None|int" = None
    dsize: "None|int" = None
    strictReplication: bool = field(default=True)

    def __post_init__(self):
        self.name = str(self.type)

        if self.useClarkson:
            self.name += "_Clarkson"

        if self.useGPU:
            self.name += "_GPU"

        if self.iteration is not None:
            self.name += f"_iter{self.iteration}"

        if self.dsize is not None:
            self.name += f"_dsize{self.dsize}"

        if "strictReplication" in self.implementationArgs:
            self.strictReplication = self.implementationArgs["strictReplication"]

        """ if self.strictReplication:
            self.name += "_strictReplication"
        else:
            self.name += "_noStrictReplication" """

    def get_parameters(self) -> Dict[str, Any]:
        return {"Optimizer" : self.name,
            "Type": str(self.type),
            "Clarkson": self.useClarkson,
            "GPU": self.useGPU,
            "Iteration": self.iteration,
            "Division Size": self.dsize,
            "Strict Replication": self.strictReplication
        }

    def custom_copy(self, **kwargs) -> "OptimizerType":
        otDict = asdict(self)
        del otDict["name"]
        otDict.update(kwargs)
        return OptimizerType(**otDict)

@dataclass
class OptimizationResult:
    # Execution time of the optimization
    #time: Timer = field(default_factory=lambda: Timer())
    # Objective value, i.e., result of optimization
    value: float = -1
    # Choice of object stores over time
    objectStores: List[Set[Any]] = field(default_factory=lambda: [])
    # Assignment of application regions to object stores over time
    assignments: List[Dict[Any, Set[Any]]] = field(default_factory=lambda: [])
    # Migration of portion of object from between object stores at time t
    migrations: List[Dict[Any, Dict[Any, float]]] = field(default_factory=lambda: [])

    def __post_init__(self):
        # Filter out empty migrations
        self.migrations = [
            { k:v for (k,v) in m.items() if len(v) > 0} for m in self.migrations
        ]

@dataclass
class Problem:
    storagePriceFileName: str = field(default_factory=lambda: None) # Name of input file for object stores' prices
    networkPriceFileName: str = None # Name of input file for network transfer prices
    network_latency_file: str = None # Name of input file for network latency
    selector: str = "" # Selector string to match object store and application region vendor and region
    T: List[int] = field(default_factory=list) # Time slots, Legacy of SpanStore was: Epoch duration
    Count: int = 1 # Object count is obsolete
    AS: List[Any] = field(default_factory=list) # List of appliction regions (a.k.a. access set accessing the object)
    #f: int = 0 # Number of replicas
    min_f: int = 0 # Minimum number of replicas
    max_f: int = 0 # Maximum number of replicas
    latency_slo: float = 0 #Dict[int, Dict[int, int]] # SLO of pth percentalile to access 
    L_C: Dict[Any, Dict[Any, float]] = field(default_factory=dict) # (i,j,latency): pth percentile latency from data center i to __data center__ j 
    L_S: Dict[int, Dict[int, float]] = field(default_factory=dict) # (i,j,latency): pth percentile latency from data center i to __object store__ j
    PUTs: List[Dict[Any, float]] = field(default_factory=list) # List of put per time slot per application region i: [(i: puts), ...]
    GETs: List[Dict[Any, float]] = field(default_factory=list) # List of put per time slot per application region i: [(i: gets), ...]
    Egress: List[Dict[Any, float]] = field(default_factory=list) # List of egress volume per time slot per application region i: [(i: egress), ...]
    Ingress: List[Dict[Any, float]] = field(default_factory=list) # List of ingress volume per time slot per application region i: [(i: ingress), ...]
    Size_total: float = 0 # Total size of the objects
    Size_avg: float = field(init=False, default=1) # Average size of the objects
    PriceGet: Dict[Any, float] = field(default_factory=dict) # Price per get per object store: (i: price)
    PricePut: Dict[Any, float] = field(default_factory=dict) # Price per put per object store: (i: price)
    PriceStorage: Dict[Any, float] = field(default_factory=dict) #Price per sizeUnit per object store: (i:price)
    PriceNet: Dict[Any, Dict[Any, float]] = field(default_factory=dict) # Price of network transfer per networkUnit, from data center i to data center j: (i,j, price)
    initialState: OptimizationResult = field(default_factory=OptimizationResult) # Initial state of the system for historical data and hinting future migrations
    verbose: int = 0 # Verbosity level
    loadFromFile: bool = True # Load from file if filenames are provided
    applicationRegionLoad: Dict[Any, int] = field(default_factory=dict) # List of application regions to load from file
    storageLoad: List[Any] = field(default_factory=list) # List of destinations to load from file
    region_selector: str = "" # Selector string to match considered cloud vendor and region
    object_store_selector: str = "" # Selector string to match object store tier
    # Derived helper inputs
    AStranslate: Dict[Any, int] = field(init=False) # Mapping from original AS to dense AS
    ASdense: List[int] = field(init=False) # List of dense AS
    dest: Dict[Any, int] = field(init=False) # Mapping from original destination to dense destination
    destDense: List[int] = field(init=False) # List of dense destinations
    destTranslate: Dict[int, Any] = field(init=False) # Mapping from dense IDs to object stores
    localSchemes: Dict[Any,ReplicationScheme] = field(default_factory=lambda: []) # Precomputed locally optimal schemes
    threads: int = 0 # Number of threads to use for parallelization, 0 means use all available

    def __loadCSV_old(self,*, storagePriceFileName: str, networkPriceFileName: str, applicationRegionLoad: Dict[Any, int] = dict(), storageLoad: List[Any] = list(), selector: str = "", verbose: int = 0):
        dfStore = pd.read_csv(storagePriceFileName, sep=",", low_memory=False)
        storage = dict()
        applicationRegions = dict()

        # Load pricing for storage
        # Key = vendor-region-name-tier

        useStorageLoad = len(storageLoad) > 0
        useApplictionRegionLoad = len(applicationRegionLoad) > 0

        def loadStorageAndRegions(x):
            applicationKey = x['Vendor'].iloc[0] + "-" + x['Region'].iloc[0]
            storageKey = x['Vendor'].iloc[0] + "-" + x['Region'].iloc[0] + "-" + x['Name'].iloc[0] + "-" + x['Tier'].iloc[0]

            if (useStorageLoad and storageKey in storageLoad) or (not useStorageLoad and selector in storageKey):
                # Drop volume discount and just take the most expensive price
                maxPrice = x[['Vendor', 'Region', 'Name', 'Tier', 'Group', 'PricePerUnit']].groupby('Group').max().reset_index()
                storage.update(
                    {storageKey: dict(
                            zip(maxPrice['Group'].values.tolist() + ["Vendor", "Region", "Name", "Tier", "NetworkCost"], maxPrice['PricePerUnit'].values.tolist() + [maxPrice['Vendor'].iloc[0], maxPrice['Region'].iloc[0], maxPrice['Name'].iloc[0], maxPrice['Tier'].iloc[0], dict()])
                        )
                    }
                )

            if (useApplictionRegionLoad and applicationKey in applicationRegionLoad) or (not useApplictionRegionLoad and selector in applicationKey):
                applicationRegions.update({applicationKey: {"Vendor": x['Vendor'].iloc[0], "Region": x['Region'].iloc[0], "NetworkCost": dict()}})

            """
            if selector in key:
                storage.update(
                    {key: dict(
                            zip(x['Group'].values.tolist() + ["Vendor", "Region", "Name", "Tier", "NetworkCost"], x['PricePerUnit'].values.tolist() + [x['Vendor'].iloc[0], x['Region'].iloc[0], x['Name'].iloc[0], x['Tier'].iloc[0], dict()])
                        )
                    }
                )
                applicationRegions.update({x['Vendor'].iloc[0] + "-" + x['Region'].iloc[0]: {"Vendor": x['Vendor'].iloc[0], "Region": x['Region'].iloc[0], "NetworkCost": dict()}})
            """

        dfStore.groupby(['Vendor', 'Region', 'Name', 'Tier']).apply(loadStorageAndRegions)

        # Load pricing for network as dict["src", dict["dest", float]]
        def loadNetworkPriceStore(x, store):
            # Add network cost from object store to application region
            destApplicationRegionKey = x['dest_vendor'].iloc[0] + "-" + x['dest_region'].iloc[0]
            if destApplicationRegionKey in applicationRegions:
                storage[store]["NetworkCost"].update({destApplicationRegionKey: x['cost'].iloc[0]})

            # Find all object stores in destination region
            #dfQuery = dfStore.query("Vendor == '" + x['dest_vendor'].iloc[0] + "' and Region == '" + x['dest_region'].iloc[0] + "'")
            dfQuery = dfStore[(dfStore['Vendor'] == x['dest_vendor'].iloc[0]) & (dfStore['Region'] == x['dest_region'].iloc[0])]
            if dfQuery.empty:
                if verbose > 1:
                    print("No object store found for vendor " + x['dest_vendor'].iloc[0] + " and region " + x['dest_region'].iloc[0])
            else:
                dfQuery.groupby(['Vendor', 'Region', 'Name', 'Tier']).apply(lambda y: storage[store]["NetworkCost"].update({y['Vendor'].iloc[0] + "-" + y['Region'].iloc[0] + "-" + y['Name'].iloc[0] + "-" + y['Tier'].iloc[0]: x['cost'].iloc[0]}))


        def loadNetworkPriceApplictionRegion(x):
            # Add network cost from application region to object store
            srcApplicationRegionKey = x['src_vendor'].iloc[0] + "-" + x['src_region'].iloc[0]
            # Find all object stores in destination region
            #dfQuery = dfStore.query("Vendor == '" + x['dest_vendor'].iloc[0] + "' and Region == '" + x['dest_region'].iloc[0] + "'")
            dfQuery = dfStore[(dfStore['Vendor'] == x['dest_vendor'].iloc[0]) & (dfStore['Region'] == x['dest_region'].iloc[0])]
            if dfQuery.empty:
                if verbose > 1:
                    print("No object store found for vendor " + x['dest_vendor'].iloc[0] + " and region " + x['dest_region'].iloc[0])
            else:
                dfQuery.groupby(['Vendor', 'Region', 'Name', 'Tier']).apply(lambda y: applicationRegions[srcApplicationRegionKey]["NetworkCost"].update({y['Vendor'].iloc[0] + "-" + y['Region'].iloc[0] + "-" + y['Name'].iloc[0] + "-" + y['Tier'].iloc[0]: x['cost'].iloc[0]}))


        dfNet = pd.read_csv(networkPriceFileName, sep=",", low_memory=False)
        # Ensure that there is network costs from the region to itself
        # Find regions that have network costs to themselves
        missingRecords = []
        def addNetworkMissingSelf(x) -> None:
            # d.apply(lambda x: {'src_vendor': x['src_vendor'].iloc[0], 'src_region': x['src_region'].iloc[0], 'dest_vendor': x['src_vendor'].iloc[0], 'dest_region': x['src_region'].iloc[0], 'cost': 0} if x[(x['dest_vendor'] == x['src_vendor']) & (x['dest_region'] == x['src_region'])].empty else None)
            if x[(x['dest_vendor'] == x['src_vendor']) & (x['dest_region'] == x['src_region'])].empty:
                missingRecords.append({'src_vendor': x['src_vendor'].iloc[0], 'src_region': x['src_region'].iloc[0], 'dest_vendor': x['src_vendor'].iloc[0], 'dest_region': x['src_region'].iloc[0], 'cost': 0})
        dfNet.groupby(['src_vendor', 'src_region']).apply(addNetworkMissingSelf)
        #missingRecords = dfNetSelf.apply(lambda x: {'src_vendor': x['src_vendor'].iloc[0], 'src_region': x['src_region'].iloc[0], 'dest_vendor': x['src_vendor'].iloc[0], 'dest_region': x['src_region'].iloc[0], 'cost': 0}, axis=1)
        dfMissingRecords = pd.DataFrame(missingRecords, columns=dfNet.columns.values)
        dfNet = pd.concat([dfNet, dfMissingRecords], ignore_index=True)

        # From object store to appliction region dict["srcObjectStore"]["dstRegion"] = price
        for store in storage:
            dfNet.query("src_vendor == '" + storage[store]["Vendor"] + "' and src_region == '" + storage[store]["Region"] + "'").groupby(['dest_vendor', 'dest_region']).apply(loadNetworkPriceStore, store=store)

        for region in applicationRegions:
            dfNet.query("src_vendor == '" + applicationRegions[region]["Vendor"] + "' and src_region == '" + applicationRegions[region]["Region"] + "'").groupby(['dest_vendor', 'dest_region']).apply(loadNetworkPriceApplictionRegion)

        # Set member variables with loaded data
        self.PriceGet = {key: storage[key]["get request"] for key in storage}
        self.PricePut = {key: storage[key]["put request"] for key in storage}
        self.PriceStorage = {key: storage[key]["storage"] for key in storage}
        self.PriceNet = {key: storage[key]["NetworkCost"] for key in storage}
        self.PriceNet.update({key: applicationRegions[key]["NetworkCost"] for key in applicationRegions})

        # Heal missing network pricing between application region and object stores in same region
        for store in storage:
            if store not in self.PriceNet[store]:
                self.PriceNet[store][store] = 0
            for app in applicationRegions:
                if app not in self.PriceNet[store] and app in store:
                    self.PriceNet[store][app] = 0

                if store not in self.PriceNet[app] and app in store:
                    self.PriceNet[app][store] = 0

        # Add get transfer cost to cost of network transfer from object store to application region
        for store in storage:
            if "get transfer" in storage[store]:
                getTransfer = storage[store]["get transfer"]
                for region in self.PriceNet[store]:
                    if region != store:
                        self.PriceNet[store][region] += getTransfer
            if "put transfer" in storage[store]:
                putTransfer = storage[store]["put transfer"]
                for region in self.PriceNet:
                    if region != store:
                        self.PriceNet[region][store] += putTransfer

        # Take mapping of application regions to dense keys from outside, such that it's identical with oracle's mapping
        if len(self.applicationRegionLoad) > 0:
            self.AS = list(self.applicationRegionLoad.keys())
            self.AStranslate = self.applicationRegionLoad
        else:
            self.AS = list(applicationRegions.keys())

    def __loadCSV(self,*, storagePriceFileName: str, networkPriceFileName: str, applicationRegionLoad: Dict[Any, int] = dict(), storageLoad: List[Any] = list(), network_latency_file: str = None, latency_SLO: float = None, region_selector: str = None, object_store_selector: str = None, verbose: int = 0):
        
        if latency_SLO is None:
            network_latency_file = None
            
        # If object store selector is given, region selector must be given as well or default to empty string
        if object_store_selector is not None:
            region_selector = region_selector or ""

        # If region selector is given, object stores to load and application regions to load are ignored
        if region_selector is not None:
            storageLoad = []
            applicationRegionLoad = {}

        loader = PyLoader(networkPriceFileName, storagePriceFileName, storageLoad, applicationRegionLoad, network_latency_file, latency_SLO, verbose, region_selector=region_selector, object_store_selector=object_store_selector)

        #storage = loader.object_store_names()
        applicationRegions = loader.application_region_mapping()

        # Assert that no data is missing, i.e., the application regions to be loaded are acutally loaded
        assert len(applicationRegionLoad) == 0 or set(applicationRegionLoad.keys()) == set(applicationRegions.keys()), "Application regions to be loaded are not loaded from file. Requested: " + str(set(applicationRegionLoad.keys())) + ", loaded: " + str(set(applicationRegions.keys()))

        # Set member variables with loaded data
        self.PriceGet = loader.get_price()
        self.PricePut = loader.put_price()
        self.PriceStorage = loader.storage_price()
        self.PriceNet = loader.network_price()
        self.L_S = loader.network_latency

        # Take mapping of application regions to dense keys from outside, such that it's identical with oracle's mapping
        # XXX TB: I think that we should always use the mapping from the loader
        self.AS = list(applicationRegions.keys())
        self.AStranslate = applicationRegions
        #if len(self.applicationRegionLoad) > 0:
        #    self.AS = list(self.applicationRegionLoad.keys())
        #    self.AStranslate = self.applicationRegionLoad
        #else:
        #    self.AS = list(applicationRegions.keys())

    def __post_init__(self):

        if self.loadFromFile and self.storagePriceFileName is not None and self.networkPriceFileName is not None:
            if len(self.applicationRegionLoad) == 0:
                print("Warning: No application regions to load from file specified. All application regions will be loaded from file.")
            # Load input from file and afterwards initialize
            self.__loadCSV(storagePriceFileName=self.storagePriceFileName, networkPriceFileName=self.networkPriceFileName, verbose=self.verbose, applicationRegionLoad=self.applicationRegionLoad, storageLoad=self.storageLoad, network_latency_file=self.network_latency_file, latency_SLO=self.latency_slo, region_selector=self.region_selector, object_store_selector=self.object_store_selector)

            self.loadFromFile = False

        # Time slots
        self.T = list(range(len(self.PUTs)))
        
        self.Size_avg = self.Size_total / self.Count

        # Generate derived helper inputs
        # Number of data centers in access set
        #i = len(AS)
        if len(self.AStranslate) == 0:
            self.AStranslate = { orig: denseKey for (denseKey, orig) in enumerate(self.AS)}
        self.ASdense = list(self.AStranslate.values())
        # Number of data centers with storage services
        #j = len(PriceStorage_i)
        self.dest = { orig: denseKey for (denseKey, orig) in enumerate(self.PriceGet.keys())}
        self.destDense = list(self.dest.values())
        self.destTranslate = {v:k for k,v in self.dest.items()}

        if len(self.initialState.objectStores) > 0:
            # With historical state
            assert len(self.PUTs) > 1, "With historical state, at least two time slots are required, i.e., the historical and the present state."

    def setPUTs(self, PUTs: List[Dict[Any, float]]):
        self.PUTs = PUTs
        self.__post_init__()

    def setSize_total(self, Size_total: float):
        self.Size_total = Size_total
        self.__post_init__()
        
    def setGETs(self, GETs: List[Dict[Any, float]]):
        self.GETs = GETs
        self.__post_init__()
    
    def setEgress(self, Egress: List[Dict[Any, float]]):
        self.Egress = Egress
        self.__post_init__()
    def setIngress(self, Ingress: List[Dict[Any, float]]):
        self.Ingress = Ingress
        self.__post_init__()

    def setWorkload(self, workload: Workload):
        if isinstance(workload, Workload):
            assert(len(workload.put_raw) == len(workload.get)), "Individual PUTs and GETs of each application required!"
            self.PUTs = [{AS: workload.put_raw[dense] for AS, dense in self.AStranslate.items()}]
            self.GETs = [{AS: workload.get[dense] for AS, dense in self.AStranslate.items()}]
            self.Egress = [{AS: workload.egress[dense] for AS, dense in self.AStranslate.items()}]
            self.Ingress = [{AS: workload.ingress[dense] for AS, dense in self.AStranslate.items()}]
            self.Size_total = workload.size
        elif isinstance(workload, list) and len(workload) > 1 and isinstance(workload[0], Workload):
            self.PUTs = []
            self.GETs = []
            self.Egress = []
            self.Ingress = []
            for w in workload:
                assert(len(w.put_raw) == len(w.get)), "Individual PUTs and GETs of each application required!"
                self.PUTs.append({AS: w.put_raw[dense] for AS, dense in self.AStranslate.items()})
                self.GETs.append({AS: w.get[dense] for AS, dense in self.AStranslate.items()})
                self.Egress.append({AS: w.egress[dense] for AS, dense in self.AStranslate.items()})
                self.Ingress.append({AS: w.ingress[dense] for AS, dense in self.AStranslate.items()})
            self.Size_total = workload.size
        elif isinstance(workload, np.ndarray) and len(workload.shape) == 1:
            """no_apps =  len(self.AStranslate)
            if len(workload.shape) == 1:
                assert(len(workload) == 1+2*no_apps), "Size and individual PUTs and GETs of each application required!"
                self.Size_total = workload[0]
                self.PUTs = [{AS: workload[1+dense] for AS, dense in self.AStranslate.items()}]
                self.GETs = [{AS: workload[1+no_apps+dense] for AS, dense in self.AStranslate.items()}]
            elif len(workload.shape) == 2:
                self.Size_total = workload[0,0]
                self.PUTs = []
                self.GETs = []
                for i in range(workload.shape[0]):
                    self.PUTs.append({AS: workload[i,1+dense] for AS, dense in self.AStranslate.items()})
                    self.GETs.append({AS: workload[i,1+no_apps+dense] for AS, dense in self.AStranslate.items()})
            else:
                raise ValueError("Workload must be a 1D or 2D numpy array!")
            """
            raise NotImplementedError("Workload must be of type Workload or List[Workload]. np.ndarray is depricated.")
        else:
            raise ValueError("Workload must be of type Workload or List[Workload]")

        self.__post_init__()

    def setInitialState(self, initialState: OptimizationResult):
        self.initialState = initialState
        self.__post_init__()

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        elif isinstance(o, OracleType):
            return o.value
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, set):
            return list(o)
        elif isinstance(o, torch.dtype):
            return o.__str__()
        else:
            return super().default(o)
    
class EnhancedJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        # Decode optimizer type
        if 'type' in obj and 'useClarkson' in obj and 'useGPU' in obj and 'implementation' in obj:
            obj["type"] = MosekOptimizerType.from_str(obj["type"])
            obj["implementation"] = OracleType(obj["implementation"])
            if "implementationArgs" in obj and "dataType" in obj["implementationArgs"]:
                name_to_dtype = {
                    f.__str__(): f for f in [torch.float64, torch.float32, torch.float16, torch.bfloat16]
                }
                obj["implementationArgs"]["dataType"] = name_to_dtype[obj["implementationArgs"]["dataType"]]
            return OptimizerType(**obj)
        return obj
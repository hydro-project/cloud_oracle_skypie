import copy
from dataclasses import dataclass, field

from skypie.util.my_dataclasses import *
from skypie.oracle_impls.oracle_ilp.horizonControlOptimizer import recedingHorizonControl, Problem
#from .oracle import Oracle

@dataclass
class OracleImplMosek:
    networkPriceFileName: str
    storagePriceFileName: str
    network_latency_file: str = None
    latency_slo: float = None
    strictReplication: bool = True
    minReplicationFactor: int = 0
    threads: int = 0
    access_cost_heuristic: bool = False
    __problem: Problem = field(init=False)
    __oracle: "Oracle" = field(init=False)
    __problemArgs: Dict[str, Any] = field(init=False, default_factory=dict)

    def prepare_schemes(self, oracle: "Oracle"):
        """
        Extract the replication factor and other parameters from the scenarion.
        Then load  the problem
        """

        self.__oracle = oracle

        storageLoad = oracle.get_object_stores_considered()
        applicationRegionLoad = oracle.get_application_regions()
        minReplicationFactor = int(oracle.get_min_replication_factor() if self.minReplicationFactor <= 0 else self.minReplicationFactor)-1
        if self.strictReplication:
            maxReplicationFactor = int(oracle.get_max_replication_factor())-1
        else:
            maxReplicationFactor = len(oracle.get_object_stores_considered())
        problemArgsNow = {"storagePriceFileName": self.storagePriceFileName, "networkPriceFileName": self.networkPriceFileName, "applicationRegionLoad": applicationRegionLoad, "storageLoad": storageLoad, "min_f": minReplicationFactor, "max_f": maxReplicationFactor, "verbose": oracle.verbose, "threads": self.threads, "network_latency_file": self.network_latency_file, "latency_slo": self.latency_slo}
        doLoad = len(self.__problemArgs) < 1 or not np.all([k in problemArgsNow and v == problemArgsNow[k] for k, v in self.__problemArgs.items()])
        if doLoad:
            self.__problem = Problem(**problemArgsNow)
            self.__problemArgs = problemArgsNow
        else:
            # We can reuse the problem as is, the arguments are the same
            pass

    def query(self, w: "Workload|List[Workload]", timer: Timer = None) -> "List[float, OptimizationResult]":
        """
        This function solves an integer linear program for the optimal scheme for the given workload, using Mosek.
        XXX: It does not support batching but instead solves for each workload individually.
        It returns the list of indexes of the optimal schemes.

        TODO: Batching by computing for workloads over time without migration costs, but we need the individual opt. values for the workloads.
        """

        if isinstance(w, Workload):
            w = [w]

        # Result
        res = []

        for workload in w:            
            print(f"Query done: {len(res)}/{len(w)}")
            sys.stdout.flush()
            self.__problem.setWorkload(workload)
            res.append(recedingHorizonControl(**self.__problem.__dict__, withRouting = False, withMigrationCosts=False, timer=timer))

            if self.__oracle.verbose > 1:
                print(self.__problem)
                res[-1].__dict__["problem"] = copy.deepcopy(self.__problem)

            print(f"Query done: {len(res)}/{len(w)}")
            sys.stdout.flush()

        return [ (r.value, r) for r in res]
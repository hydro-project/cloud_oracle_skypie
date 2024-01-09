import numpy as np
from sky_pie_baselines import ProfitBasedOptimizer, Workload as WorkloadRust

from dataclasses import dataclass, field

from skypie.oracle_impls.oracle_interface import OracleInterface
from skypie.util.my_dataclasses import *
#from .load_proto import is_decision, load_cost_matrix

@dataclass
class OracleImplProfit(OracleInterface):
    networkPriceFileName: str
    storagePriceFileName: str
    threads: int = 0
    oracle: "Oracle" = field(init=False, default=None)
    max_iterations: int = 100
    __problemArgs: Dict[str, Any] = field(init=False, default_factory=dict)
    
    def prepare_schemes(self, oracle: "Oracle"):
        self.oracle = oracle

        storageLoad = self.oracle.get_object_stores_considered()
        applicationRegionLoad = self.oracle.get_application_regions()

        # Profit-based optimizer does not consider replication factor
        minReplicationFactor = 0
        maxReplicationFactor = 0

        problemArgsNow = {"storagePriceFileName": self.storagePriceFileName, "networkPriceFileName": self.networkPriceFileName, "applicationRegionLoad": applicationRegionLoad, "storageLoad": storageLoad, "min_f": minReplicationFactor, "max_f": maxReplicationFactor, "verbose": self.oracle.verbose, "threads": self.threads}
        doLoad = len(self.__problemArgs) < 1 or not np.all([k in problemArgsNow and v == problemArgsNow[k] for k, v in self.__problemArgs.items()])
        if doLoad:
            self.optimizer = ProfitBasedOptimizer(self.networkPriceFileName, self.storagePriceFileName, storageLoad, applicationRegionLoad)
        else:
            # We can reuse the problem as is, the arguments are the same
            pass

    def query(self, workloads: "Workload|List[Workload]", timer: Timer = None) -> "List[Tuple[float, Scheme]]":

        if isinstance(workloads, Workload):
            workloads = [workloads]

        # Convert into Rust workloads
        workloads = [
            WorkloadRust(
                size=w.size,
                puts=w.put,
                gets=w.get.tolist(),
                ingress=w.ingress.tolist(),
                egress=w.egress.tolist()
            ) for w in workloads
        ]

        # Ignore the above conversion in timing
        if timer is not None:
            timer.continueOverhead()

        if timer is not None:
            timer.continueComputation()

        res = self.optimizer.optimize_batch(workloads)

        if timer is not None:
            timer.stop()

        #Return list of costs and placement
        return res
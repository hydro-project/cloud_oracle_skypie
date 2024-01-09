import numpy as np
from sky_pie_baselines import KmeansOptimizer, Workload as WorkloadRust

from dataclasses import dataclass, field

from skypie.oracle_impls.oracle_interface import OracleInterface
from skypie.util.my_dataclasses import *
#from .load_proto import is_decision, load_cost_matrix

@dataclass
class OracleImplKmeans(OracleInterface):
    networkPriceFileName: str
    storagePriceFileName: str
    minReplicationFactor: int = 0
    maxReplicationFactor: int = 0
    threshold: float = 0.1 # Threshold to terminate k-means iterations
    threads: int = 0
    strictReplication: bool = True
    oracle: "Oracle" = field(init=False, default=None)
    max_iterations: int = 100
    __problemArgs: Dict[str, Any] = field(init=False, default_factory=dict)
    
    def prepare_schemes(self, oracle: "Oracle"):
        self.oracle = oracle

        storageLoad = self.oracle.get_object_stores_considered()
        applicationRegionLoad = self.oracle.get_application_regions()
        minReplicationFactor = int(self.minReplicationFactor if self.minReplicationFactor > 0 else self.oracle.get_min_replication_factor())
        
        maxReplicationFactor = self.maxReplicationFactor if self.maxReplicationFactor > 0 else minReplicationFactor
        if not self.strictReplication:
            maxReplicationFactor = maxReplicationFactor * 5

        problemArgsNow = {"storagePriceFileName": self.storagePriceFileName, "networkPriceFileName": self.networkPriceFileName, "applicationRegionLoad": applicationRegionLoad, "storageLoad": storageLoad, "min_f": minReplicationFactor, "max_f": maxReplicationFactor, "verbose": self.oracle.verbose, "threads": self.threads}
        doLoad = len(self.__problemArgs) < 1 or not np.all([k in problemArgsNow and v == problemArgsNow[k] for k, v in self.__problemArgs.items()])
        if doLoad:
            self.optimizer = KmeansOptimizer(self.networkPriceFileName, self.storagePriceFileName, storageLoad, applicationRegionLoad, minReplicationFactor, max_num_replicas=maxReplicationFactor, verbose=self.oracle.verbose)
        else:
            # We can reuse the problem as is, the arguments are the same
            pass

    def query(self, workloads: "Workload|List[Workload]", timer: Timer = None) -> "List[Tuple[float, Scheme]]":
            """
            This function applies k-means clustering to find the "f" object stores with lowest read costs for the given workload.
            It returns a list of the costs and the according placement.
            """

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

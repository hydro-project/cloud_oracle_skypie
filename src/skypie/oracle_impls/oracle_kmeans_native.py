import numpy as np
import torch

from dataclasses import dataclass, field

from skypie.oracle_impls.oracle_interface import OracleInterface
from skypie.util.my_dataclasses import *
#from .load_proto import is_decision, load_cost_matrix

@dataclass
class OracleImplKmeans(OracleInterface):
    #device_query: str = "cpu" # We can query for the optimal scheme using different device types, e.g., "cpu" or gpu ("cuda:0")
    #device_check: str = "cpu" # We can check the optimal scheme using different device types, e.g., "cpu" or gpu ("cuda:0")
    #dataType: "torch.dtype" = torch.float64
    networkPriceFileName: str
    storagePriceFileName: str
    minReplicationFactor: int = 0
    threshold: float = 0.1 # Threshold to terminate k-means iterations
    threads: int = 0
    C: List[int] = field(init=False, default_factory=list) # List of client indexes
    S: List[int] = field(init=False, default_factory=list) # List of object store indexes
    #get_costs_o_c: "ndarray[2,float]" = field(init=False, default=None) # List of object stores
    oracle: "Oracle" = field(init=False, default=None)
    max_iterations: int = 100
    __problem: Problem = field(init=False, default=None)
    __problemArgs: Dict[str, Any] = field(init=False, default_factory=dict)
    #__queryData: "torch.Tensor" = field(default_factory=lambda: torch.tensor([])) # Tensor of shape: 1 x schemes x len(costWLHalfplane)
    #__checkData: "torch.Tensor" = field(default_factory=lambda: torch.tensor([])) # Tensor of shape: schemes x 1 x inequalities (i.e., neighbors x len(cost))
    #__shipToQueryDevice: bool = False
    
    def prepare_schemes(self, oracle: "Oracle"):
        self.oracle = oracle
        
        """
        # Set number of threads for pytorch
        if self.threads > 0:
            torch.set_num_threads(self.threads)
        else:
            torch.set_num_threads(1)
        """

        storageLoad = self.oracle.get_object_stores_considered()
        applicationRegionLoad = self.oracle.get_application_regions()
        minReplicationFactor = int(self.oracle.get_min_replication_factor() if self.minReplicationFactor <= 0 else self.minReplicationFactor)
        
        maxReplicationFactor = minReplicationFactor
        problemArgsNow = {"storagePriceFileName": self.storagePriceFileName, "networkPriceFileName": self.networkPriceFileName, "applicationRegionLoad": applicationRegionLoad, "storageLoad": storageLoad, "min_f": minReplicationFactor, "max_f": maxReplicationFactor, "verbose": self.oracle.verbose, "threads": self.threads}
        doLoad = len(self.__problemArgs) < 1 or not np.all([k in problemArgsNow and v == problemArgsNow[k] for k, v in self.__problemArgs.items()])
        if doLoad:
            self.__problem = Problem(**problemArgsNow)
            self.__problemArgs = problemArgsNow

            # Load object store indexes S
            self.S = self.__problem.destDense
            # Load client indexes C
            self.C = self.__problem.ASdense
        else:
            # We can reuse the problem as is, the arguments are the same
            pass
        

        # Create a tensor from the matrix and ship it to the device
        #self.__queryData = torch.tensor(data=matrix, dtype=self.dataType, #device=self.device_query)

        #self.__shipToQueryDevice = "cpu" not in self.device_query

    def query(self, w: "Workload|List[Workload]", timer: Timer = None) -> "List[Tuple[float, Scheme]]":
            """
            This function applies k-means clustering to find the "f" object stores with lowest read costs for the given workload.
            It returns a list of the costs and the according placement.
            """

            if timer is not None:
                timer.continueOverhead()

            if isinstance(w, Workload):
                w = [w]
            
            # Convert workload into tensor
            """
            if isinstance(w, np.ndarray) and not isinstance(w[0], Workload):
                workloadVectorLocal = torch.tensor(data=w, dtype=self.dataType)
            else:
                workloadVectorLocal = torch.tensor(data=[workload.equation for workload in w], dtype=self.dataType)
            
            if self.__shipToQueryDevice:
                workloadVectorDevice = workloadVectorLocal.to(self.device_query)
            else:
                workloadVectorDevice = workloadVectorLocal
            """

            if timer is not None:
                timer.continueComputation()

            res = []
            for workload in w:

                # K-means per workload
                read_choices = self.weighted_k_means(self.__problem.min_f, workload)

                # Compute cost of resulting placement
                cost_total = 0
                for o, C in read_choices.items():
                    o_translated = self.__problem.destTranslate[o]
                    
                    # Put, storage, and ingress costs of object store
                    cost_total += \
                        self.__problem.PricePut[o_translated] * workload.put \
                        + self.__problem.PriceStorage[o_translated] * workload.size
                    
                    # Ingress of this object store and each client
                    for c_name, c_index in self.__problem.AStranslate.items():
                        cost_total += self.__problem.PriceNet[c_name][o_translated] * workload.egress[c_index]
                    
                    # Get and egress costs for assigned clients
                    for c in C:
                        cost_total += self.get_costs_o_c[o, c]

                # Result: List of costs and placement
                res.append((cost_total, None))

            if timer is not None:
                timer.stop()

            #Return list of costs and placement
            return res

    #Algorithm 3 Weighted K-Means for choosing replica locations.
    #1: // Lfixed: set of fixed replica locations, which can’t be moved
    #2: // num replicas: total number of replicas to be placed
    #3: procedure weighted-k-means(Lfixed, num replicas)
    def weighted_k_means(self, num_replicas: int, w: "Workload"):

        assert num_replicas > 0, "Number of replicas must be greater than 0"
        
        # Compute read costs (get + egress) of each application for each object store under given workloads
        # Optimization to avoid recomputation
        get_costs_o_c = np.zeros((len(self.S), len(self.C)))
        for o_name, o_id in self.__problem.dest.items():
            for c_name, c_id in self.__problem.AStranslate.items():
                get_costs_o_c[o_id, c_id] = self.__problem.PriceGet[o_name] * w.get[c_id] + self.__problem.PriceNet[o_name][c_name] * w.egress[c_id]
        
        # IDs of application regions
        C = self.C
        S = self.S
        #C = [i for i,_ in enumerate(A)]
        w_c = [w.get[c] for c in C]
        #4: // pick initial centroids
        #5: G ← Lfixed
        G = []
        #6: sort all client clusters c ∈ C by descending wc
        C.sort(key=lambda x: w_c[x])
        #7: while |G| < num replicas and more client clusters remain
        #8: c ← next client cluster in C
        for c in C:
            if len(G) >= num_replicas:
                break

            #9: if not nearest(c, S) ∈ G then
            #10: add nearest(c, S) to G
            n = self.nearest(c,S)
            if not n in G:
                G.append(n)

        #11: new cost ← cost(G)
        new_cost = self.cost(C, G)
        #12: repeat
        Cg = dict()
        for i in range(self.max_iterations):
            if self.oracle.verbose > 1:
                print(f"K-means iteration {i+1}/{self.max_iterations}")
            #13: prev cost ← new cost
            prev_cost = new_cost
            #14: // cluster clients according to nearest centroid
            #15: ∀g ∈ G let Cg ← {c | g = nearest(c, G)}
            Cg = {g:[] for g in G}
            for c in C:
                Cg[self.nearest(c,G)].append(c)
            #16: // attempt to adjust centroids
            #17: for each g ∈ G \ Lfixed
            for i, g in enumerate(G):
                #18: g′ ← v ∈ S s.t. ∑ c∈Cg wc · rtt(i) c,v is minimized
                g_new = self.argmin(S, Cg[g])
                #19: update centroid g to g′
                G[i] = g_new

                if self.oracle.verbose > 1 and g != g_new:
                    print(f"Centroid {g} changed to {g_new}")
            #20: new cost ← cost(G)
            new_cost = self.cost(C, G)
            #21: until new cost − prev cost < threshold
            if prev_cost - new_cost < self.threshold:
                if self.oracle.verbose > 0:
                    print(f"K-means converged after {i+1} iterations at: {new_cost}")
                break

            if self.oracle.verbose > 1:
                print(f"K-means {i}: {prev_cost} -> {new_cost}")
        #22: return G
        if len(Cg) != num_replicas:
            print(f"Unexpected number of replicas: {len(Cg)}")
        return Cg

    def cost(C: List[int], S: List[int], get_costs_o_c: "ndarray[2,float]"):
        """
        Cost for clients C of object store S
        C: indexes of clients
        S: indexes of object stores
        """
        res = 0
        for o in S:
            for c in C:
                res += get_costs_o_c[o,c]

        return res

    def nearest(self, c:int, S: List[int], get_costs_o_c: "ndarray[2,float]"):
        """
        Compute for client c the closest object store in S
        """
        return self.argmin(S, [c])

    def argmin(self, S: List[int], C: List[int], get_costs_o_c: "ndarray[2,float]"):
        """
        Compute for clients C the jointly cheapest object store in S
        """
        arg = S[0]
        cost_cur = self.cost(C, [arg])

        for s in S[1:]:
            cost_tmp = self.cost(C, [s])
            if cost_cur > cost_tmp:
                cost_cur = cost_tmp
                arg = s

        return arg
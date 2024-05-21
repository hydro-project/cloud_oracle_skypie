import torch
from dataclasses import dataclass, field

import os
import psutil

from skypie.oracle_impls.oracle_interface import OracleInterface
from skypie.util.my_dataclasses import *
from skypie.util.load_proto import is_decision, load_cost_matrix
from skypie.oracle_impls.oracle_pytorch.minimization_query import MinimizationQuery
from skypie.oracle_impls.oracle_pytorch.directed_drift_query import DirectedDriftQuery

@dataclass
class OracleImplPyTorch(OracleInterface):
    device_query: str = "cpu" # We can query for the optimal scheme using different device types, e.g., "cpu" or gpu ("cuda:0")
    data_type: "torch.dtype" = torch.float64
    threads: int = 0
    compiled: bool = False
    max_batch_size: int = 64
    __queryData: "torch.Tensor" = field(init=False, default_factory=lambda: torch.tensor([])) # Tensor of shape: 1 x schemes x len(costWLHalfplane)
    __oracle: "Oracle" = field(init=False)
    __minimization_query: "MinimizationQuery" = field(init=False)
    __directed_drift_query: "DirectedDriftQuery" = field(init=False)
    
    def prepare_schemes(self, oracle: "Oracle"):
        schemes = oracle.get_schemes()
        self.__oracle = oracle

        # Set number of threads for pytorch
        if self.threads > 0:
            torch.set_num_threads(self.threads)
        else:
            torch.set_num_threads(1)
        
        if self.__oracle.verbose > 1:
            print("Loading schemes into pytorch...")

        # TODO: Make sure to load cost matrix as __planes__!
        # Convert into single matrix
        if len(schemes) > 0:
            if is_decision(schemes[0]):
                #matrix = np.array(load_cost_matrix(schemes))
                matrix = load_cost_matrix(schemes)
            # Is this a raw cost matrix?
            elif (isinstance(schemes, np.ndarray)) or (isinstance(schemes, list) and isinstance(schemes[0], list) and isinstance(schemes[0][0], float)):
                #matrix = np.array(schemes)
                matrix = schemes
            else:
                #matrix = np.array([scheme.costWorkloadHalfplane[:-1] for scheme in schemes])
                matrix = [scheme.costWorkloadHalfplane[:-1] for scheme in schemes]
                print(f"Matrix shape: {matrix.shape}, dtype: {matrix.dtype}, memory size: {matrix.nbytes / 1024 / 1024} MB")
        if self.__oracle.verbose > 1:
            print(f"Loaded schemes. RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

        if self.__oracle.verbose > 1:
            print("Creating tensor...")

        # Wrapping the matrix into another dimension for batching
        shape = torch.Size((1, len(matrix), len(matrix[0])))
        # Create a tensor from the matrix and ship it to the device
        self.__queryData = torch.as_tensor(data=matrix, dtype=self.data_type).view(shape).to(device=self.device_query)

        if self.__oracle.verbose > 1:
            print(f"Loaded tensor. RSS={psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

        # Construct state for querying
        queryData_non_batched = self.__queryData.view(self.__queryData.shape[1], self.__queryData.shape[2])
        self.__minimization_query = MinimizationQuery(planes=queryData_non_batched, device=self.device_query, dtype=self.data_type, compile=self.compiled, max_batch_size=self.max_batch_size)
        # XXX: Be sure that the planes of the query data have intercept as first element and cost -1 as last element!
        self.__directed_drift_query = DirectedDriftQuery(planes=queryData_non_batched, device=self.device_query, dtype=self.data_type, compile=self.compiled)

    def query(self, w: "Workload|List[Workload]", timer: Timer = None) -> "List[Tuple[float, int]]":
        """
        This function queries the oracle for the optimal scheme for the given workload, using pytorch.
        It supports batching.
        It returns the list of the costs and indexes of the optimal schemes.
        """

        if timer is not None:
            timer.continueOverhead()

        # Convert workload into tensor
        workloadVectorLocal = self.__convert_to_tensor(w)
        
        if timer is not None:
            timer.continueComputation()

        # Make sure the batch size fits, reinitialize if necessary
        if workloadVectorLocal.shape[0] > self.max_batch_size:
            queryData_non_batched = self.__queryData.view(self.__queryData.shape[1], self.__queryData.shape[2])
            self.__minimization_query = MinimizationQuery(planes=queryData_non_batched, device=self.device_query, dtype=self.data_type, compile=self.compiled, max_batch_size=self.max_batch_size)

        # Compute cost of each scheme for all workloads and return the index of the cheapest scheme
        values, indices = self.__minimization_query.query(inputs=workloadVectorLocal)

        if timer is not None:
            timer.stop()

        return list(zip(values.squeeze(dim=-1).tolist(), indices.squeeze(dim=-1).tolist()))
    
    def query_directed_drift(self, w: "Workload", drift: "Workload", timer: Timer = None) -> "Tuple[float, List[float], int]":
        """
        This function queries the oracle for the optimal decision for when the given workload drifts in the given direction, using pytorch.
        It does not support batching.
        It returns the distance, the point (workload+cost) of the next optimal decision, and the index of the next optimal decision.
        """

        if timer is not None:
            timer.continueOverhead()

        # Convert workload into tensor
        workloadVectorLocal = self.__convert_to_tensor(w)
        driftLocal = self.__convert_to_tensor(drift)

        # Execute query and get back Tensors
        distance, point, index = self.__directed_drift_query.query(inputs=workloadVectorLocal, drift=driftLocal)

        if timer is not None:
            timer.stop()

        # Convert to native python types
        return distance.item(), point.tolist(), index.item()

    def __convert_to_tensor(self, w: "Workload|List[Workload]"):
        if isinstance(w, Workload):
            w = [w]
        
        # Convert workload into tensor
        if isinstance(w, np.ndarray) and not isinstance(w[0], Workload):
            workloadVectorLocal = torch.tensor(data=w, dtype=self.data_type)
        else:
            workloadVectorLocal = torch.tensor(data=[workload.equation for workload in w], dtype=self.data_type)

        return workloadVectorLocal
        

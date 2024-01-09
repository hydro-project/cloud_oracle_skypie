import torch
import unittest
import time

from skypie.util.dummy_data import load_linear_functions

class DirectedDriftQuery:
    """
    State and functionality for a directed drift query in an oracle implementation for PyTorch.

    Args:
        planes (torch.Tensor): The planes matrix, non-normalized normal vectors with first element for intercept and last element -1 for "height".
        device (torch.device): The device on which the tensors will be allocated.
        dtype (torch.dtype): The desired data type of the tensors.
        compile (bool, optional): Flag indicating whether to compile the directed drift query. Defaults to True.
    """

    def __init__(self, planes: torch.Tensor, device: torch.device, dtype: torch.dtype, compile = True):

        _num_functions, num_params = planes.shape

        # Reference to planes matrix
        self.__planes = planes

        # Preallocate outputs and inputs
        self.__outputs = torch.zeros((1), device=device, dtype=dtype, requires_grad=False)
        self.__outputs_index = torch.zeros((1), device=device, dtype=torch.int64, requires_grad=False)
        self.__output_intersection_point = torch.zeros((num_params-1), device=device, dtype=dtype, requires_grad=False)
        
        self.__input_parameters = torch.empty((num_params), device=device, dtype=dtype, requires_grad=False)
        self.__input_parameters[0] = 1 # Intercept is always 1
        self.__input_parameters[-1] = 0 # Last parameter for cost is set later

        self.__input_drift = torch.empty((num_params), device=device, dtype=dtype, requires_grad=False)
        self.__input_drift[0] = 0 # No drift in intercept
        self.__input_drift[-1] = 0 # XXX: Drift in cost for projection

        self.__compiled = compile

        # Warmup compilation
        if self.__compiled:
            self.__compiled_directed_drift_query()

    def query(self, current_parameter: torch.Tensor, drift: torch.Tensor):
        """
        Computes the distance, the workload+cost, and the index of the next optimal decision given the current parameter,
        a drift, and the planes of all decisions in the oracle.

        Args:
            current_parameter (torch.Tensor): The current parameter.
            drift (torch.Tensor): The drift vector, drift rate per workload parameter (excluding intercept and cost).
            planes (torch.Tensor): The planes matrix, non-normalized normal vectors.

        Returns:
            distance (torch.Tensor): The distance to the next optimal decision, proportional to the input drift vector.
            workload+cost (torch.Tensor): The workload+cost of the next optimal decision.
            next_index (torch.Tensor): The index of the next optimal decision.
        """    
        assert current_parameter.shape[0]+2 == self.__input_parameters.shape[0], f"Input parameter dimension {current_parameter.shape[0]} does not match the expected input dimension {self.__input_parameters.shape[0]-2}"
        assert drift.shape[0]+2 == self.__input_drift.shape[0], f"Input drift dimension {drift.shape[0]} does not match the expected input dimension {self.__input_drift.shape[0]-2}"

        # Set the inputs, skipping the intercept and the last parameter (cost)
        self.__input_parameters[1:-1].copy_(current_parameter)
        self.__input_drift[1:-1].copy_(drift)
        if self.__compiled:
            return self.__compiled_directed_drift_query()
        else:
            return self.__directed_drift_query()
        
    @torch.compile(options={"trace.graph_diagram": False, "trace.enabled": False}, fullgraph=False, dynamic=False)
    def __compiled_directed_drift_query(self):
        return self.__directed_drift_query()
    
    def __directed_drift_query(self):

        # Get current minimum decision, skipping cost dimension
        curr = torch.matmul(self.__planes[:,:-1], self.__input_parameters[:-1]).min(dim=0)
        index  = curr.indices
        value  = curr[0]

        # Point on plane of current minimum decision
        self.__input_parameters[-1] = value

        # Project drift onto current minimum decision
        # Scale for non-normalized planes
        rescale_drift = self.__input_drift @ self.__planes[index]
        rescale_plane = self.__planes[index] @ self.__planes[index]
        scale_plane = rescale_drift / rescale_plane
        #scale_plane = torch.dot(self.__input_drift, self.planes[index])

        projected_drift = self.__input_drift - scale_plane * self.__planes[index]
        #projected_drift = self.__input_drift - (torch.dot(self.__input_drift, self.planes[index])) * self.planes[index]

        # Ray shooting from current minimum point in direction of projected drift
        #torch.min(((self.planes.matmul(-1*self.__input_parameters)) / (self.planes.matmul(projected_drift))), dim=0, out=(self.outputs, self.outputs_index))

        # Implementation of the above ray shooting but with masking out the current optimum
        part_0 = self.__planes[0:index]
        part_1 = self.__planes[index+1:]

        if part_0.shape[0] > 0:
            part_0_nom = (part_0 @ self.__input_parameters)
            part_0_denom = (part_0 @ projected_drift)
            part_0_min = torch.min(part_0_nom*-1 / part_0_denom, dim=0)

        if part_1.shape[0] > 0:
            part_1_nom = (part_1 @ self.__input_parameters)
            part_1_denom = (part_1 @ projected_drift)
            part_1_min = torch.min(part_1_nom*-1 / part_1_denom, dim=0)

        if part_0.shape[0] > 0 and part_0_min[0] < part_1_min[0]:
            self.__outputs = part_0_min.values
            self.__outputs_index = part_0_min.indices
        else:
            self.__outputs = part_1_min.values
            self.__outputs_index = part_1_min.indices + index + 1

        # Distance to intersection point (without intercept)
        self.__output_intersection_point = self.__input_parameters[1:] + self.__outputs * projected_drift[1:]
        # Adjust to scale of input drift without intercept and cost/height
        self.__outputs = torch.norm(self.__output_intersection_point[:-1] - self.__input_parameters[1:-1], dim=0) / torch.norm(self.__input_drift, dim=0)

        return self.__outputs, self.__output_intersection_point, self.__outputs_index
    
## Code for testing and debugging
def test_drift_query(*,test: unittest.TestCase=None, compile=False, device=torch.device('cpu')):
    num_functions = 3
    num_params = 1
    data = load_linear_functions(device=device, num_functions=num_functions, num_params=num_params+1, dtype=torch.float32, as_planes=True, random=False)

    # Set intercepts to 0
    data[:, 0] = 0
    # Set last parameter to -1
    data[:, -1] = -1

    optimal_index = 0
    optimal_val = torch.full(size=(1,), fill_value=1)
    #data[optimal_index] *= optimal_val
    data[optimal_index, 1] = optimal_val

    # Set intercept and slope of next optimal decision
    next_optimal_index = 1
    next_optimal_val = torch.full(size=(1,), fill_value=0.5)
    #data[next_optimal_index] *= next_optimal_val
    data[next_optimal_index, 0:2] = next_optimal_val

    # Set high  intercept for the rest of the decisions
    data[2:, 0] = 100

    current_parameter = torch.zeros((num_params,), device=device, dtype=torch.float32)

    drift = torch.full(size=(num_params,), fill_value=0.5)
    expected_distance = torch.Tensor([2.0])
    
    query = DirectedDriftQuery(planes=data, device=device, dtype=torch.float32, compile=compile)

    start_time = time.time()
    distance, point, idx = query.query(current_parameter=current_parameter, drift=drift)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")

    idx = idx.item()

    if test is None:
        print(f"{idx} {distance}")
    else:
        test.assertEqual(idx, next_optimal_index, f"Expected optimal index {next_optimal_index} but got {idx}")
        test.assertTrue(torch.isclose(distance, expected_distance), f"Expected optimal value {expected_distance} but got {distance.item()}")

class TestDirectedDriftQuery(unittest.TestCase):
    def test_directed_drift_query(self):
        test_drift_query(test=self)

    def test_directed_drift_query_compiled(self):
        test_drift_query(test=self, compile=True)

if __name__ == "__main__":
    test_drift_query()
    test_drift_query(compile=True)
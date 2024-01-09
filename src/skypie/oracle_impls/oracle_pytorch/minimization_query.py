import torch
import unittest

from skypie.util.dummy_data import load_linear_functions

class MinimizationQuery:
    def __init__(self, planes: torch.Tensor, device: torch.device, dtype: torch.dtype, max_batch_size: int = 1, compile = True):
        
        _num_functions, num_params = planes.shape

        self.planes = planes

        # Preallocate outputs and inputs
        self.outputs = torch.zeros((max_batch_size, 1), device=device, dtype=dtype, requires_grad=False)
        self.outputs_index = torch.zeros((max_batch_size, 1), device=device, dtype=torch.int64, requires_grad=False)
        self.__inputs = torch.empty((max_batch_size, num_params), device=device, dtype=dtype, requires_grad=False)

        self.compiled = compile

        # Warmup compilation
        if self.compiled:
            self.__compiled_minimization_query()

    @torch.compile(options={"trace.graph_diagram": False, "trace.enabled": False}, fullgraph=False, dynamic=False)
    def __compiled_minimization_query(self):
        return self.__minimization_query()

    def __minimization_query(self):
        # Compute cost for each parameter and sum them and take the minimum
        planes_batch = self.planes.view(1, self.planes.shape[0], self.planes.shape[1])
        inputs_batch = self.__inputs.view(self.__inputs.shape[0], self.__inputs.shape[1], 1)
        intermediate = torch.matmul(planes_batch, inputs_batch)
        torch.min(intermediate, dim=1, keepdim=False, out=(self.outputs, self.outputs_index))

        return (self.outputs, self.outputs_index)
    
    def query(self, inputs: torch.Tensor):
        assert inputs.shape[0] <= self.__inputs.shape[0], f"Batch size {inputs.shape[0]} is larger than the maximum batch size {self.__inputs.shape[0]}"
        assert inputs.shape[1] == self.__inputs.shape[1], f"Input dimension {inputs.shape[1]} does not match the expected input dimension {self.__inputs.shape[1]}"
        # Set the inputs
        self.__inputs.copy_(inputs)
        if self.compiled:
            return self.__compiled_minimization_query()
        else:
            return self.__minimization_query()
        
class TestMinimizationQuery(unittest.TestCase):
    def _execute_test(self, compile_query):
        batch_size = 2
        device = torch.device('cpu')
        data = load_linear_functions(device=device, num_functions=10, num_params=10, dtype=torch.float32, as_planes=True, random=False)

        optimal_index = 3
        optimal_val = torch.full(size=(1,), fill_value=0.5)
        data[optimal_index] *= optimal_val
        
        query = MinimizationQuery(planes=data, max_batch_size=batch_size, device=device, dtype=torch.float32, compile=compile_query)

        res = query.query(inputs=torch.ones((batch_size, 11), device=device, dtype=torch.float32))

        for batch_i in range(batch_size):
            idx = res[1][batch_i].item()
            self.assertEqual(idx, optimal_index, f"Expected optimal index {optimal_index} but got {idx}")
            self.assertTrue(torch.isclose(res[0][batch_i], optimal_val), f"Expected optimal value {optimal_val} but got {res[0][batch_i].item()}")

    def test_minimization_query(self):
        self._execute_test(compile_query=False)

    def test_minimization_query_compiled(self):
        self._execute_test(compile_query=True)
import torch

"""
This file contains the stochastic simulation query for the pytorch oracle
as implemented for the CIDR 2024 paper. It has not been integrated into the oracle,
since this kind of simulation is a higher order function that should be implemented on top of the query() function of the oracle.
"""

def init_query(*, planes: torch.Tensor, planes_other: torch.Tensor, batch_size: int, num_params: int, device: torch.device, dtype: torch.dtype):
    # Verify that planes have the correct number of parameters
    assert planes.shape[1] == num_params+1
    assert planes_other.shape[1] == num_params+1
    
    outputs = torch.zeros((batch_size, 1), device=device, dtype=dtype)
    
    # Allocate inputs without additional dimension. This is deferred to the query.
    inputs = torch.empty((batch_size, planes.shape[1]), device=device, dtype=dtype)

    # Simulate a historic trace
    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    mean=1000
    inputs.normal_(mean=mean, std=100)

    return (outputs, inputs)

def load_random_inputs(inputs: torch.Tensor):
        return inputs.exponential_()

@torch.compile(options={"trace.graph_diagram": False, "trace.enabled": False}, fullgraph=False, dynamic=False)
def simulation_query(*, planes: torch.Tensor, planes_other: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor):
    """
    Perform a simulation query using the given parameters.

    Args:
        planes (torch.Tensor): The new planes tensor.
        planes_other (torch.Tensor): The other planes tensor.
        inputs (torch.Tensor): The inputs tensor.
        outputs (torch.Tensor): The outputs tensor.

    Returns:
        torch.Tensor: The outputs with the savings of the new planes.
    """

    inputs_batch = inputs.view(inputs.shape[0], inputs.shape[1], 1)

    planes_batch = planes.view(1, planes.shape[0], planes.shape[1])
    intermediate = torch.matmul(planes_batch, inputs_batch)
    min_new = intermediate.min(dim=1, keepdim=False)

    planes_batch_other = planes_other.view(1, planes_other.shape[0], planes_other.shape[1])
    intermediate_other = torch.matmul(planes_batch_other, inputs_batch)
    min_other = intermediate_other.min(dim=1, keepdim=False)

    #torch.sub(min_other.values, min_new.values, out=outputs)
    torch.divide(min_new.values, min_other.values, out=outputs)

    # Compute quantiles by sorting
    percentiles = [0.025, 0.50, 0.975]
    sorted_outputs = torch.sort(outputs, dim=0, descending=False).values

    # Compute indexes of percentiles
    indexes = (sorted_outputs.shape[0]-1) * torch.tensor(percentiles, device=outputs.device, dtype=torch.float32)
    indexes = indexes.round().long()
    percentile_values = sorted_outputs[indexes]

    # Flatten the percentile values to a single dimension
    percentile_values = percentile_values.view((3,))

    return percentile_values.cpu().tolist()
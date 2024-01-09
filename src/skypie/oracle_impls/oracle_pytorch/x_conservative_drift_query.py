import torch

"""
This is the conservative drift query which computes the distance to any next optimal decision.
This is the original implementation for CIDR'24 and has not yet been integrated into the oracle.
"""

@torch.compile(options={"trace.graph_diagram": False, "trace.enabled": False}, fullgraph=False, dynamic=False)
def conservative_drift_query(*, current_parameter, planes):
    """
    Computes the distance and index of the next optimal decision in any drift direction
    given the current parameter and the planes of all decisions in the oracle.

    Args:
        current_parameter (torch.Tensor): The current parameter.
        planes (torch.Tensor): The planes matrix.

    Returns:
        distance (torch.Tensor): The distance to the next optimal decision.
        next_index (torch.Tensor): The index of the next optimal decision.
    """
    
    # Get current minimum function
    curr = torch.matmul(planes, current_parameter).min(dim=0)
    index  = curr.indices
    value  = curr[0]

    # Point on plane of current minimum function
    current_parameter[-1] = value

    ## Mask out current minimum function
    # Copy intercept of current minimum function
    temp = planes[index][0]
    planes[index][0] = torch.finfo(planes.dtype).max

    # Other, plane, start
    # *, current_parameter, drift, planes
    
    # Step 15: Compute alpha (dot product of Other and plane)
    alpha = torch.matmul(planes, planes[index])

    # Step 16: Compute V'
    V_prime = alpha.outer(planes[index]) - planes

    # Step 17: Compute alpha'
    alpha_prime = torch.sqrt(1 - alpha ** 2)

    # Step 18: Update V
    V = V_prime / alpha_prime.unsqueeze(1)

    # Step 19: Compute distance
    distance = -torch.matmul(planes, planes[index]) / (torch.sum(planes * V, axis=1))

    # Step 20: Find the minimum distance and its index
    min_dist, min_idx = torch.min(distance, dim=0)

    # Restore plane of current minimum function
    planes[index][0] = temp.item()

    return min_dist, min_idx

def load_data(*, num_functions, num_parameters, device, dtype):
    Other = torch.full((n, m), 1/3, dtype=dtype, device=device)  # n x m matrix of 1/3s
    # m vector of 1/3s
    plane = torch.full((m,), 1/3, dtype=dtype, device=device)
    # m vector of 0s
    start = torch.zeros((m,), dtype=dtype, device=device)

if __name__ == "__main__":
    dtype = torch.float16
    device = torch.device('cpu')
    # If cuda is available, use it
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    # If MPS is available, use it
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # Example usage
    n, m = 10000, 3  # Example dimensions
    Other = torch.full((n, m), 1/3, dtype=dtype, device=device)  # n x m matrix of 1/3s
    # m vector of 1/3s
    plane = torch.full((m,), 1/3, dtype=dtype, device=device)
    # m vector of 0s
    start = torch.zeros((m,), dtype=dtype, device=device)

    min_dist, min_idx = solve_relaxed_problem(Other, plane, start)
    print("Minimum Distance:", min_dist)
    print("Index of Minimum Distance:", min_idx)

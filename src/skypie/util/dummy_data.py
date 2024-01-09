import torch

def load_linear_functions(*, num_functions, num_params, coeff_min = 0, coeff_max = 1, dtype, device, as_planes=False, random=False):
    
    # Add one parameter for the additional dimension of planes
    if as_planes:
        num_params += 1

    # Shape of the tensor: (num_functions, num_params)
    shape = (num_functions, num_params)

    functions_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

    if random:
        # Fill with random values
        functions_tensor.uniform_(coeff_min, coeff_max)
    else:
        # Fill with coefficients summing to 1
        functions_tensor.fill_(1/num_params)

    return functions_tensor
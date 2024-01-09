import torch
import time

def compute_pareto_frontier_math(window_points, *, device=torch.device('cpu')):
    window_points_on_device = window_points.to(device)
    n_points = window_points.shape[0]
    
    zeros = torch.zeros_like(window_points_on_device[0])
    dominated_mask = torch.zeros(n_points, dtype=torch.bool)
    # For each point
    for i in range(n_points):
        point = window_points_on_device[i]
        res =  window_points_on_device - point
        dominated_vec = ((res) < zeros)
        dominated_mask[i] = dominated_vec.all(dim=1).any()
        #print(f"res: {res}, vec: {dominated_vec}, dominated:{dominated_mask[i]}")
    
    return ~dominated_mask

def compute_pareto_frontier_std(window_points, *, device=torch.device('cpu')):
    window_points_on_device = window_points.to(device)
    window_repeated = window_points.unsqueeze(1).repeat(1, window_points_on_device.shape[0], 1)
    window_repeated_T = window_repeated.transpose(0, 1)

    dominance_mask = (window_repeated <= window_repeated_T).all(dim=-1) & (window_repeated < window_repeated_T).any(dim=-1)
    dominated_mask = dominance_mask.any(dim=0)

    #pareto_frontier = window_points[~dominated_mask]
    #return pareto_frontier
    return ~dominated_mask

def compute_pareto_frontier(*args, math=False, **kwargs):
    if math:
        return compute_pareto_frontier_math(*args, **kwargs)
    else:
        return compute_pareto_frontier_std(*args, **kwargs)

def sliding_window(points, window_size, shift):
    n_points = points.shape[0]
    #windows = []

    for i in range(shift, n_points, window_size):
        yield points[i:i + window_size]
        #windows.append(window_points)
    
    #return windows

def shifted_sliding_window_pareto_frontier(points,*, window_size, shift, **kwargs):
    n_points = points.shape[0]

    pareto_frontiers = [compute_pareto_frontier(window, **kwargs) for window in sliding_window(points, window_size, shift)]
    
    first_window_points = points[:window_size]
    last_window_points = points[n_points-window_size:]
    whole_window_points = torch.cat((last_window_points, first_window_points), dim=0)

    whole_window_pareto_frontier = compute_pareto_frontier(whole_window_points, **kwargs)
    pareto_frontiers.append(whole_window_pareto_frontier)

    pareto_frontier = torch.cat(pareto_frontiers)

    return pareto_frontier

def benchmark(function, *args, **kwargs):
    start_time = time.time()
    function(*args, **kwargs)
    end_time = time.time()

    return end_time - start_time

# Only execute if main
if __name__ == "__main__":
    device = torch.device('cuda:0')
    dtype = torch.float32
    points = torch.tensor([[2.5, 3.0, 2.5], [1.0, 2.0, 3.0], [2.0, 2.5, 2.0]], dtype=dtype, device=device)
    frontier = compute_pareto_frontier_std(points, device=device)
    print(frontier)
    frontier2 = compute_pareto_frontier_math(points, device=device)
    print(frontier2)
    print(f"Is equal: {frontier.to('cpu') == frontier2.to('cpu')}")

    points_counts = [100, 1000, 10000]
    #points_counts = [10**5, 10**6, 10**7, 10**8]
    points_counts = [10**2, 5*10**2,10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]
    dimensions_counts = [180]
    window_size = 200
    #math = [False, True]
    maths = [True]

    for math in maths:
        for points_count in points_counts:
            for dimension_count in dimensions_counts:
                points = torch.rand(points_count, dimension_count, dtype=dtype)
                #execution_time = benchmark(compute_pareto_frontier, points, device=device)
                execution_time = benchmark(compute_pareto_frontier, points, device=device, math=math)
                #execution_time = benchmark(shifted_sliding_window_pareto_frontier, points, window_size=window_size, shift=0, device=device, math=math)
                print(f'Execution time with math={math} for {points_count} points and {dimension_count} dimensions: {execution_time} seconds')
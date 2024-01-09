import os

from skypie.experiments.experiment import Experiment

def query_batching(*, precomputation_root_dir: str, precomputation_file_base_name: str = "experiment.json", output_file_name: str="", cuda_device: str = "cuda:0"):

    if not output_file_name:
        output_file = "query_batching_result.pandas.pickle"
        root_dir = os.path.join(os.getcwd(), "results", "query_batching")
    else:
        root_dir = os.path.dirname(output_file_name)
        output_file = os.path.basename(output_file_name)

    args = dict(
        experiment_dir = os.path.join(root_dir, "cpu"),
        output_file = output_file,
        device = "cpu",
        threads = 40,
        optimizer = [], # Only use the optimizer of the optimal decisions
        precision = "float32",
        batch_sizes = [1, 10, 100, 1000]
    )

    args_gpu = None
    if cuda_device != "cpu":
        args_gpu = args.copy()
        args_gpu["experiment_dir"] = os.path.join(root_dir, "gpu")
        args_gpu["device"] = cuda_device

    experiments = [Experiment.from_precomputation(args=args, args_gpu=args_gpu, precomputation_root_dir=precomputation_root_dir, precomputation_file_base_name=precomputation_file_base_name)]

    # Run the experiments
    Experiment.run_all(experiments=experiments, output_file=os.path.join(root_dir, output_file))
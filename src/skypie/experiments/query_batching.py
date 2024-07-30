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
        batch_sizes = [1, 10, 100, 1000],
        no_workloads = 1000,
        add_optimizers_from_input = True,
    )
    args_list = [args]

    if cuda_device != "cpu":
        args_gpu = args.copy()
        args_gpu["experiment_dir"] = os.path.join(root_dir, "gpu")
        args_gpu["device"] = cuda_device

        args_list.append(args_gpu)

    # Filter for the largest scenario and replication factor 3
    experiments = [ *Experiment.from_precomputation(args_list=args_list, precomputation_root_dir=precomputation_root_dir, precomputation_file_base_name=precomputation_file_base_name)]

    experiments = [e for e in experiments if e.min_replication_factor == 3 and e.precomputation_details["region_selector"] == "azure|aws"]

    if len(experiments) == 0:
        raise ValueError("No precomputation files found for scenario aws-azure and replication factor 3")

    # Run the experiments
    Experiment.run_all(experiments=experiments, output_file=os.path.join(root_dir, output_file))
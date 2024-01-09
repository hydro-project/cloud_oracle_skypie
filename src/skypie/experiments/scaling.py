import os

from skypie.experiments.experiment import Experiment

def scaling(*, precomputation_root_dir: str, precomputation_file_base_name: str = "experiment.json", output_file_name: str="", cuda_device: str = "cuda:0"):
    # Create a list of experiments for each experiment file in the directory of precomputed files
    # Run each experiment for each optimizer: ILP, Candidates, optimizers from file

    if not output_file_name:
        output_file = "scaling_result.pandas.pickle"
        root_dir = os.path.join(os.getcwd(), "results", "query_scaling")
    else:
        root_dir = os.path.dirname(output_file_name)
        output_file = os.path.basename(output_file_name)

    args = dict(
        experiment_dir = os.path.join(root_dir, "cpu"),
        output_file = output_file,
        no_workloads = 1000,
        optimizer = [], #["ILP"],
        device = "cpu",
        threads = 40,
        precision = "float32",
    )
    args_list = [args]

    if cuda_device != "cpu":
        args_gpu = args.copy()
        args_gpu["experiment_dir"] = os.path.join(root_dir, "gpu")
        args_gpu["optimizer"] = []
        args_gpu["device"] = cuda_device
        #fixed_args_gpu["precision"] = "float16"

        args_list.append(args_gpu)

    # Create an experiment for each experiment.json within the precomputation_root_dir
    experiments = list([*Experiment.from_precomputation(args_list=args_list, precomputation_root_dir=precomputation_root_dir, precomputation_file_base_name=precomputation_file_base_name)])

    # Adjust the number of workload for large problems
    for experiment in experiments:
        if experiment.min_replication_factor > 2 and \
            "azure" in experiment.precomputation_file and \
            "aws" in experiment.precomputation_file:

            experiment.no_workloads = 10

        if experiment.device == "cpu" and "aws-eu" == experiment.precomputation_details["region_selector"]:
            experiment.optimizer = ["Candidates", "ILP"]

    print(f"Running {len(experiments)} experiments")

    # Sort experiments by the replication factor
    experiments.sort(key=lambda x: x.min_replication_factor)

    # Run CPU experiments first
    experiments.sort(key=lambda x: x.device, reverse=False)

    print("Running experiments for files:")
    for e in experiments:
        print(f"\t{e.precomputation_file}")

    # Run the experiments
    Experiment.run_all(experiments=experiments, output_file=os.path.join(root_dir, output_file))
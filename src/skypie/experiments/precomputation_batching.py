import os
import copy

from skypie.experiments.experiment import Experiment

def precomputation_batching(*, precomputation_root_dir: str, precomputation_file_base_name: str = "experiment.json", output_file_name: str="", cuda_device: str = "cuda:0"):

    if not output_file_name:
        output_file = "precomputation_batching_batching_result.pandas.pickle"
        root_dir = os.path.join(os.getcwd(), "results", "precomputation_batching")
    else:
        root_dir = os.path.dirname(output_file_name)
        output_file = os.path.basename(output_file_name)

    args = dict(
        experiment_dir = os.path.join(root_dir, "cpu"),
        output_file = output_file,
        #no_workloads = 1000,
        optimizer = ["Candidates"],
        device = "cpu",
        threads = 40,
        precision = "float32",
    )
    args_list = [args]

    #experiments = [Experiment.from_precomputation(args=args, precomputation_root_dir=precomputation_root_dir, precomputation_file_base_name=precomputation_file_base_name)]
    if cuda_device != "cpu":
        args_gpu = copy.deepcopy(args.copy())
        args_gpu["experiment_dir"] = os.path.join(root_dir, "gpu")
        args_gpu["optimizer"] = []
        args_gpu["device"] = cuda_device

        args_list.append(args_gpu)

    # Create an experiment for each experiment.json within the precomputation_root_dir
    experiments = [*Experiment.from_precomputation(args_list=args_list, precomputation_root_dir=precomputation_root_dir, precomputation_file_base_name=precomputation_file_base_name)]

    # Filter out experiment used in the paper:
    experiments = [e for e in experiments if e.min_replication_factor == 3 and e.precomputation_details["region_selector"] == "azure|aws"]
    
    if len(experiments) == 0:
        raise ValueError("No precomputation files found for scenario aws-azure and replication factor 3")

    print("Running experiments for files:")
    for e in experiments:
        print(f"\t{e.precomputation_file}")

    # Run the experiments
    Experiment.run_all(experiments=experiments, output_file=os.path.join(root_dir, output_file))
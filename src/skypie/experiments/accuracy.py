import os
import pandas as pd

from skypie.experiments.experiment import Experiment

def accuracy(*, precomputation_root_dir: str, precomputation_file_base_name: str = "experiment.json", output_file_name: str = "", cuda_device: str = "cuda:0"):
    # Create a list of experiments for each experiment file in the directory of precomputed files
    # Run each experiment for each optimizer: ILP, Candidates, optimizers from file

    if not output_file_name:
        output_file = "accuracy_result.pandas.pickle"
        root_dir = os.path.join(os.getcwd(), "results", "query_accuracy")
    else:
        root_dir = os.path.dirname(output_file_name)
        output_file = os.path.basename(output_file_name)

    fixed_args = dict(
        experiment_dir = os.path.join(root_dir, "cpu"),
        #precomputation_file: str
        output_file = output_file,
        no_workloads = 1000,
        #optimizers.extend(createOptimizer(optimizer=optimizer, args=args.__dict__))
        optimizer = ["Candidates", "ILP"],
        device = "cpu",
        threads = 40,
        precision = "float32",
    )

    # Create an experiment for each experiment.json within the precomputation_root_dir
    # Traverse the directory tree
    experiments = []
    for path, _, files in os.walk(precomputation_root_dir):
        for file in files:
            if file == precomputation_file_base_name:
                # Create an experiment for each optimizer
                
                experiments.append(Experiment(
                    precomputation_file = os.path.join(path, precomputation_file_base_name),
                    **fixed_args
                ))

    # Adjust the number of workload for large problems
    for experiment in experiments:
        if experiment.min_replication_factor > 2 and \
            "azure" in experiment.precomputation_file and \
            "aws" in experiment.precomputation_file:

            experiment.no_workloads = 10

    print(f"Running {len(experiments)} experiments")

    df = pd.DataFrame()

    for experiment in experiments:
        print(f"Running experiment:", experiment)
        results = experiment.run()

        for result in results:
            result.update( {f"exp_{k}":v for k,v in experiment.__dict__.items()} )
        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
        df.to_pickle(os.path.join(root_dir, "accuracy_result.pandas.pickle"))
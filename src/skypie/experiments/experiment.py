import pandas as pd
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Tuple
import json

import pkg_resources
from skypie.experiments.query_random import query_random
from skypie.util.my_dataclasses import OptimizerType

@dataclass
class Experiment:
    precomputation_file: str
    output_file: str
    no_workloads: int = 5000
    workload_range: Tuple[float, float] = (0, 1000.0)
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    query_step_size: int = 0
    query_skip: int = 0
    no_warmup: bool = False
    translate_opt_schemes: bool = False
    skip_workload_results: bool = False
    experiment_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "experiments"))
    name: "str|None" = None
    impl_args: dict = field(init=False, default_factory=dict)
    experiment_dir_full: str = "" # This is set in __post_init__
    verbose: int = 0
    add_optimizers_from_input: bool = True
    min_replication_factor: int = field(init=False)
    no_strict_replication: bool = False
    network_price_file: "str|None" = None
    storage_price_file: "str|None" = None
    input_file: str = field(init=False)
    scenario_name: str = "tier_advise/replication_factor/relative=0/relative=0"
    optimizer: List["OptimizerType|str"] = field(default_factory=list)
    workload_seed: int = 42
    threads: int = 40
    precision: str = "float32"
    device: str = "cpu"
    precomputation_details: "dict" = field(init=False, default_factory=dict)

    def __post_init__(self):

        # Load precomputation data
        with open(self.precomputation_file, "r") as f:
            precom = json.load(f)
            self.precomputation_details = precom

        if not self.network_price_file:
            self.network_price_file = pkg_resources.resource_filename(__name__.split(".")[0], "data/network_cost_v2.csv")

        if not self.storage_price_file:
            self.storage_price_file = pkg_resources.resource_filename(__name__.split(".")[0], "data/storage_pricing.csv")

        self.min_replication_factor = int(precom["replication_factor"])

        # Assuming "stats.proto.bin" is the name of main precomputed file
        self.input_file = os.path.join(os.path.dirname(self.precomputation_file), "stats.proto.bin")

        # Create a translation table that replaces all unfriendly characters with -
        unfriendly_chars = ["|", "*", " ", "(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "<", ">", "/", "\\", "?", "'", "\"", "\n", "\t", "\r", "\v", "\f"]
        translation_table = str.maketrans({c: "-" for c in unfriendly_chars})

        # Use the translation table to replace all unfriendly characters
        friendly_region = precom["region_selector"].translate(translation_table)
        friendly_object_store = precom["object_store_selector"].translate(translation_table)
        if len(friendly_object_store) > 0:
            friendly_region_and_object_store = f"{friendly_region}-{friendly_object_store}"
        else:
            friendly_region_and_object_store = friendly_region

        # Create the name of the experiment
        paths = ([self.name] if self.name is not None else []) + \
            [friendly_region_and_object_store , str(self.min_replication_factor), str(precom["redundancy_elimination_workers"])]
        self.experiment_dir_full = os.path.join(self.experiment_dir, *paths)
        
        self.output_file = os.path.join(self.experiment_dir_full, self.output_file)

        # Set arguments for optimizer implementations
        self.impl_args = {
            ## Mosek arguments
            "networkPriceFile": self.network_price_file,
            "storagePriceFile": self.storage_price_file,
            "noStrictReplication": self.no_strict_replication,
            # Set the minimum replication factor from precomputation 
            "minReplicationFactor": self.min_replication_factor,
            ## Torch arguments
            "precision": self.precision,
            "torchDeviceRayShooting": self.device,
            "threads": self.threads,
            "max_batch_size": max(self.batch_sizes)
        }

    def run(self):
        """
        Run the experiment: Query the Oracle for random workload according to the experiment parameters.

        Returns:
            results (dict): The results of the experiment.
        """
        os.makedirs(self.experiment_dir_full, exist_ok=True)

        # Write experiment parameters to json file
        #with open(os.path.join(self.experiment_dir_full, "experiment.json"), "w") as f:
        #    json.dump(self.__dict__, f, indent=4)
        
        exp_args = self.__dict__.copy()
        del exp_args["impl_args"]

        results = query_random(
            inputFileName = self.input_file,
            scenarioName=self.scenario_name,
            addOptimizersFromInput=self.add_optimizers_from_input,
            optimizer=self.optimizer.copy(),
            workloadSeed=self.workload_seed,
            workloadRange=self.workload_range,
            noWorkloads=self.no_workloads,
            queryStepSize=self.query_step_size,
            querySkip=self.query_skip,
            no_warmup=self.no_warmup,
            translateOptSchemes=self.translate_opt_schemes,
            output_file=self.output_file,
            skipWorkloadResults=self.skip_workload_results,
            batchSizes=self.batch_sizes,
            implArgs=self.impl_args,
            verbose=self.verbose,
            exp_args=exp_args
        )

        return results

    @staticmethod
    def run_all(*, experiments: List["Experiment"], output_file: str):
        """
        Run all experiments and save the results to a pickle file.

        Args:
            experiments (List[Experiment]): List of Experiment objects to run.
            output_file (str): Path to the output pickle file.

        Returns:
            pd.DataFrame: DataFrame containing the results of all experiments.
        """
        df = pd.DataFrame()

        for experiment in experiments:
            print(f"Running experiment:", experiment)

            results = experiment.run()

            #for result in results:
            #    result.update( {f"exp_{k}":v for k,v in experiment.__dict__.items()} )

            df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
            df.to_pickle(output_file)

            print("Results:")
            print(df)

        return df

    @classmethod
    def from_precomputation(cls, *, args_list: List[Dict[str,Any]], precomputation_root_dir: str, precomputation_file_base_name: str = "experiment.json"):
        """
        Create instances of the Experiment class from precomputed data.

        Args:
            args_list (List[Dict[str,Any]]): A list of dictionaries containing arguments for creating Experiment instances.
            precomputation_root_dir (str): The root directory where the precomputed data is stored.
            precomputation_file_base_name (str, optional): The base name of the precomputation file. Defaults to "experiment.json".

        Yields:
            Experiment: An instance of the Experiment class.

        """
        # Traverse the directory tree
        if not os.path.exists(precomputation_root_dir):
            raise FileNotFoundError(f"Directory {precomputation_root_dir} not found")
        
        has_precomputation_file = False
        for path, _, files in os.walk(precomputation_root_dir):
            for file in files:
                if file == precomputation_file_base_name:
                    has_precomputation_file = True
                    
                    precomputation_file = os.path.join(path, precomputation_file_base_name)

                    for args in args_list:
                        yield cls(precomputation_file = precomputation_file, **args)
        
        if not has_precomputation_file:
            raise FileNotFoundError(f"Precomputation file {precomputation_file_base_name} not found in {precomputation_root_dir}")

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List
import copy

from skypie.experiments.experiment import Experiment
from skypie.experiments.query_trace import query_trace

@dataclass
class Trace:
    access_set_trace_file: str
    object_trace_file: str
    percent_remote: float

@dataclass
class RealTraceExperiment(Experiment):
    trace_file: str = None
    percent_remote: float = None
    access_set_region_mapping: Dict[str,int] = None
    filter_timestamp_by_date: str = ""
    size_scale: float = 1
    ingress_scale: float = 1
    egress_scale: float = 1
    rescale: float = 1
    skip_loaded_optimizers: bool = False

    def __post_init__(self):
        super().__post_init__()

        if not self.trace_file:
            raise ValueError("trace_file must be set")
        
        if self.percent_remote is None:
            raise ValueError("percent_remote must be set")
        
        if not self.access_set_region_mapping:
            raise ValueError("accessSetRegionMapping must be set")

    def run(self):

        os.makedirs(self.experiment_dir_full, exist_ok=True)

        exp_args = self.__dict__.copy()
        del exp_args["impl_args"]

        trace_args = dict(
            accessSetRegionMapping=self.access_set_region_mapping,
            inputFileName=self.input_file,
            scenarioName=self.scenario_name,
            addOptimizersFromInput=self.add_optimizers_from_input,
            optimizer=self.optimizer,
            noWorkloads=self.no_workloads,
            no_warmup=self.no_warmup,
            translateOptSchemes=self.translate_opt_schemes,
            output_file=self.output_file,
            skipWorkloadResults=self.skip_workload_results,
            batchSizes=self.batch_sizes,
            implArgs=self.impl_args,
            verbose=self.verbose,
            filterTimestampByDate=self.filter_timestamp_by_date,
            inputWorkloads=self.trace_file,
            sizeScale=self.size_scale,
            ingressScale=self.ingress_scale,
            egressScale=self.egress_scale,
            rescale=self.rescale,
            exp_args=exp_args,
            skip_loaded_optimizers=self.skip_loaded_optimizers,
        )

        # Deep copy the arguments
        args_copy = copy.deepcopy(trace_args)

        results = []
        for r in query_trace(**args_copy):
            results.append(r)
        #results = [*query_trace(**args_copy)]

        return results

    @classmethod
    def from_traces(cls, *, traces: List[Trace], args_list_per_object: List[Dict[str, Any]], args_list_access_set: List[Dict[str, Any]], precomputation_root_dir: str, precomputation_file_base_name: str = "experiment.json") -> Iterator["RealTraceExperiment"]:
        for trace in traces:

            # Set arguments for per-object trace
            additional_args = dict(
                trace_file = [trace.object_trace_file],
                percent_remote = trace.percent_remote,
            )
            for args in args_list_per_object:
                args.update(additional_args)

            # Set arguments for access set trace
            additional_args = dict(
                trace_file = [trace.access_set_trace_file],
                percent_remote = trace.percent_remote,
            )
            for args in args_list_access_set:
                args.update(additional_args)

            # Create experiments of all arguments for each file
            args_list = [*args_list_per_object, *args_list_access_set]

            # Forward all results from_precomputation
            yield from cls.from_precomputation(args_list=args_list, precomputation_root_dir=precomputation_root_dir, precomputation_file_base_name=precomputation_file_base_name)  
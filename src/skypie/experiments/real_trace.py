import os
import copy

from skypie.experiments.experiment_real_trace import RealTraceExperiment, Trace

def real_trace(*, precomputation_root_dir: str, precomputation_file_base_name: str = "experiment.json", trace_dir: str, output_file_name: str="", cuda_device: str = "cuda:0"):

    do_baselines = False
    do_spanstore = True
    do_skypie = False

    access_set_region_mapping = [
        "aws-us-east-2:0",
        "aws-us-west-2:1",
        "aws-eu-central-2:2",
        "aws-eu-west-2:3",
        "aws-ap-south-1:4",
        "aws-ap-southeast-2:5",
        "aws-eu-west-1:6",
        "aws-eu-south-1:7",
        "aws-me-central-1:8",
        "aws-eu-north-1:9"
    ]

    # Scaling of metrics in trace to match pricing units
    sizeScale=0.00000000000137
    ingress_scale=0.000000001
    egress_scale=0.000000001

    traces = [
        Trace("access_sets_of_10_regions_worst-case_0p.parquet", "objects_of_10_regions_worst-case_0p.0.parquet", 0.0),
        Trace("access_sets_of_10_regions_worst-case_001p.parquet", "objects_of_10_regions_worst-case_001p.0.parquet", 0.001),
        Trace("access_sets_of_10_regions_worst-case_01p.parquet", "objects_of_10_regions_worst-case_01p.0.parquet", 0.01),
        #Trace("access_sets_of_10_regions_worst-case_02p.parquet", "objects_of_10_regions_worst-case_02p.0.parquet", 0.02),
        #Trace("access_sets_of_10_regions_worst-case_05p.parquet", "objects_of_10_regions_worst-case_05p.0.parquet", 0.05),
        Trace("access_sets_of_10_regions_worst-case_10p.parquet", "objects_of_10_regions_worst-case_10p.0.parquet", 0.1),
        #Trace("access_sets_of_10_regions_worst-case_20p.parquet", "objects_of_10_regions_worst-case_20p.0.parquet", 0.2),
        #Trace("access_sets_of_10_regions_worst-case_50p.parquet", "objects_of_10_regions_worst-case_50p.0.parquet", 0.5),
        Trace("access_sets_of_10_regions_worst-case_100p.parquet", "objects_of_10_regions_worst-case_100p.0.parquet", 1.0),
    ]

    # Set the path for the trace files
    for trace in traces:
        trace.access_set_trace_file = os.path.join(trace_dir, trace.access_set_trace_file)
        trace.object_trace_file = os.path.join(trace_dir, trace.object_trace_file)

    if not output_file_name:
        output_file = "real_trace_result.pandas.pickle"
        root_dir = os.path.join(os.getcwd(), "results", "real_trace")
    else:
        root_dir = os.path.dirname(output_file_name)
        output_file = os.path.basename(output_file_name)

    args_list_per_object = []
    args_list_access_set = []
    args_base = dict(
        #experiment_dir = os.path.join(root_dir, "cpu"),
        output_file = output_file,
        #optimizer = ["ILP", "Kmeans", "Profit-based"],
        device = "cpu",
        threads = 40,
        precision = "float32",
        access_set_region_mapping=access_set_region_mapping,
        size_scale=sizeScale,
        ingress_scale=ingress_scale,
        egress_scale=egress_scale,
        rescale=1.0,
        no_strict_replication=True,
        skip_workload_results=True,
        batch_sizes=[1024],
        no_workloads = 999999999999,
        #no_workloads = 1,
        verbose=0,
        filter_timestamp_by_date="2023-02-20_06:00:00" # %Y-%m-%d_%H:%M:%S
    )

    args_spanstore = copy.deepcopy(args_base)
    args_spanstore["optimizer"] = ["ILP"]
    args_spanstore["experiment_dir"] = os.path.join(root_dir, "spanstore")
    args_spanstore["batch_sizes"] = [1]
    args_spanstore["skip_loaded_optimizers"] = True
    if do_spanstore:
        args_list_access_set.append(args_spanstore)

    args_per_object_baselines = copy.deepcopy(args_base)
    args_per_object_baselines["optimizer"] = ['Kmeans', 'Profit-based']
    args_per_object_baselines["skip_loaded_optimizers"] = True
    args_per_object_baselines["experiment_dir"] = os.path.join(root_dir, "per_object_baselines")
    if do_baselines:
        args_list_per_object.append(args_per_object_baselines)

    args_per_object_skypie = copy.deepcopy(args_base)
    args_per_object_skypie["optimizer"] = []
    args_per_object_skypie["batch_sizes"] = [1024]
    args_per_object_skypie["experiment_dir"] = os.path.join(root_dir, "per_object_skypie")
    # Accelerate with cuda for now
    args_per_object_skypie["device"] = cuda_device
    if do_skypie:
        args_list_per_object.append(args_per_object_skypie)


    if cuda_device != "cpu" and False:
        args_per_object_gpu = args_base.copy()
        args_per_object_gpu["experiment_dir"] = os.path.join(root_dir, "per_object_gpu")
        args_per_object_gpu["optimizer"] = []
        args_per_object_gpu["device"] = cuda_device
        args_list_per_object.append(args_per_object_gpu)
        pass

    # For testing only one trace
    #traces = [traces[0]]
    #args_list_per_object = [args_per_object_gpu]
    #args_list_access_set = []

    # Create an experiment for each workload file
    experiments = [*RealTraceExperiment.from_traces(args_list_per_object=args_list_per_object, args_list_access_set=args_list_access_set, precomputation_root_dir=precomputation_root_dir, precomputation_file_base_name=precomputation_file_base_name, traces=traces)]

    # Run the experiments
    print(f"Running {len(experiments)} experiment")

    # Run the experiments
    RealTraceExperiment.run_all(experiments=experiments, output_file=os.path.join(root_dir, output_file))
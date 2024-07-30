import argparse
import pickle
import json
import bz2
import pkg_resources
import os

from skypie.util.util import EnhancedJSONDecoder, EnhancedJSONEncoder, compactifyForPrecomputation, createOptimizer, setImplementationArgs
from skypie.util.my_dataclasses import *
from skypie.oracle import Oracle
from skypie.experiments.query_random import query_random
from skypie.experiments.query_trace import query_trace
from skypie.util.load_workloads import load_workloads
from skypie.experiments.benchmarking import benchmarkQuerying
from skypie.experiments.scaling import scaling
from skypie.experiments.real_trace import real_trace
from skypie.experiments.accuracy import accuracy
from skypie.experiments.precomputation_batching import precomputation_batching
from skypie.experiments.query_batching import query_batching

def __main__():

    parser = argparse.ArgumentParser(description="SkyPIE optimizer oracle for computing and querying optimal object repliction for any given workload.")
    parser.add_argument("command", type=str, help="Command to execute: 'compute', 'computeNeighbors', 'queryRandom', 'queryFile','queryAccessSetFile', 'listRegions', 'toPickle', 'toJson'.")
    parser.add_argument("input", type=str, help="Input file name.")
    parser.add_argument("--scenario", type=str, help="Scenario name.", default="default")
    parser.add_argument("--verbose", type=int, help="Verbose level.", default=0)
    parser.add_argument("--validate", help="Validate the computed optimal repliction schemes/partitions.", action='store_true')
    parser.add_argument("--dump", help="Dump the input for cdd.", action='store_true')
    #parser.add_argument("--expectedRedundant", type=int, nargs='+', help='<Required> Set flag')
    parser.add_argument("--optimizer", type=str, nargs='+', help="Optimizer to use. Either 'Lrs' or type of Mosek's optimizer: 'InteriorPoint', 'PrimalSimplex', 'Free'. Also 'Candidates' is a valid choice for querying.", default=[])
    parser.add_argument("--useClarkson", type=str, nargs='+', help="Use Clarkson's algorithm for computing the optimal repliction schemes/partitions.", default=["False"])
    parser.add_argument("--useGPU", type=str, nargs='+', help="Use GPU for computing the optimal repliction schemes/partitions.", default=["False"])
    parser.add_argument("--torchDeviceRayShooting", type=str, help="Torch device for ray shooting.", default="cpu")
    parser.add_argument("--repairInputData", help="Repair input data.", action="store_true")
    parser.add_argument("--output", type=str, help="Output file name.")
    parser.add_argument("--noWorkloads", type=int, help="Number of workloads to generate.", default=1024)
    parser.add_argument("--workloadSeed", type=int, help="Seed for generating workloads.", default=42)
    parser.add_argument("--workloadRange", type=int, nargs=2, help="Range of workload sizes.", default=[0, 100])
    parser.add_argument("--batchSizes", type=int, nargs='+', help="Batch sizes for querying.", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--networkPriceFile", type=str, help="Network price file name.", default=None)
    parser.add_argument("--storagePriceFile", type=str, help="Storage price file name.", default=None)
    parser.add_argument("--optimizerThreads", type=int, help="Number of threads for the optimizer.", default=10)
    parser.add_argument("--nonnegative", help="Use nonnegative constraints.", action='store_true', default=False)
    parser.add_argument("--useCandidates", help="Use candidates for querying.", action='store_true')
    parser.add_argument("--addOptimizersFromInput", help="Discover additional optimizers from file.", action='store_true')
    parser.add_argument("--inputWorkloads", type=str, nargs="+", help="List of input workloads file names.")
    parser.add_argument("--translateOptSchemes", help="Translate the IDs of  optimal schemes to the detailed definition.", action='store_true')
    parser.add_argument("--no_warmup", help="Do not warmup the optimizer.", action='store_true')
    parser.add_argument("--dSizes", type=int, help="Division size for the divide-and-pause computation.", nargs='+', default=[None])
    parser.add_argument("--compiter", type=int, help="Number of iterations for the divide-and-pause computation.", default=100)
    parser.add_argument("--candidateLimit", type=int, help="Limit the number of candidates for the (pre)computation.")
    parser.add_argument("--processes", type=int, help="Number of processes for the computation.", default=1)
    parser.add_argument("--compact", help="Compactify the input data.", action="store_true", default=False)
    parser.add_argument("--skip", type=int, help="Skip the first n workloads.", default=0)
    parser.add_argument("--queryStepSize", type=int, help="Step size for querying.", default=0)
    parser.add_argument("--accessSetRegionMapping", type=str, nargs='+', help="Access set region mapping: region name to index in array.", default=[])
    parser.add_argument("--noStrictReplication", help="Do not force replication factor to be exactly f, but allow higher replication as well.", action='store_true', default=False)
    parser.add_argument("--rescale", help="Rescale the input data.", type=np.float64, default=1.0)
    parser.add_argument("--filterTimestampByDate", type=str, help="Filter the timestamp by date.", default=None)
    parser.add_argument("--computeNeighborsForOptimizer", type=str, nargs="+", help="Compute neighbors for the given optimizer.", default=None)
    parser.add_argument("--outputStats", type=str, help="Output stats file name.", default=None)
    parser.add_argument("--sizeScale", type=float, help="Scale the size of the objects.", default=1.0)
    parser.add_argument("--ingressScale", type=float, help="Scale the ingress volume.", default=1.0)
    parser.add_argument("--egressScale", type=float, help="Scale the egress volume.", default=1.0)
    parser.add_argument("--minReplicationFactor", type=int, help="Override minimum replication factor.", default=0)
    parser.add_argument("--noWorkloadResults", help="Do not save results for individual worklaods!", default=False, action="store_true")
    parser.add_argument("--precision", type=str, help="Floating point precision, as by torch: float64, float32, float16, bfloat16, ...", default="float64")
    parser.add_argument("--trace_dir", type=str, help="Directory with trace files.", default=None)
    parser.add_argument("--skip_spanstore", help="Skip SpanStore in experiments.", action="store_true", default=False)
    parser.add_argument("--skip_baselines", help="Skip the assorted baselines in experiments.", action="store_true", default=False)
    parser.add_argument("--skip_skypie", help="Skip SkyPIE in experiments.", action="store_true", default=False)


    args = parser.parse_args()

    if args.scenario == "default":
        args.scenario = "tier_advise/replication_factor/relative=0/relative=0"

    if args.command == "scaling":
        print("Running scaling experiments...")
        scaling(precomputation_root_dir=args.input, output_file_name=args.output, cuda_device=args.torchDeviceRayShooting)
        return 0
    
    if args.command == "real_trace":
        print("Running real trace experiments...")
        real_trace(trace_dir=args.trace_dir, precomputation_root_dir=args.input, output_file_name=args.output, cuda_device=args.torchDeviceRayShooting, do_spanstore=not args.skip_spanstore, do_baselines=not args.skip_baselines, do_skypie=not args.skip_skypie)
        return 0
    
    if args.command == "accuracy":
        print("Running accuracy experiments...")
        accuracy(precomputation_root_dir=args.input, output_file_name=args.output, cuda_device=args.torchDeviceRayShooting)
        return 0

    if args.command == "precomputation_batching":
        print("Running precomputation batching experiments...")
        precomputation_batching(precomputation_root_dir=args.input, output_file_name=args.output, cuda_device=args.torchDeviceRayShooting)
        return 0

    if args.command == "query_batching":
        print("Running query batching experiments...")
        query_batching(precomputation_root_dir=args.input, output_file_name=args.output, cuda_device=args.torchDeviceRayShooting)
        return 0
    
    if args.command == "merge_panda_results":
        print("Merging files...")
        # Traverse the directory and merge all files into a single file
        expected_columns = None
        dfs = []
        root_path_len = len(args.input.split(os.path.sep))
        for root, _, files in os.walk(args.input, topdown=True):
            
            #if len(root.split(os.path.sep)) >= root_path_len+1:
            #    break

            for file in files:
                if file.endswith(".pandas.pickle"):
                    print(f"Merging {root}/{file}...")
                    dfs.append(pd.read_pickle(os.path.join(root, file)))

                    cur_columns = set(dfs[-1].columns)
                    if not expected_columns:
                        expected_columns = cur_columns
                    else:
                        if expected_columns != cur_columns:
                            print(f"WARN: Skipping file {root}/{file}, due to unexpected columns:\nMissing columns: {expected_columns - cur_columns}\nAdditional columns: {cur_columns - expected_columns}")
                            del dfs[-1]

        df = pd.concat(dfs, ignore_index=True)
        output = args.output if args.output else os.path.join(args.input, "merged.pandas.pickle")
        df.to_pickle(output)
        return 0

    if not args.networkPriceFile:
        args.networkPriceFile = pkg_resources.resource_filename(__name__, "data/network_cost_v2.csv")
    if not args.storagePriceFile:
        args.storagePriceFile = pkg_resources.resource_filename(__name__, "data/storage_pricing.csv")

    optimizers = []
    for optimizer in args.optimizer:
        optimizers.extend(createOptimizer(optimizer=optimizer, args=args.__dict__))

    if args.useCandidates:
        optimizers.extend(createOptimizer(optimizer="Candidates", args=args.__dict__))

    args.optimizer = optimizers if len(optimizers) > 0 else [OptimizerType(type="None", useClarkson=False, useGPU=False)]
    if args.verbose > 0:
        print(f"Using optimizer: ", args.optimizer)

    inputFileName = args.input
    scenarioName = args.scenario

    if args.command in ["toJson","toJSON", "toPickle"]:
        readMode = "rb" if "pickle" in inputFileName else "r"
        if inputFileName.endswith("bz2"):
            inF = bz2.BZ2File(inputFileName, readMode)
        else:
            inF = open(inputFileName, readMode)
        if "pickle" in inputFileName:
            data = pickle.load(inF)
        elif "json" in inputFileName:
            data = json.load(inF, cls=EnhancedJSONDecoder)
        else:
            print("Input file must be json or pickle!")
            exit(1)

        inF.close()

        if args.compact:
            data = compactifyForPrecomputation(data)

        outputFilename = args.output
        if args.command in ["toJson","toJSON"]:
            #outputFilename = "".join(inputFileName.split(".")[:-1]) + ".json"
            with open(outputFilename, "w") as outF:
                json.dump(data, outF, cls=EnhancedJSONEncoder, indent=4)
        elif args.command == "toPickle":
            #outputFilename = "".join(inputFileName.split(".")[:-1]) + ".pickle"
            with open(outputFilename, "wb") as outF:
                pickle.dump(data, outF)
        else:
            raise Exception("Unknown command!")

        print(f"Converted to {outputFilename}")

    elif args.command == "listRegions":
        oracle = Oracle(inputFileName=inputFileName, verbose=args.verbose)
        oracle.prepare_scenario(scenario_name=scenarioName, optimizer_type=args.optimizer[-1])
        regions = list(oracle.get_application_regions().keys())
        regions.sort()
        print(f"Available regions ({len(regions)}):")
        print(regions)

    elif args.command == "statisticsPrecomputation":
        oracle = Oracle(inputFileName=inputFileName, verbose=args.verbose)
        
        if args.addOptimizersFromInput:
            args.optimizer.extend(oracle.__discoverOptimizerTypes(scenarioName=scenarioName))

        oracle.statistics(scenarioName=scenarioName, optimizers=args.optimizer, output=args.output)

    elif args.command == "mergeScenarios":
        pass

    elif args.command == "testWorkloadCost":
        oracle = Oracle(inputFileName=inputFileName, verbose=args.verbose)
        oracle.prepare_scenario(scenario_name=scenarioName, optimizer_type=args.optimizer[-1])

        scheme = oracle.get_schemes()[0]
        assert(scheme.cost.storage == scheme.cost.equation[1])
        no_apps = oracle.no_apps
        # Test size price
        w = oracle.create_workload(size=1, put=np.zeros(no_apps), get=np.zeros(no_apps))
        value = oracle.__compute_optimal_value(optimalSchemeIndex=0, workload=w)
        expected = scheme.cost.storage
        if value != expected:
            print(f"Error storage: {value} == {expected}")

        for i in range(no_apps):
            # Test put and ingress price
            put=np.zeros(no_apps)
            get=np.zeros(no_apps)
            put[i] = 1
            w = oracle.create_workload(size=1, put=put, get=get)
            value = oracle.__compute_optimal_value(optimalSchemeIndex=0, workload=w)
            expected = scheme.cost.storage + scheme.cost.ingress[i] + scheme.cost.put
            if value != expected:
                print(f"Error put + ingress + storage: {value} != {expected}")

            put[i] = 0
            get[i] = 1
            w = oracle.create_workload(size=1, put=put, get=get)
            value = oracle.__compute_optimal_value(optimalSchemeIndex=0, workload=w)
            expected = scheme.cost.storage + scheme.cost.egress[i] + scheme.cost.get[i]
            if value != expected:
                print(f"Error get + ingress + storage: {value} != {expected}")

    elif args.command == "queryAccessSetFile":
        #benchmarkQueryAccessSetFile(args=args)
        trace_args = dict(
            accessSetRegionMapping=args.accessSetRegionMapping,
            inputFileName=inputFileName,
            scenarioName=scenarioName,
            addOptimizersFromInput=args.addOptimizersFromInput,
            optimizer=args.optimizer,
            noWorkloads=args.noWorkloads,
            no_warmup=args.no_warmup,
            translateOptSchemes=args.translateOptSchemes,
            output_file=args.output,
            skipWorkloadResults=args.noWorkloadResults,
            batchSizes=args.batchSizes,
            implArgs=args.__dict__,
            verbose=args.verbose,
            filterTimestampByDate=args.filterTimestampByDate,
            inputWorkloads=args.inputWorkloads,
            sizeScale=args.sizeScale,
            ingressScale=args.ingressScale,
            egressScale=args.egressScale,
            rescale=args.rescale
        )
        query_trace(
            **trace_args
        )

    elif args.command == "queryFile":
        print ("Querying the oracle for the optimum of the given workload file.")

        oracle = Oracle(inputFileName=inputFileName, verbose=args.verbose)
        oracle.prepare_scenario(scenario_name=scenarioName, optimizer_type=args.optimizer[-1])

        # Discover optimizers from precomputation
        if args.addOptimizersFromInput:
            args.optimizer.extend(oracle.get_precomputation_optimizers())

            for optimizer in args.optimizer:
                optimizer.implementationArgs = setImplementationArgs(implementation=optimizer.implementation, args=args.__dict__)

            print("Discovered optimizers from precomputation: ", args.optimizer)

        # Load workloads from file
        workloads = load_workloads(oracle, input=args.inputWorkloads)

        benchmarkQuerying(oracle, workloads=workloads, scenarioName=scenarioName, optimizers=args.optimizer, batchSizes=args.batchSizes, no_warmup=args.no_warmup, translateOptSchemes=args.translateOptSchemes, output=args.output, noWorkloadResults=args.noWorkloadResults)

    elif args.command == "queryRandom":
        query_random(
            inputFileName=inputFileName,
            scenarioName=scenarioName,
            addOptimizersFromInput=args.addOptimizersFromInput,
            optimizer=args.optimizer,
            workloadSeed=args.workloadSeed,
            workloadRange=args.workloadRange,
            noWorkloads=args.noWorkloads,
            queryStepSize=args.queryStepSize,
            querySkip=args.skip,
            no_warmup=args.no_warmup,
            translateOptSchemes=args.translateOptSchemes,
            output_file=args.output,
            skipWorkloadResults=args.noWorkloadResults,
            batchSizes=args.batchSizes,
            implArgs=args.__dict__,
            verbose=args.verbose
        )
        
    elif args.command == "compute" or args.command == 'computeNeighbors':

        doNormalCompute = args.command == "compute"

        if doNormalCompute:
            print(f"Computing optima for {scenarioName} using {inputFileName}...")
        else:
            print(f"Computing neighbors for {scenarioName} using {inputFileName}...")
        sys.stdout.flush()

        oracle = Oracle(inputFileName=inputFileName, verbose=args.verbose)
        # Create oracle instance by prepare_scenario
        scenarioArgs = {"scenarioName": scenarioName, "repairInputData": args.repairInputData}
        oracle.prepare_scenario(**scenarioArgs, optimizer_type=args.optimizer[-1])

        if args.verbose >= 0:
            print("Oracle instance created. Computing...")

        sys.stdout.flush()

        algoArgs = {
            k:args.__dict__[k] for k in ["torchDeviceRayShooting", "normalize", "optimizerThreads", "nonnegative"] if k in args 
        }

        # Find which optimizers were used for the first precomputation step
        optimizersPrecomputation = oracle.__discoverOptimizerTypes(scenarioName)

        if args.computeNeighborsForOptimizer and len(args.computeNeighborsForOptimizer) > 0:
            optimizersPrecomputation = [o for o in optimizersPrecomputation if o.name in args.computeNeighborsForOptimizer]

        for dsize in args.dSizes:
            for optimizer in args.optimizer:
                if not optimizer.useClarkson and optimizer.useGPU:
                    print("Only Clarkson mode supports GPU. Skipping ", optimizer.name, "...")
                    continue

                if args.verbose >= 0:
                    print(f"Computing optima with optimizer {optimizer.name}")
                    sys.stdout.flush()

                if doNormalCompute:
                    oracle.compute_optima(optimizerType=optimizer, divisionSize=dsize, iterations=args.compiter, candidateLimit=args.candidateLimit, processes=args.processes, algoArgs=algoArgs)
                    if args.verbose >= 0:
                        print(f"Computed with optimizer {optimizer.name}")
                        sys.stdout.flush()
                    if args.validate:
                        #oracle.validate(baseline=, comp=, dump=args.dump)
                        # TODO
                        pass
                    if args.output is not None:
                        oracle.save(args.output)
                    if args.outputStats:
                            oracle.statistics(optimizers=args.optimizer, output=args.outputStats, scenarioName=scenarioName)

                else:
                    print(f"Computing neighbors for precomputed optimizers {optimizersPrecomputation}")
                    for optimizerPrecomputation in optimizersPrecomputation:
                        oracle.prepare_scenario(**scenarioArgs, optimizer_type=optimizerPrecomputation, skip_instance=True)
                        oracle.compute_neighborhood(optimizerType=optimizer, divisionSize=dsize, iterations=args.compiter, processes=args.processes, algoArgs=algoArgs)
                        if args.verbose >= 0:
                            print(f"Computed with optimizer {optimizer.name} based on initial precomputation of {optimizerPrecomputation.name}")
                            sys.stdout.flush()

                        if args.output is not None:
                            oracle.save(args.output)
                        if args.outputStats:
                            oracle.statistics(optimizers=optimizersPrecomputation, output=args.outputStats, scenarioName=scenarioName)


    else:
        raise ValueError(f"Unknown command {args.command}")

if __name__ == "__main__":
    __main__()
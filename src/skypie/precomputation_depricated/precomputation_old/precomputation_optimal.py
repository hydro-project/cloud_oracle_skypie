def compute_optima(self,*, divisionSize: "int|None"=None, iterations=1000, candidateLimit: "int|None"=None, processes: int=1, optimizerType: "OptimizerType" = OptimizerType(type=MosekOptimizerType.InteriorPoint, useClarkson=True, useGPU=True), algoArgs: "Dict[str, Any]" = {"torchDeviceRayShooting": "cpu", "normalize": NormalizationType.No, "nonnegative" : True, "optimizerThreads": 1}):
    """
    This function computes the optimal partitions for the current scenario.
    It filters the candidate partitions for those that are optimal for some workload.

    Note that this function only updates the scenario data.
    For using the updated data, the scenario has to be prepared again!
    The data neither is persisted! The data has to be written back for persistence!

    The computation either uses a acurrate one-shot or the divide-and-pause approach.
    One-shot:
    If divisonSize or iterations is None, then optimal partitions are computed in a single step,
    resulting in the actual minimal set of optimal partitions
    Divide-and-pause:
    If a divisionSize and iterations are given, then the candidates are divided into groups which are separately filtered.
    The filtered result again is divided into groups and filtered again, until the number of iterations is reached.
    Notably, the final set of partitions is not minimal (containing some non-optimal partitions),
    but the computation is much faster and still permits optimal queries.

    XXX: Importantly, redundancy elimination currently does not compute an interior point on it's own,
    but requires an interior point to be given!
    This interior point must lie strictly within the polytope of the cost-workload halfplanes!
    Currently, this interior point is hacked as -1 cost and epsilon workload,
    exploiting that the cost-worklaod halfplanes are all in the positive orthant
    and the lower bound planes are ignored when the point lies outside their halfspace.
    The hacked interior point seems sufficient for all our uses cases,
    though when it is insufficient it causes silent catastrophic corruption!

    Algo args:
    torchDeviceRayShooting
    nonnegative
    normalize
    optimizerThreads
    """

    if self.verbose > 0:
        print("Computing optimal partitions with arugments:")
        print(f"\tdivisionSize: {divisionSize}")
        print(f"\titerations: {iterations}")
        print(f"\tcandidateLimit: {candidateLimit}")
        print(f"\tprocesses: {processes}")
        print(f"\toptimizerType: {optimizerType}")
        print(f"\talgoArgs: {algoArgs}")
        sys.stdout.flush()

    if not self.__candidateSchemes:
        print("No candidate schemes found.")
        return

    optimizerTypeName = optimizerType.name

    doDivide = divisionSize is not None and iterations is not None

    """
    Convert candidate schemes to single tensor of inequalities:
    - Add lower bound of 0 workload and cost, i.e., for each dimension: x_i >= 0
    - Get costWLhalfplanes of each scheme
    - (Optionally: Add upper bound of bigM cost)
    """

    no_dim = len(self.__candidateSchemes[0].costWorkloadHalfplane)

    if candidateLimit is None:
        candidateLimit = len(self.__candidateSchemes)
    
    inequalities = np.vstack([s.costWorkloadHalfplane for s in self.__candidateSchemes[:candidateLimit]])

    assert inequalities.shape[1] == no_dim
    assert inequalities.shape[0] == len(self.__candidateSchemes[:candidateLimit]) #+ no_dim - 1

    interiorPoint = None

    if optimizerType.useGPU == False:
        algoArgs["torchDeviceRayShooting"] = "cpu"
    elif optimizerType.useGPU and "cuda" in algoArgs.get("torchDeviceRayShooting", "") and not torch.cuda.is_available():
        raise RuntimeError("cuda is unavailable, but requested for rayshooting")

    '''
    Divide-and-pause approach:
    If a divisionSize and iterations are given, then the candidates are divided into groups which are separately filtered.
    After each iteration, the redundant inequalities are removed and the non-redundant inequalities are used as new candidates.

    In the first iteration, the candidates are divided into groups of size divisionSize.
    In the subsequent iterations, the filtered result is assigned random groups.
    The original candidate ids have to be stored for the final result.
    '''

    if doDivide:
        timer = Timer()
        wallTimer = Timer()

        for nonredundant3, redundant3, localOptimizerType in parallelRedundancyElimination(processes=processes, iterations=iterations, divisionSize=divisionSize, interiorPoint=interiorPoint,inequalities=inequalities, algoArgs=algoArgs,optimizerType=optimizerType, verbose=self.verbose, timer=timer, wallTimer=wallTimer):

            # Give total wall clock time of redundancy elimination
            clarksonDuration = timer.getTotalTime()

            # Save result under name of optimizer type, batch size and iteration
            # Copy non redundant candidates to optimal
            if "optimal_partitions_by_optimizer" not in self.__scenario:
                self.__scenario["optimal_partitions_by_optimizer"] = dict()

            name = localOptimizerType.name #+ "_dsize" + str(divisionSize) + "_iter" + str(i)
            self.__scenario["optimal_partitions_by_optimizer"][name] = dict()
            result = self.__scenario["optimal_partitions_by_optimizer"][name]
            result["optimizer_type"] = localOptimizerType
            # Save optimal partitions
            result["optimal_partitions"] = []
            #for pos in nonredundant3:
            #    result["optimal_partitions"].append(self.__scenario["candidate_partitions"][pos])

            # Write back additional stats
            # Number of optimal repliction schemes
            no_facets = len(nonredundant3)
            result["no_facets"] = no_facets
            # Number of redundnat repliction schemes
            result["no_redundant_facets"] = len(redundant3)
            # Ids of optimal repliction schemes
            result["optimal_partition_ids_delta"] = list(deltaEncode(nonredundant3))
            # Ids of redundant repliction schemes
            result["nonoptimal_partition_ids_delta"] = list(deltaEncode(redundant3))
            # Time for computing the optimal repliction schemes/partitions
            result["partitioner_wall_time_ns"] = wallTimer.getOverheadTime() # Wall clock time overall
            result["partitioner_time_ns"] = timer.getTotalTime() # Time of parallel processes total (overhead+compute) 
            result["partitioner_computation_time_ns"] = timer.getComputationTime() # Time of parallel processes for computation 

            if self.verbose > 0:
                print(result)
            print(f"Redundancy elimination done. {no_facets} facets found, after {wallTimer.getOverheadTime()/1e9:.2f} seconds (wall clock time).")

    else:
        timer=Timer()
        if optimizerType.useClarkson:
            res3 = redundancyEliminationClarkson(inequalities=inequalities, interiorPointOrig=interiorPoint, verbose=self.verbose, optimizerType=optimizerType.type, timer=timer, **algoArgs)
        else:
            res3 = redundancyElimination(inequalities=inequalities, verbose=self.verbose, optimizerType=optimizerType.type, timer=timer, **algoArgs)

        # Get ids of nonredundant/optimal and redundant/nonoptimal partitions and ignore the additional bounds
        nonredundant3 = [ i for i, (r, _) in enumerate(res3[:len(self.__candidateSchemes)]) if r == True]
        redundant3 = [ i for i, (r, _) in enumerate(res3[:len(self.__candidateSchemes)]) if r == False]
        clarksonDuration = timer.getTotalTime()
        if self.verbose > 0:
            print(f"Redundancy elimination by {optimizerType.name} in {clarksonDuration} ns\nwith compute time {timer.getComputationTime()} ns\nNonredundant {len(nonredundant3)}: {nonredundant3}")

        # Save result under name of optimizer type
        # Copy non redundant candidates to optimal
        if "optimal_partitions_by_optimizer" not in self.__scenario:
            self.__scenario["optimal_partitions_by_optimizer"] = dict()

        self.__scenario["optimal_partitions_by_optimizer"][optimizerType.name] = dict()
        result = self.__scenario["optimal_partitions_by_optimizer"][optimizerType.name]
        result["optimizer_type"] = optimizerType
        # Save optimal partitions
        result["optimal_partitions"] = []
        #for i in nonredundant3:
        #    result["optimal_partitions"].append(self.__scenario["candidate_partitions"][i])

        # Write back additional stats
        # Number of optimal repliction schemes
        no_facets = len(nonredundant3)
        result["no_facets"] = no_facets
        # Number of redundnat repliction schemes
        result["no_redundant_facets"] = len(redundant3)
        # Ids of optimal repliction schemes
        result["optimal_partition_ids_delta"] = list(deltaEncode(nonredundant3))
        # Ids of redundant repliction schemes
        result["nonoptimal_partition_ids_delta"] = list(deltaEncode(redundant3))
        # Time for computing the optimal repliction schemes/partitions
        result["partitioner_time_ns"] = clarksonDuration
        result["partitioner_computation_time_ns"] = timer.getComputationTime()

        if self.verbose > 0:
            print(result)
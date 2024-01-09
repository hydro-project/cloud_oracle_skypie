def compute_neighborhood(self,*, divisionSize: "int|None"=None, iterations=1, processes: int=1, optimizerType: "OptimizerType" = OptimizerType(type=MosekOptimizerType.InteriorPoint, useClarkson=False, useGPU=False), algoArgs: "Dict[str, Any]" = {"torchDeviceRayShooting": "cpu", "normalize": NormalizationType.No, "nonnegative" : True, "optimizerThreads": 1}):
        """
        This function computes which optimal partitions are inner neighbors for the current scenario.
        This is a very similar algorithm to compute_optima!

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
            print(f"\tprocesses: {processes}")
            print(f"\toptimizerType: {optimizerType}")
            print(f"\talgoArgs: {algoArgs}")
            sys.stdout.flush()

        if not self.__replicationSchemes or len(self.__replicationSchemes) == 0:
            print(f"No optimal replication schemes found! (Did you run compute_optima() for this optimizer {optimizerType}?)")
            return

        doTest = False
        if doTest:
            inequalities = np.array([
                [1,1,0], # 1 >= 1x + 0y <- irredundant
                [3,1,0], # 2 >= 1x + 0y <- redundant
                [3,0,1], # 2 >= 0x + 1y <- redundant
                [1,0,1], # 1 >= 0x + 1y <- redundant
                [-0.1,-1,0] # 0.1 <= 1x + 0y <- irredundant
            ])
        else:
            no_dim = len(self.__replicationSchemes[0].costWorkloadHalfplane)
            
            inequalities = np.vstack([s.costWorkloadHalfplane for s in self.__replicationSchemes])

            assert inequalities.shape[1] == no_dim
            assert inequalities.shape[0] == len(self.__replicationSchemes) #+ no_dim - 1

        interiorPoint = None

        if optimizerType.useGPU == False:
            algoArgs["torchDeviceRayShooting"] = "cpu"
        elif optimizerType.useGPU and "cuda" in algoArgs.get("torchDeviceRayShooting", "") and not torch.cuda.is_available():
            raise RuntimeError("cuda is unavailable, but requested for rayshooting")

        if not doNormalCompute:
            algoArgs["overestimate"] = False

        timer = Timer()
        wallTimer = Timer()

        """
        # Compute neighborhood of optimal partitions using redundancy elimination,
        # i.e., irredudant partitions are neighbors of each other
        Loop over each partition and gradually annotate the neighbors:
        - Set costplane of partition under check as equality
        - Pass in known neighbors as hint
        - Compute all its neighbors
        - Add itself as neighbor to all its neighbors
        """

        neighbors = { i : set() for i in range(inequalities.shape[0])}

        pool = Pool(processes=processes)

        for current_index in range(inequalities.shape[0]):
            # Set costplane of partition under check as equality
            algoArgs["equalities"] = inequalities[current_index:current_index+1,:]
            knownIrredundant = neighbors[current_index] | {current_index}

            tight_neighbors = set()
            for tight_neighbors, _redundant, _localOptimizerType in parallelRedundancyElimination(knownIrredundant=knownIrredundant, processes=processes, iterations=iterations, divisionSize=divisionSize, interiorPoint=interiorPoint,inequalities=inequalities, algoArgs=algoArgs,optimizerType=optimizerType, verbose=self.verbose, timer=timer, wallTimer=wallTimer, pool=pool):
                pass
            # We don't care about the information of the individual iterations here, simply take the final
            neighbors[current_index] = set(tight_neighbors)
            
            if current_index in neighbors[current_index]:
                neighbors[current_index].remove(current_index) # Remove itself as neighbor

            # Add itself as neighbor to all its neighbors
            for neighbor in neighbors[current_index]:
                neighbors[neighbor].add(current_index)     

        # Save result under name of optimizer type, batch size and iteration
        localOptimizerType = optimizerType.custom_copy(dsize=divisionSize)
        name = localOptimizerType.name
        result = dict()
        result["optimizer_type"] = localOptimizerType
        # Ids of optimal repliction schemes
        result["incident_partitions_id"] = neighbors
        # Time for computing the optimal repliction schemes/partitions
        result["incidence_wall_time_ns"] = wallTimer.getOverheadTime() # Wall clock time overall
        result["incidence_time_ns"] = timer.getTotalTime() # Time of parallel processes total (overhead+compute) 
        result["incidence_computation_time_ns"] = timer.getComputationTime() # Time of parallel processes for computation 

        self.set_neighborhood(name, result)

        if self.verbose > 0:
            print(result)

        pool.close()
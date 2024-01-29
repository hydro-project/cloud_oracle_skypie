import numpy as np
import sys
from typing import Set
import torch
from skypie.precomputation_depricated.ray_shooting import *
from skypie.util.my_dataclasses import Timer
from multiprocessing import Pool
#from torch.multiprocessing import Pool, Process, set_start_method
#try:
#     set_start_method('spawn')
#except RuntimeError:
#    pass
from skypie.precomputation_depricated.pareto_brute_force import compute_pareto_frontier
from skypie.precomputation_depricated.redundancy_elimination import redundancyElimination
from skypie.precomputation_depricated.redundancy_elimination_clarkson import redundancyEliminationClarkson

def redundancyEliminationJob(*, start, ineqMapping, diff, optimizerType, **algoArgs):

    print(f"Start redundancy elimination for inequalities {start} to {start+diff-1} using {optimizerType}...")
    sys.stdout.flush()

    timerLocal = Timer()
    if optimizerType.type == "Pareto":
        tensor = torch.from_numpy(algoArgs["inequalities"][:,1:-1].copy())
        #tensor *= -1
        resOther = compute_pareto_frontier(tensor, device=algoArgs["torchDeviceRayShooting"], math=False)
        res3 = [r for r in resOther[:diff]]
    else:
        if optimizerType.useClarkson:
            res3 = redundancyEliminationClarkson(timer=timerLocal,optimizerType=optimizerType.type, **algoArgs)
            res3 = [r for (r, _) in res3[:diff]]
        else:
            res3 = redundancyElimination(timer=timerLocal, optimizerType=optimizerType.type, **algoArgs)
            res3 = [r for (r, _) in res3[:diff]]

    # res is a List[Tuple[boolean, float]]. The first element of the tupe at list index i is true if the ith inequality is non-redundant
    # res may contain entries for additional inequalities, we ignore these by iteration only util :diff
    # This list contains the id of the non-redundant inequalities. We convert the inequality id of the batch to the inequality id of the iteration/the row in the inequalityRun array
    if ineqMapping is None:
        # List of ids of the non-redundant inequalities
        nonredundant = [start+pos for pos, r in enumerate(res3) if r == True]
        # List of ids of the redundant inequalities
        redundant = [start+pos for pos, r in enumerate(res3) if r == False]
    else:
        # Tranlate random inequality ids of iteration back to original ids
        nonredundant = [ineqMapping[start+pos] for pos, r in enumerate(res3) if r == True]
        # List of ids of the redundant inequalities with same conversion as above
        redundant = [ineqMapping[start+pos] for pos, r in enumerate(res3) if r == False]

    return nonredundant, redundant, timerLocal

def parallelRedundancyElimination(*, processes, iterations, divisionSize, interiorPoint, inequalities, wallTimer, algoArgs, verbose, timer, optimizerType, shuffleFirstIteration = False, knownIrredundant : "Set[int]" = set(), pool = None):
    
    no_dim = inequalities.shape[1] - 1

    # FIXME: This is not working correctly. With repliction factor 2 the number of non-redundant inequalties falsely redunces with each iteration
    # At the end, this array should contain the original row ids of the redundant inequalities
    redundant3 = []
    # At the end, this array should contain the original row ids of the (supposed) non-redundant inequalities
    nonredundant3 = list(range(inequalities.shape[0])) # Allocate np.array latern on

    # Mapping from current id to original id
    # Position i contains the original id of the current id i
    ineqMapping = None # Allocate mapping later on

    poolWasNone = pool is None
    if poolWasNone:
        pool = Pool(processes=processes)
    jobs = []
    for i in range(iterations):
        print(f"Starting iteration {i+1}/{iterations} with remaining {len(nonredundant3) if len(nonredundant3) != 0 else len(inequalities)} inequalities and division size {divisionSize}.")
        sys.stdout.flush()

        localOptimizerType = optimizerType.custom_copy(dsize=divisionSize, iteration=i)
        wallTimer.continueOverhead()

        knownIrredundantMapped = set()

        # Shuffle current nonredundant inequalities
        if i > 0 or shuffleFirstIteration:
            if len(nonredundant3) < no_dim:
                print("Not enough non-redundant inequalities found. Stopping.")
                break

            # Allocate mapping if not already done, only needs to be done once and the number of nonredundant inequalities is shirnking
            if ineqMapping is None:
                ineqMapping = np.empty(len(nonredundant3), dtype=int)

            # Compute random IDs for original IDs of non redundant inequalities by shuffling the array of original IDs, i.e., the random position of the original ID is the random ID
            np.random.shuffle(nonredundant3)
            # Store from the random ID to the original ID
            for randomID, originalID in enumerate(nonredundant3):
                ineqMapping[randomID] = originalID
                # Also translate knownIrredundant
                if originalID in knownIrredundant:
                    knownIrredundantMapped.add(randomID)

            # Move the nonredundant inequalities to the random positions, i.e., by a list stating which original ID should be at the position of the inequalityRun
            inequalitiesRun = inequalities[nonredundant3]
        else:
            # TODO: does referencing spoit the original input?
            inequalitiesRun = inequalities
            knownIrredundantMapped = knownIrredundant

        # reset nonredundant inequalities
        nonredundant3.clear()

        # Compute redundancy for each batch of size divisonSize
        for start in range(0, len(inequalitiesRun), divisionSize):
            # The current batch is from start to end
            # The batch must remain full dimensional, i.e., its size must be at least the number of dimensions
            # Hence, if the last batch is smaller than the number of dimensions, we merge it with the previous batch
            totalRemaining = len(inequalitiesRun) - start
            if (totalRemaining - divisionSize) < no_dim:
                end = len(inequalitiesRun)
            else:
                end = min(start+divisionSize, len(inequalitiesRun))
            
            # The actual batch size is this diff
            diff = end - start

            #assert diff >= (no_dim-1), f"Must have at least as many inequalities as dimensions! But found {diff} inequalities and {no_dim-1} dimensions."


            # Get the known irredundant inequalities for the current batch
            knownIrredundantCurrent = set()
            for irredID in knownIrredundantMapped:
                if start <= irredID:
                    knownIrredundantCurrent.add(irredID - start)
                elif irredID >= end:
                    break

            # Get ids of nonredundant/optimal and redundant/nonoptimal partitions

            # Start job for current batch
            jobArgs = dict(start=start,knownIrredundant=knownIrredundantCurrent, ineqMapping=ineqMapping, diff=diff, inequalities=inequalitiesRun[start:end], interiorPointOrig=interiorPoint, verbose=verbose, optimizerType=localOptimizerType, **algoArgs)
            job = pool.apply_async(redundancyEliminationJob, kwds=jobArgs)
            jobs.append(job)

            if end == len(inequalitiesRun):
                # We are done with the last iteration, terminate the loop explicitly as we may have merged the last batch with the previous one
                break

        # Get results of all jobs
        for job in jobs:
            nonredundant, redundant, timerLocal = job.get()
            #print(f"Found {len(nonredundant)} non-redundant and {len(redundant)} redundant inequalities in batch.")
            nonredundant3.extend(nonredundant)
            redundant3.extend(redundant)
            timer += timerLocal
        assert len(inequalities) == len(nonredundant3) + len(redundant3), f"Expected {len(inequalities)} = {len(nonredundant3)} + {len(redundant3)}"
        jobs.clear()
        wallTimer.stop()

        if len(nonredundant3) == 0:
            raise RuntimeError("No nonredundant3 found in iteration " + str(i) + ". This should not happen.")

        # Sort the ids of the non-redundant inequalities, for convenience
        nonredundant3.sort()
        redundant3.sort()

        yield nonredundant3, redundant3, localOptimizerType

        if len(nonredundant3) < no_dim and optimizerType.useClarkson:
            print("Not enough non-redundant inequalities remaining for Clarkson's algorithm. Stopping...")
            break
    
    if poolWasNone:
        pool.close()
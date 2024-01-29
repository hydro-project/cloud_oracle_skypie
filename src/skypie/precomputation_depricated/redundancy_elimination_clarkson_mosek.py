import sys
import numpy as np
import torch
from typing import List, Tuple, Set
from mosek.fusion import *

from skypie.util.my_dataclasses import Timer, MosekOptimizerType, NormalizationType
from skypie.util.mosek_settings import setSolverSettings
from skypie.precomputation_depricated.data_normalization import dataNormalization
from skypie.precomputation_depricated.ray_shooting import *

def redundancyEliminationClarkson(*, inequalities: "np.ndarray", interiorPointOrig: "np.ndarray|None" = None, verbose=0, torchDtype=torch.float64, torchDeviceRayShooting: str ="cpu", torchDeviceOther: str ="cpu", normalize=NormalizationType.No, lastColSpecial = False, nonnegative = True, optimizerType = MosekOptimizerType.Free, optimizerThreads = 0, timer: "Timer|None" = None, equalities: "np.ndarray" = None, knownIrredundant: "Set[int]" = set(), overestimate = True, **ignoreArgs) -> List[Tuple[bool, float]]:
    """
    Eliminate redundancies in the polytope Ax >= b, provided as a list of inequalities in form b  + a0 x0 + a1 x1 + ... >= 0.
    This is the output-sensitive algorithm of Clarkson.
    It uses ray-shooting to only solves LPs with at most the no. of constraints of nonredundant inequalities.

    By default the polytope in form Ax >= b is expected, which means it has to be converted to hit the LP formulation of Clarkson/Szedlak, i.e., hence the default convert=True.
    Alternatively a polytope in form Ax <= b can be provided, in which case convert=False.

    The interior point must be properly inside the polytope, i.e., it must be a point that is not on the boundary of the polytope.
    """

    if verbose >= 0:
        assert inequalities.shape[0] >= (inequalities.shape[1]-1), f"Must have at least as many inequalities as dimensions! But found {inequalities.shape[0]} inequalities and {inequalities.shape[1]-1} dimensions."
        assert inequalities.dtype == np.double, "Inequalities must be of type np.double"

        print(f"Using Mosek optimizer: {optimizerType}")
        print(f"Using Mosek threads: {optimizerThreads}")
        print(f"Using Normalization: {normalize.name}")

    if verbose > 1:
        print(f"Ignoring arguments: {ignoreArgs}")

    if timer is not None:
        timer.startOverhead()

    res_infeasible = np.inf if overestimate else -np.inf

    #converted = inequalities.to(device="cpu", dtype=torchDtype)
    converted = inequalities.copy()

    # Normalize inequalities after conversion for coefficients to be positive
    # XXX: Assuming cost-workload halfplane with x_0 + x_1 + ... - x' <= b
    if normalize:
        normalizedInequalities = converted[:,1:] if not lastColSpecial else converted[:,1:-1]
        normalizedInequalities, interiorPointOrig = dataNormalization(type=normalize, inequalities=normalizedInequalities, interiorPoint=interiorPointOrig)
        if not lastColSpecial:
            converted[:,1:] = normalizedInequalities
        else:
            converted[:,1:-1] = normalizedInequalities

        if equalities is not None:
            normalizedEqualities = equalities[:,1:] if not lastColSpecial else equalities[:,1:-1]
            normalizedEqualities, _ = dataNormalization(type=normalize, inequalities=normalizedEqualities, interiorPoint=None)
            if not lastColSpecial:
                equalities[:,1:] = normalizedEqualities
            else:
                equalities[:,1:-1] = normalizedEqualities

    b = converted[:, 0]
    A = converted[:, 1:]

    m, n = A.shape

    # Compute interior point if no is provided
    if interiorPointOrig is None:
        interiorPointOrig = np.zeros_like(inequalities[0])

        if verbose > 1:
            print("Computing interior point...")

        with Model("InteriorPoint") as M:
            
            setSolverSettings(M=M, optimizerType=optimizerType, optimizerThreads=optimizerThreads, verbose=verbose, normalize=normalize)

            # Setup variables
            x_ = M.variable("x*", 1, Domain.lessThan(1))
            if nonnegative:
                x = M.variable("x", n, Domain.greaterThan(0))
            else:
                x = M.variable("x", n, Domain.unbounded())

            # Setup constraints
            for i in range(m):
                # A_i x + x_ <= b_i
                M.constraint("c"+str(i), Expr.add(Expr.dot(A[i], x), x_), Domain.lessThan(b[i]))

            # Constraints of equalities
            if equalities is not None:
                b_e = equalities[:, 0]
                A_e = equalities[:, 1:]
                for i in range(b_e.shape[0]):
                    M.constraint("c_eq"+str(i), Expr.add(Expr.dot(A_e[i], x), x_), Domain.lessThan(b_e[i]))

            # Setup objective
            M.objective("obj", ObjectiveSense.Maximize, x_)

            try:
                M.solve()

                if verbose > 2:
                    print("Model:")
                    sys.stdout.flush()
                    M.writeTaskStream("ptf", sys.stdout.buffer)
                    sys.stdout.flush()

                res = M.primalObjValue()

            except Exception as e:
                print(e)
                raise e

            interiorPointOrig[1:] = x.level()
            if verbose > 0 and res > 0:
                print("Interior point found: ", interiorPointOrig)
            if verbose > 2:
                print("Interior point optimal solution: ", res)

            assert not(res < 0), "Polytope of inequalities is empty!"
            assert not(res == 0), "Interior point not found! Needs further dual search, not implemented yet..."

    # Further setup after computing interior point
    if verbose >= 0:
        assert inequalities.shape[1] == interiorPointOrig.shape[0], "Inequalities and interior point must have the same dimension"

    # List indicating whether the ith inequality is redundant (0), nonredundant (1), or not yet tested (-1)
    status = np.full(fill_value=-1, shape=inequalities.shape[0], dtype=np.int8)

    # The ray shooting algorithm demands the 0th dimension to be 1
    interiorPointOrig[0] = 1

    # Numpy setup
    solutionNumpy = np.zeros_like(interiorPointOrig)

    # torch setup
    interiorPointOther = torch.Tensor(interiorPointOrig).to(device=torchDeviceOther, dtype=torchDtype)
    #directionTorchOther = interiorPointOther.clone()
    #solutionTorchOther = interiorPointOther.clone()
    # Bridge between LP solver and torch. solutionNumpy and solutionFromNumpyTorch share the same memory.
    solutionFromNumpyTorch = torch.from_numpy(solutionNumpy)

    # Format of this function is Ax <= b but of ray_shooting it is b'+A'x>=0 -> A'x >= -b' -> -A'x <= 'b -> A' = A, b' = b
    if equalities is not None:
        convertedRayShooting = torch.Tensor(np.concatenate((equalities, inequalities), axis=0)).to(device=torchDeviceRayShooting, dtype=torchDtype)
    else:
        convertedRayShooting = torch.Tensor(converted).to(device=torchDeviceRayShooting, dtype=torchDtype)
    # XXX: Converting A = -A' for ray shooting, where [,0] is b
    convertedRayShooting[:, 1:] = -convertedRayShooting[:, 1:]

    if torchDeviceRayShooting != torchDeviceOther:
        interiorPointRayShooting = interiorPointOther.to(device=torchDeviceRayShooting)
    else:
        interiorPointRayShooting = interiorPointOther
    # Crucially the first entry of the direction must remain 0, as required by the ray-shooting algorithm.
    directionRayShooting = torch.zeros_like(interiorPointRayShooting)
    pointRayShooting = interiorPointRayShooting.clone()

    # Precompute part of the ray shooting algorithm
    T1 = precompute_ray_shooting(M=convertedRayShooting, p=interiorPointRayShooting, verbose=verbose)

    with Model("RedundancyEliminationClarkson") as M:

        w = setSolverSettings(M=M, optimizerType=optimizerType, optimizerThreads=optimizerThreads, verbose=verbose, normalize=normalize)
        
        # Setup initial model. We will add constraints of nonredundant inequalities on the go.
        
        # Setup variables
        if nonnegative:
            x = M.variable("x", n, Domain.greaterThan(0))
        else:
            x = M.variable("x", n, Domain.unbounded())

        # Setup parameters
        # c: Coefficients A[i] of the inequality under test
        c = M.parameter("c", n)
        c.setValue(np.ones(n))

        # Setup constraint for inequality under test
        # c^Tx <= b[i] + 1 -> c^Tx - b[i] <= 1
        constraintInequalityTest = M.constraint("constraint for inequality under test", Expr.sub(Expr.dot(A[0], x), b[0]), Domain.lessThan(1))

        # Constraints of known nonredundant inequalities: A[j]x <= b[j]
        for j in knownIrredundant:
            M.constraint("nonredundant inequality "+str(j), Expr.dot(A[j], x), Domain.lessThan(b[j]))

        # Constraints of equalities: A[j]x = b[j]
        if equalities is not None:
            b_e = equalities[:, 0]
            A_e = equalities[:, 1:]
            for j in range(b_e.shape[0]):
                M.constraint(f"equality{j}", Expr.dot(A_e[j], x), Domain.equalsTo(b_e[j]))

        # Objective by inequality under test: Maximize c^Tx
        M.objective("Maximize c^Tx",ObjectiveSense.Maximize, Expr.dot(c,x))

        def run(i, *, solution: "np.ndarray") -> "Tuple[np.float64, int]":
            """
            This function solves the LP for testing the ith inequality for redundancy with respect to the current nonredudant inequalities.
            """

            # Set coefficient and bounds of inequality under test
            c.setValue(A[i])
            
            # c^Tx <= b[i] + 1 -> c^Tx - b[i] <= 1
            constraintInequalityTest.update(Expr.sub(Expr.dot(A[i], x), b[i]))

            execTime=-1
            res = res_infeasible
            try:
                if timer is not None:
                    timer.continueComputation()
                M.solve()
                if timer is not None:
                    execTime = timer.stopComputation()

                if verbose > 2:
                    print("Model:")
                    sys.stdout.flush()
                    M.writeTaskStream("ptf", sys.stdout.buffer)
                    sys.stdout.flush()

                if M.getProblemStatus() == ProblemStatus.PrimalAndDualFeasible:
                    res = M.primalObjValue()
                    solution[1:] = x.level()
                else:
                    if verbose > 1:
                        print("Problem status: ", M.getProblemStatus())
                        sys.stdout.flush()
                    res = res_infeasible

            except Exception as e:
                if verbose > 1 and verbose < 2:
                    print("Model:")
                    sys.stdout.flush()
                    M.writeTaskStream("ptf", sys.stdout.buffer)
                    sys.stdout.flush()

                print(e)

            """finally:
                if verbose > 2:
                    w.seek(0)
                    regex = re.compile(r'(?<=Optimizer terminated. Time: )[0-9](\.[0-9]+)?')
                    for s in w:
                        # Extract the execution time
                        match = regex.search(s)
                        if match:
                            execTime = float(match.group(0))
                    sys.stdout.flush()
                    w.truncate(0)
                    w.seek(0)
            """

            return res, execTime

        execTimes = np.full(m, -1, dtype=np.double)

        # TODO: Initialize some nonredundant inequalities using ray-shooting, as cdd does in cddlp.c:2895

        for i in range(m):
            
            if verbose >= 0 and i % 1000 == 0:
                print(f"Checking inequality {i}")

            if i in knownIrredundant:
                status[i] = 1
                execTimes[i] = 0
                if verbose > 1:
                    print(f"Skipping known nonredundant inequality {i}")
                continue

            # If the inequality is not yet tested for redundancy. Due to ray-shooting, we may get results ahead of the index iteration
            if status[i] == -1:
                # The inequalitiy i is nonredundant if res > bi
                optVal, execTime = run(i, solution=solutionNumpy)
                nonredundant = optVal > b[i]

                if verbose > 1:
                    print(f"Found inequaltiy {i} to be redundant: {not nonredundant}")
                if verbose > 2:
                    print(f"Optimal value: {optVal}, Solution: {solutionNumpy}")

                """
                Even if the LP indicates nonredundancy, we need to verify this by ray-shooting.
                Ray-shooting may either verify the nonredundancy of this inequality or curically it may find a new nonredundant inequality.
                So, we first mark the current inequality as redundant and then possibly mark it as nonredudant.
                """
                status[i] = 0

                if nonredundant:
                    # Find unique non-redundant inequality by stable ray-shooting within the polytope from the interior point to the solution

                    if optVal != np.inf:
                        # Ray-shoot if a valid solution was found

                        # Note that direction[0] == 0 has to hold, this is the case when solutionTorch[0] = 1 and interiorPoint[0] = 1
                        pointRayShooting[1:] = solutionFromNumpyTorch[1:]

                        if timer is not None:
                            timer.continueComputation()
                        
                        torch.subtract(pointRayShooting, interiorPointRayShooting, out=directionRayShooting)
                        # XXX: Make sure format fits, ray_shooting expects b'+A'x >=0 -> A'x >= -b' -> -A'x <= 'b
                        #j = ray_shooting(M=convertedRayShooting, p=interiorPointRayShooting, r=directionRayShooting, stable=True, verbose=verbose)
                        # Rayshooting with precomputation
                        j = ray_shooting(M=convertedRayShooting, T1=T1, r=directionRayShooting, stable=True, verbose=verbose)

                        if timer is not None:
                            execTime += timer.stopComputation()
                        j = j.item()
                    else:
                        # If no valid solution was found, conservatively keep the inequality
                        j = i

                    # Add constraint of unique non-redundant inequality to model, if not already added
                    if status[j] != 1:
                        # Add constraint A[j]x <= b[j] of unique non-redundant inequality to model
                        constraint = M.constraint(f"constraint_{j}", Expr.dot(A[j], x), Domain.lessThan(b[j]))
                        # Update status of unique non-redundant inequality
                        status[j] = 1

                        if verbose > 2:
                            print(f"Added constraint_{j} to model:", constraint, f"{A[j]}x <= {b[j]}")

                    if verbose > 1:
                        print(f"When checking inequality {i}, found nonredundant inequality {j}")

                execTimes[i] = execTime

    if verbose > 0:
        for i, s in enumerate(status):
            assert s >= 0, f"status[{i}] = {s}"

    if timer is not None:
        timer.stopOverhead()

    return [ (s == 1, execTime) for (s, execTime) in zip(status, execTimes) ]
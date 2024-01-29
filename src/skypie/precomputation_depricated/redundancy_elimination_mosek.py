import sys
import numpy as np
from typing import List, Tuple, Set
from mosek.fusion import *

from skypie.util.my_dataclasses import Timer, MosekOptimizerType, NormalizationType
from skypie.util.mosek_settings import setSolverSettings
from skypie.precomputation_depricated.data_normalization import dataNormalization

def redundancyElimination(*, inequalities: "np.ndarray", verbose=0, normalize=NormalizationType.No, lastColSpecial = False, nonnegative = True, optimizerType = MosekOptimizerType.Free, optimizerThreads = 0, timer: "Timer|None" = None, equalities: "np.ndarray" = None, knownIrredundant: "Set[int]" = set(), overestimate = True, **ignoreArgs) -> List[Tuple[bool,float]]:
    """
    Refer to redundancy_elimination.py for documentation.
    """

    if verbose >= 0:
        assert inequalities.dtype == np.float64, "Inequalities must be of type np.float64"

        print(f"Using Mosek optimizer: {optimizerType}")
        print(f"Using Mosek threads: {optimizerThreads}")
        print(f"Using Normalization: {normalize.name}")
        print(f"Using Nonnegative: {nonnegative}")

    if verbose > 1:
        print(f"Ignoring arguments: {ignoreArgs}")

    if timer is not None:
        timer.startOverhead()

    with Model("RedundancyElimination") as M:

        w = setSolverSettings(M=M, optimizerType=optimizerType, optimizerThreads=optimizerThreads, verbose=verbose, normalize=normalize)

        res_infeasible = np.inf if overestimate else -np.inf

        converted = inequalities.copy()

        # Normalize inequalities after conversion for coefficients to be positive
        # XXX: Assuming cost-workload halfplane with x_0 + x_1 + ... - x' <= b
        if normalize:
            normalizedInequalities = converted[:,1:] if not lastColSpecial else converted[:,1:-1]
            normalizedInequalities, _ = dataNormalization(type=normalize, inequalities=normalizedInequalities, interiorPoint=None)
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

        # Setup variables
        if nonnegative:
            x = M.variable("x", n, Domain.greaterThan(0))
        else:
            x = M.variable("x", n, Domain.unbounded())
        #x = M.variable("x", n, Domain.greaterThan(normalizedZeros[1:]))

        # Setup parameters
        c = M.parameter("c", n)
        c.setValue(np.zeros(n))

        # Setup constraints

        # Constraints of inequalities: A[j]x <= b[j]
        # Later we update the constraint tested to A[j]x <= b[j] + 1, i.e., A[j]x - 1 <= b[j]
        constraints = [M.constraint(f"constraint{j}", Expr.dot(A[j], x), Domain.lessThan(b[j])) for j in range(m)]

        # Constraints of equalities: A[j]x = b[j]
        if equalities is not None:
            b_e = equalities[:, 0]
            A_e = equalities[:, 1:]
            for j in range(b_e.shape[0]):
                M.constraint(f"equality{j}", Expr.dot(A_e[j], x), Domain.equalsTo(b_e[j]))

        # Objective by inequality under test: Maximize c^Tx
        M.objective("Maximize c^Tx",ObjectiveSense.Maximize, Expr.dot(c,x))

        def run(i):

            if verbose >= 0 and i % 1000 == 0:
                print(f"Checking inequality {i}")
                
            # Set parameters
            c.setValue(A[i])
            # The inequality we are testing for redundancy gets slack by +1
            constraints[i].update(Expr.sub(Expr.dot(A[i], x), 1))

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
                print("Exception: ", e)
                raise e

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

            # The inequalitiy i in nonredundant if res > bi
            nonredundant = res > b[i]

            # Reset slack parameter
            constraints[i].update(Expr.dot(A[i], x))
            
            return nonredundant, execTime

        # Skip known irredundant inequalities
        res = [ run(i) if len(knownIrredundant) == 0 or not i in knownIrredundant else (True, -1) for i in range(m) ]

    if timer is not None:
        timer.stopOverhead()

    return res
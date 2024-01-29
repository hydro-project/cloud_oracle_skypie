import cvxpy as cp
import numpy as np
from typing import Set, List, Tuple

from skypie.util.my_dataclasses import Timer, MosekOptimizerType, NormalizationType
from skypie.precomputation_depricated.data_normalization import dataNormalization

def redundancyElimination(*, inequalities: "np.ndarray", verbose=0, normalize=NormalizationType.No, lastColSpecial = False, nonnegative = True, solverType = None, optimizerThreads = 0, timer: "Timer|None" = None, equalities: "np.ndarray" = None, knownIrredundant: "Set[int]" = set(), overestimate = True, epsilon = 10**-7, **ignoreArgs) -> List[Tuple[bool,float]]:
    """
    # Arguments
    epsilon: Tolerance for the inequality to be redundant. If the inequality is redundant, the objective value is at least b_i + epsilon

    Refer to redundancy_elimination.py for further documentation.
    """

    if verbose >= 0:
        assert inequalities.dtype == np.float64, "Inequalities must be of type np.float64"

        print(f"Using CVXPY optimizer: {solverType}")
        print(f"Using CVXPY threads: {optimizerThreads}")
        print(f"Using Normalization: {normalize.name}")
        print(f"Using Nonnegative: {nonnegative}")

    if verbose > 1:
        print(f"Ignoring arguments: {ignoreArgs}")

    if timer is not None:
        timer.startOverhead()

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
        x = cp.Variable(n, nonneg=True)
    else:
        x = cp.Variable(n)

    # Setup constraints
    slack_param = cp.Parameter(m)
    slack_param.value = np.zeros(m)
    constraints = [A[j] @ x - slack_param[j] <= b[j] for j in range(m)]

    # Constraints of equalities: A[j]x = b[j]
    if equalities is not None:
        b_e = equalities[:, 0]
        A_e = equalities[:, 1:]
        constraints += [A_e[j] @ x == b_e[j] for j in range(b_e.shape[0])]

    # Objective by inequality under test: Maximize c^Tx
    c = cp.Parameter(n)
    c.value = np.zeros(n)
    objective = cp.Maximize(c @ x)

    # Create problem
    problem = cp.Problem(objective, constraints)

    def run(i):

        if verbose >= 0 and i % 1000 == 0:
            print(f"Checking inequality {i}")

        # Set parameters
        c.value = A[i]

        # The inequality we are testing for redundancy gets slack by +1
        slack_param.value[i] = 1

        execTime=-1
        res = res_infeasible
        try:
            if timer is not None:
                timer.continueComputation()
            problem.solve(solver=solverType, verbose=verbose > 2, mosek_params={"MSK_IPAR_NUM_THREADS": optimizerThreads})
            if timer is not None:
                execTime = timer.stopComputation()

            if problem.status == cp.OPTIMAL:
                res = problem.value
            else:
                if verbose > 1:
                    print("Problem status: ", problem.status)
                res = res_infeasible

        except Exception as e:
            if verbose > 1 and verbose < 2:
                print("Model:")
                print(problem)
            print("Exception: ", e)
            raise e

        # The inequalitiy i in nonredundant if res > bi
        nonredundant = res - epsilon > b[i]

        # Reset slack parameter
        slack_param.value[i] = 0

        return nonredundant, execTime

    # Skip known irredundant inequalities
    res = [ run(i) if len(knownIrredundant) == 0 or not i in knownIrredundant else (True, -1) for i in range(m) ]

    if timer is not None:
        timer.stopOverhead()

    return res
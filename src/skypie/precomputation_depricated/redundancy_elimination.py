import numpy as np
from typing import List, Tuple, Set

from skypie.util.my_dataclasses import Timer, MosekOptimizerType, NormalizationType

def redundancyElimination(*, inequalities: "np.ndarray", verbose=0, normalize=NormalizationType.No, lastColSpecial = False, nonnegative = True, optimizerType = MosekOptimizerType.Free, optimizerThreads = 0, timer: "Timer|None" = None, equalities: "np.ndarray" = None, knownIrredundant: "Set[int]" = set(), overestimate = True, **ignoreArgs) -> List[Tuple[bool,float]]:
    """
    Classic redundancy elimination of a set of inequalities.

    The implementation follows: ["Lecture - Polyhedral Computation, Spring 2015" by Fukuda](https://people.inf.ethz.ch/fukudak/lect/pclect/notes2015/PolyComp2015.pdf) and ["Redundancy in Linear Systems: Combinatorics, Algorithms and Analysis" by Szedlak](http://hdl.handle.net/20.500.11850/167108)

    # Arguments
    - inequalities: The set of inequalities to be processed. It should be a numpy array of shape (m, n), where m is the number of inequalities and n is the number of variables.
    - equalities: The set of equality constraints to be considered. It should be a numpy array of shape (p, n), where p is the number of equality constraints and n is the number of variables.
    - knownIrredundant: A set of indices indicating the known irredundant inequalities.
    - normalize: The type of normalization to be applied. It should be a value from the NormalizationType enum.
    - overestimate: Flag indicating whether to overestimate the solution.
    - nonnegative: Flag indicating whether the variables are nonnegative.
    - lastColSpecial: Flag indicating whether the last column of the inequalities matrix has a special meaning.
    - optimizerType: The type of optimizer to be used. It should be a value from the MosekOptimizerType enum.
    - optimizerThreads: The number of threads to be used by the optimizer.
    - timer: An optional timer object for measuring the execution time.
    - verbose: Verbosity level. Set to 0 for no output, 1 for minimal output, and higher values for more detailed output.

    # Input format
    Expecting this format for inequalities and equalities:
    A'x <= b'

    Eliminate redundancies in the polytope provided as a list of inequalities in form A'x <= 'b.
    From
    b + a0 x0 + a1... <= x' # Cost of workload is below or equal x'
    a0 x0 + a1... - x' <= -b # Same, but with x' on lhs
    -b -a0 x0 - a1 ... + x' >= 0 # Same, but with >=
    The conversion is A' = -A and b' = -b

    XXX: Normalization by log scale does not work for negative coefficients! Normalization is tailored to the cost-workload halfplanes!
    """

    # Try to use Mosek, otherwise fallback to CVXPY
    try:
        from skypie.precomputation_depricated.redundancy_elimination_mosek import redundancyElimination as redundancyEliminationMosek
        return redundancyEliminationMosek(inequalities=inequalities, verbose=verbose, normalize=normalize, lastColSpecial=lastColSpecial, nonnegative=nonnegative, optimizerType=optimizerType, optimizerThreads=optimizerThreads, timer=timer, equalities=equalities, knownIrredundant=knownIrredundant, overestimate=overestimate, **ignoreArgs)
    except ImportError:
        print("WARN: Mosek not available, falling back to CVXPY")
        from skypie.precomputation_depricated.redundancy_elimination_cvxpy import redundancyElimination as redundancyEliminationCVXPY
        return redundancyEliminationCVXPY(inequalities=inequalities, verbose=verbose, normalize=normalize, lastColSpecial=lastColSpecial, nonnegative=nonnegative, solverType=optimizerType, optimizerThreads=optimizerThreads, timer=timer, equalities=equalities, knownIrredundant=knownIrredundant, overestimate=overestimate, **ignoreArgs)
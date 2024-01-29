import cvxpy as cp
from mosek.fusion import *
import numpy as np
import itertools
import time
import json
import sys

def computeIncidences(inequalities):

    # Setup problem
    A = inequalities[:, 1:]
    b = inequalities[:, 0]
    m = A.shape[0]
    n = A.shape[1]
    x = cp.Variable(n)
    bEqualityParam = cp.Parameter([2], nonneg=False, value=b[0:2])
    AEqualityParam = cp.Parameter([2,n], nonneg=False, value=A[0:2])
    
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x >= b, AEqualityParam @ x == bEqualityParam, x >= 0])

    # Initialize incidence list
    incidence = [ [] for i in range(m)]

    start = time.time_ns()
    # Compute incidences of all inequalities
    for (i, j) in itertools.combinations(range(m), 2):
        AEqualityParam.value = A[[i,j]]
        bEqualityParam.value = b[[i,j]]

        #print("Constraints:")
        #for k in range(m):
        #    print(f"{A[k]} @ x >= {b[i]}")
        #print(f"{AEqualityParam.value[0]} @ x == {bEqualityParam.value[0]}")
        #print(f"{AEqualityParam.value[1]} @ x == {bEqualityParam.value[1]}")

        res = prob.solve()
        #print(f"{prob.status}: {res}")

        incident = not prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE] #in [cp.OPTIMAL, cp.UNBOUNDED, cp.OPTIMAL_INACCURATE, cp.UNBOUNDED_INACCURATE]

        if incident:
            #print(f"Facets {i} and {j} are incident")
            incidence[i].append(j)
            incidence[j].append(i)
        #else:
        #    print(f"Facets {i} and {j} are not incident")

        #print(prob)
        #print(incident)
        #print(prob.status)
        #print(prob.value)
        #print(x.value)

    end = time.time_ns()
    return incidence, end-start

def det_rootn(M, t, n):
    # Setup variables
    Y = M.variable(Domain.inPSDCone(2 * n))

    # Setup Y = [X, Z; Z^T , diag(Z)]
    X   = Y.slice([0, 0], [n, n])
    Z   = Y.slice([0, n], [n, 2 * n])
    DZ  = Y.slice([n, n], [2 * n, 2 * n])

    # Z is lower-triangular
    M.constraint(Z.pick([[i,j] for i in range(n) for j in range(i+1,n)]), Domain.equalsTo(0.0))
    # DZ = Diag(Z)
    M.constraint(Expr.sub(DZ, Expr.mulElm(Z, Matrix.eye(n))), Domain.equalsTo(0.0))

    # (Z11*Z22*...*Znn) >= t^n
    M.constraint(Expr.vstack(DZ.diag(), t), Domain.inPGeoMeanCone())

    # Return an n x n PSD variable which satisfies t <= det(X)^(1/n)
    return X

def lownerjohn_inner(A, b):
    """
    
    This formulation expects a different format: Ax <= b (rather than Ax >= b)
    """
    with Model("lownerjohn_inner") as M:
        M.setLogHandler(sys.stdout)
        m, n = len(A), len(A[0])

        # Setup variables
        t = M.variable("t", 1, Domain.greaterThan(0.0))
        C = det_rootn(M, t, n)
        d = M.variable("d", n, Domain.unbounded())

        # (b-Ad, AC) generate cones
        M.constraint("qc", Expr.hstack(Expr.sub(b, Expr.mul(A, d)), Expr.mul(A, C)),
                     Domain.inQCone())

        # Objective: Maximize t
        M.objective(ObjectiveSense.Maximize, t)

        #M.writeTask("lj-inner.ptf")
        start = time.time_ns()
        M.solve()
        end = time.time_ns()
        print(f"Ellipsoid computation took: {(end-start)/1000/1000}ms")

        retC = np.array(C.level()).reshape(C.getShape())
        retd = np.array(d.level())

        # Convert matrix retC to Cholesky factorization and ... for point in ellipsoid query

        return (retC, retd)
        #return ([C[i:i + n] for i in range(0, n * n, n)], d)
    
def computeEllipsoid(inequalities):
    A = np.array(inequalities[:, 1:], dtype=np.double)
    m, n = A.shape
    b = np.array(inequalities[:, 0], dtype=np.double)
    #b = b.reshape(m)

    try:
        # For now, convert format from Ax >= b to -Ax <= -b
        A = -A
        b = -b
        res = lownerjohn_inner(A, b)
        return res
    except Exception as e:
        print(f"Exception: {e}")
        return None
    
class ComplexEncoder(json.JSONEncoder):
    """
    An extension of the default JSON encoder.
    It converts np.ndarray to standard python lists.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def scenarioIterator(inputJSON):
    """
    Iterate over all scenarios in the input JSON, either directly or for all repliction factors if present
    """
    if "replication_factor" in inputJSON:
        for (repliction_factor, scenarios) in inputJSON["replication_factor"].items():
            for (name, scenario) in scenarios.items():
                #print(f"{repliction_factor}: {name}")
                yield (name, scenario)
    else:
        for (name, scenario) in inputJSON.items():
            yield (name, scenario)

def computePartitions(*, inputFileName: str, outputFileName: str = None, scenarioSelector: str = ""):

    with open(inputFileName, 'r') as f:
        tierAdvise = json.load(f)
                    

        for (name, scenario) in scenarioIterator(tierAdvise["tier_advise"]): #tierAdvise["tier_advise"].items():
            
            # Skip scenarios that do not match the selector
            if not scenarioSelector in name:
                continue

            print(f"Precomputing partitions for scenario {name}")

            #partitions = tierAdvise["tier_advise"]['gcp-us-west4-a']['optimal_partitions']
            partitions = scenario['optimal_partitions']
            halfplanes = np.array([ p["costWLHalfplane"] for p in partitions], np.float64)
            #print(halfplanes)

            incidences, incidenceDuration = computeIncidences(halfplanes)
            scenario["incidence_computation_time_ns"] = incidenceDuration
            print(f"Incidence computation took: {(incidenceDuration)/1000/1000}ms")

            #print(incidences)

            for partitionID in range(len(partitions)):
                p = partitions[partitionID]
                incidence = incidences[partitionID]
                # Convert to list for JSON serialization
                p["incidentPartitions"] = incidence

                """
                Compute inequalities of partition's polytope in __workload space__!
                We __project__ from cost-workload space __onto the workload space__ by dropping the cost (i.e., last) column of the costWLHalfplane!
                The inequalities of the partition describe the inner space "left" of the halfplanes with the incident partitions.
                A workload point for which all inequalities are satisfied is in the polytope/partition.
                
                inequality: h'(w) - h(w) >= 0
                This is stored as a single matrix, in standard form b + Ax >= 0
                """
                # Convert to lists for JSON serialization
                p["inequalities"] = np.array([ halfplanes[i][:-1] - halfplanes[partitionID][:-1] for i in incidence ])

                # TODO: compute ellipsoid of partition's polytope in __workload space__!

    if outputFileName:
        with open(outputFileName, 'w') as f:

            # Write back to file
            json.dump(tierAdvise, f, indent=4, cls=ComplexEncoder)

    #print(tierAdvise["tier_advise"])
    return tierAdvise

if __name__ == "__main__":

    testCase = ""

    # x1 >= 0, x2 >= 0, x1 <= 1, x2 <= 1
    cube = np.array([
        # x1 >= 0
        [0, 1, 0],
        # x2 >= 0
        [ 0, 0, 1],
        # x1 <= 1 -> -x1 >= -1
        [-1, -1, 0],
        # x2 <= 1 -> -x2 >= -1
        [ -1, 0, -1],
    ])

    if len(sys.argv) > 1:

        inputFile = sys.argv[1]
        outputFile = sys.argv[2] if len(sys.argv) > 2 else None

        #inputFile = '../data/precomputed_optima/all-s3_all-aws-appliction-region_rep1.json'
        #outputFile = '../data/precomputed_optima/all-s3_all-aws-appliction-region_rep1_with-partitions.json
        #inputFile = '../data/precomputed_optima/all-s3_20-aws-application-regions_rep-2.json'
        #outputFile = '../data/precomputed_optima/all-s3_20-aws-application-regions_rep-2_with-partitions.json'

        result = computePartitions(inputFileName=inputFile, outputFileName=outputFile)
        if outputFile is None:
            json.dump(result, sys.stdout, indent=4, cls=ComplexEncoder)

    if testCase == "computeIncidences":
        incidence = computeIncidences(cube)
        print(incidence)
    elif testCase == "computeElipsoids":

        res = computeEllipsoid(cube)
        print(f"Ellipsoid of example cube: {res}")

# Compute boundaries based off of incidences

# Compute ellipsoid approximation of boundaries
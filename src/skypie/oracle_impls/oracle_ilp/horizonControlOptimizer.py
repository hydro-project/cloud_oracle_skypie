from mosek.fusion import *
import numpy as np
from typing import List, Dict, Any
import sys
import random
from datetime import datetime
import copy
from skypie.util.my_dataclasses import *
from skypie.util.mosek_settings import MyStream

def recedingHorizonControl(*, withRouting: bool = False, verbose = 1, withMigrationCosts = True, initialState: OptimizationResult = OptimizationResult(), Count, T, AS: List[int], min_f: int, max_f: int, latency_slo: Dict[int, Dict[int, int]], L_C: Dict[int, Dict[int, int]], L_S: Dict[int, Dict[int, float]], PUTs: List[Dict[Any, float]], GETs: List[Dict[Any, float]], Ingress: List[Dict[Any,float]], Egress: List[Dict[Any,float]], Size_total: int, PriceGet: Dict[int, int], PricePut: Dict[int, int], PriceStorage: Dict[int, int], PriceNet: Dict[int, Dict[int, int]], Size_avg: int, dest: Dict[int, int], AStranslate: Dict[int, int], ASdense: List[int], destDense: List[int], localSchemes: Any, timer: "Timer|None" = None, optimizerThreads: int = 0, access_cost_heuristic: bool = False, **ignoredArgs) -> OptimizationResult:
    """
    This optimizer is based on receding horizon control (RHC).
    Based on the workload over a time horizon, it jointly optimizes the replication scheme (i.e., the choice of object stores and the assignment of application regions)
    and the migration between replication schemes.

    The workload per time slot is given by a list of PUTs and GETs per time slot and application region.

    For historical state, the optimizer can be initialized with an initial state,
    where the first time slot (t=0) is assumed to be the historical state.
    Additionally, the initial state can provide hints for further time slots.
    """

    assert(min_f <= max_f)
    assert(0 <= max_f)

    if timer is not None:
        timer.continueOverhead()

    with Model("RecedingHorizonControl") as M:

        if verbose > 2:
            w = MyStream("", verbose=verbose)
            M.setLogHandler(w)

        # Enable warm start
        M.setSolverParam("mioConstructSol", "on")

        M.setSolverParam("numThreads", optimizerThreads)

        if verbose > 0:
            print(f"|AS|={len(ASdense)}\n|dest|={len(destDense)}")

        if verbose > 1:
            print(f"|AS|={len(ASdense)}\ndest={dest}")

        # Actual price for network considering average
        #PriceNetAct_ij = np.array(PriceNet_ij) * Size_avg
        # Price of get sent from i to j: GETs[i] * (PriceGet[j] + PriceNet[j,i] * Size_avg)

        # Get Price Coefficient
        PriceGetCoefTime_M = []
        for t in T:
            PriceGetCoef = dict()
            for (i, iDense) in AStranslate.items():
                PriceGetCoef[iDense] = dict()
                for (j, jDense) in dest.items():
                    # GETs_i * (PriceGet_j + PriceNet_ji * SizeAVG)
                    #PriceGetCoef[iDense][jDense] = GETs[t][i] * (PriceGet[j] + PriceNet[j][i] * Size_avg)
                    # The price for reading: Number of gets * Price of gets + Egress volume * Price of egress
                    PriceGetCoef[iDense][jDense] = GETs[t][i] * PriceGet[j] + Egress[t][i] * PriceNet[j][i]

        
            PriceGetCoef_M = Matrix.dense([list(PriceGetCoef_j.values()) for PriceGetCoef_j in PriceGetCoef.values()])
            PriceGetCoefTime_M.append(PriceGetCoef_M)

        # Simplified   
        PricePutCoefCSimpleTime = [
            #[ np.sum([ PUTs[t][i] * (PricePut[j] + PriceNet[i][j] * Size_avg) for i in AStranslate.keys()]) for j in dest.keys()] for t in T
            # The price for writing: Number of puts * Price of puts + Ingress volume * Price of ingress
            [ np.sum([ PUTs[t][i] * PricePut[j] + Ingress[t][i] * PriceNet[i][j] for i in AStranslate.keys()]) for j in dest.keys()] for t in T
        ]
        #PricePutCoefCSimple = [ np.sum([ PUTs[i] * (PricePut[j] + PriceNet[i][j] * Size_avg) for i in AStranslate.keys()]) for j in dest.values()]

        # Price for migration
        # For now, this is the price for a direct network transfer from i to j by the object size
        PriceMigrationCoef = np.array([
            [ PriceNet[i][j] * Size_total for j in dest.keys()] for i in dest.keys()
        ])

        # Sanity check that migration within same DC/object store is free
        for (i, iDense) in dest.items():
            if PriceMigrationCoef[iDense, iDense] > 0:
                print(f"WARNING: Migration from {i} to {i} is not free!")

        if verbose > 1:
            for t in T:
                for (i, iDense) in AStranslate.items():
                    for (j, jDense) in dest.items():
                        print(f"GetCoef {t} {i} {j}: {PriceGetCoefTime_M[t].get(iDense, jDense)}")
                for (j, jDense) in dest.items():
                    print(f"PutCoefC {t} {j}: {PricePutCoefCSimpleTime[t][jDense]}")

            for (i, iDense) in dest.items():
                for (j, jDense) in dest.items():
                    print(f"MigrationCoef {i} {j}: {PriceMigrationCoef[iDense, jDense]}")
    
        # Variables
        # (13) R Replica: j is a replica to i
        R = M.variable("R", [len(T), len(ASdense), len(destDense)], Domain.binary())

        # R_ij for L_Sij <= latency_slo (translated to constraint)
        for t in T:
            for (i, iDense) in AStranslate.items():
                for (j, jDense) in dest.items():
                    if i not in L_S or j not in L_S[i]:
                        if verbose >1:
                            print(f"WARNING: Missing latency data: L_S[{i}][{j}]")
                    #elif L_S[i][j] > latency_slo:
                    #    constraint = M.constraint(f"L_S{t}{i}{j}", R.index(t, iDense, jDense), Domain.equalsTo(0.0))
                    #    if verbose > 2:
                    #        print(f"{constraint.toString()}")
                    else:
                        constraint = M.constraint(f"L_S{t}{iDense}{jDense}", Expr.mul(R.index(t, iDense, jDense), L_S[i][j]), Domain.lessThan(latency_slo))
                        if verbose > 2:
                            print(f"{constraint.toString()}")

        """
        # (14) P_Sijk: wheather i synchronously forwards its PUTs via j to k
        P_S = M.variable("P_S", [[len(ASdense), len(destDense)**2]])
        # Latency of path i -> j -> k must be below latency_slo
        for (i, iDense) in AStranslate.items():
            for (jDense, j) in dest.items():
                for (kDense, k) in dest.items():
                    if L_C[i][j] + L_S[j][k] > latency_slo:
                        indexDense = jDense * len(destDense) + kDense
                        M.constraint(f"P^S_{i},{j},{k}", P_S.index(iDense,indexDense), Domain.equalsTo(0.0))
        """

        # !!! XXX: Routing is not adjusted to horizon control!
        if withRouting and False:
            # (14) PS_i,j,k: whether i synchronously forwards its PUTs to k via j; only permitted if a VM at i can forward data to the storage service at j via a VM at k within the latency_slo
            PS = M.variable("PS", [len(AS), len(destDense), len(destDense)], Domain.binary())

            # (15) PA_i,j,k,m: whether i’s PUTs are asynchronously forwarded to m via j and k
            PA = M.variable("PA", [len(AS), len(destDense), len(destDense), len(destDense)], Domain.binary())

            # (16) F_i,j,k: i forwards its puts via j to k
            F = M.variable("F", [len(AS), len(destDense), len(destDense)], Domain.binary())

        # (17) C is a replica
        if not access_cost_heuristic:
            C = M.variable("C", [len(T),len(destDense)], Domain.binary())

        # New variables for migration from object store j to object store i at time t
        # The linear domain >= 0 is sufficient, as we are enforcing the sum of migrations to be >= 1
        Migration = M.variable("Migration", [len(T), len(destDense), len(destDense)], Domain.greaterThan(0))

        # ** Constraints **
        # (27) Failure: Assign f+1 replicas against failure
        for t in T:
            # Simply creating constraints for each slice of R for time t
            # XXX: Different from SpanStore each application has to have at least one replica rather than min_f+1
            #constraint = M.constraint(f"27: Failures at time {t}", Expr.sum(R.slice([t,0,0], [t+1, len(ASdense), len(destDense)]).reshape(R.getShape()[1], R.getShape()[2]),1), Domain.greaterThan(f+1) )
            constraint = M.constraint(f"27: Failures at time {t}", Expr.sum(R.slice([t,0,0], [t+1, len(ASdense), len(destDense)]).reshape(R.getShape()[1], R.getShape()[2]),1), Domain.greaterThan(1))
            if verbose > 2:
                print(f"{constraint.toString()}")

            if not access_cost_heuristic:
                # XXX: Different from SpanStore we want to have min_f to max_f replicas in total
                constraint = M.constraint(f"Extra constraint for min_f+1 <= replicas at time {t}", Expr.sum(C.slice([t,0], [t+1, C.getShape()[1]]).reshape(C.getShape()[1])), Domain.greaterThan(min_f+1))
                constraint = M.constraint(f"Extra constraint for max_f+1 >= replicas at time {t}", Expr.sum(C.slice([t,0], [t+1, C.getShape()[1]]).reshape(C.getShape()[1])), Domain.lessThan(max_f+1))

        # (29) j s a replica if it is a GET/PUT replica for any i in the access set
        # Reformulated to common conditional structure: sum(Rij) + M*(1-Cj) >= 1
        # For storage cost
        if not access_cost_heuristic:
            LargeInt = len(destDense)**2
            for t in T:
                C_expr = Expr.add(
                    Expr.sum(R.slice([t,0,0], [t+1, R.getShape()[1], R.getShape()[2]]).reshape(R.getShape()[1], R.getShape()[2]),0),
                    Expr.mul(-LargeInt, C.slice([t,0], [t+1, C.getShape()[1]]).reshape(C.getShape()[1]))
                )
                constraint = M.constraint(f"29: Cj is a replica at time {t}", C_expr, Domain.lessThan(0))
                if verbose > 2:
                    print(f"{constraint.toString()}")

        # !!! XXX: Routing is not adjusted to horizon control!
        if withRouting and False:
            # (31) i’s PUTs must be synchronously forwarded to k iff k is one of i’s replicas
            # For all i: if R_i,k is set then at least on P_i,j,k must be set
            for i in ASdense:
                Rslice = R.slice([i,0],[i+1, len(dest)]).reshape(len(dest))
                Pslice = PS.slice([i,0,0],[i+1, len(dest), len(dest)]).reshape(len(dest), len(dest))
                e = Expr.add(Rslice, Expr.neg(Expr.sum(Pslice, 0)))
                M.constraint(f"31-{i}: Sync forward", e, Domain.lessThan(0))

                # (33) For every data center in access set, its PUTs much reach every replica
                # For all i in AS: C_m = sum(PS_ijm + sum(PA_ijkm))
                # Equivalent to: 0 = sum(PS_ijm + sum(PA_ijkm)) - C_m

                for m in destDense:
                    PS_slice = PS.slice([i,0,m], [i+1, len(dest), m+1]).reshape(len(dest))
                    PA_slice = PA.slice([i,0,0,m], [i+1,len(dest), len(dest), m+1]).reshape(len(dest), len(dest))

                    e1 = Expr.sum(Expr.sum(PA_slice))
                    e = Expr.sub(Expr.add(Expr.sum(PS_slice), e1), C.index(m))
                    M.constraint(f"33-{i}-{m}", e, Domain.equalsTo(0))

                # (35) Puts from i can be forwarded over the path from j to k as part of either sync. or async. forwarding
                for j in destDense:
                    if i != j:
                        for k in destDense:
                            pass
    
        # **Constraints for migration

        # Migration applies to time steps t+1 to T
        if not access_cost_heuristic:
            for t in T[1:]:

                # If a replica is used, it requires migration
                # sum_i(Migration_tij) - x_tj >= 0
                for (j, jDense) in dest.items():
                    expr = Expr.sub(Expr.sum(Migration.slice([t,0,jDense], [t+1,Migration.getShape()[1], jDense+1]).reshape(Migration.getShape()[2]), 0), C.index(t, jDense))
                    constraint = M.constraint(f"Migration: Replica {j} at time {t} requires migration", expr, Domain.greaterThan(0))

                    # Migration is only possible from a replica used at time t-1
                    # x_(t-1)i - z_tij >= 0
                    for (i, iDense) in dest.items():
                        expr = Expr.sub(C.index(t-1, iDense), Migration.index(t, iDense, jDense))
                        constraint = M.constraint(f"Migration: Replica {j} at time {t} can only be migrated if replica {i} is used at time {t-1}", expr, Domain.greaterThan(0))
            
        # ** Constraint for historical initial state (t=0) and hint for future state (t>0)
        
        # Hint R for all hinted state and for t=0 force R to match assignments of initial state
        for (t, assignments) in enumerate(initialState.assignments):
            for (i, choosenObjectStore) in assignments.items():
                for (j, jDense) in dest.items():
                    hint = j in choosenObjectStore
                    iDense = AStranslate[i]
                    R.index(t, iDense, jDense).setLevel([hint])
                    if t == 0 and len(T) > 1:
                        constraint = M.constraint(f"Initial state: R_{i}_{j} = 1", R.index(t, iDense, jDense), Domain.equalsTo(hint))
                    
        # Hint C for all hinted state and for t=0 force C to match object stores of initial state
        if not access_cost_heuristic:
            for (t, objectStores) in enumerate(initialState.objectStores):
                for o, iDense in dest.items():
                    #iDense = dest[o]
                    choice = o in objectStores
                    C.index(t, iDense).setLevel([choice])
                    if t == 0 and len(T) > 1:
                        constraint = M.constraint(f"Initial state: C_{o} = 1", C.index(t, iDense), Domain.equalsTo(choice))
                
        # Hint Migration for all hinted state
        for (t, migrations) in enumerate(initialState.migrations):
            for (i, migrationAtTime) in migrations.items():
                for (j, migration) in migrationAtTime.items():
                    iDense = dest[i]
                    jDense = dest[j]
                    Migration.index(t, iDense, jDense).setLevel([migration])
                        
        # **Objective**
        # R_ij (PriceGet_j + PriceNet_ji)
        #costGet = Expr.dot(R, PriceGetCoef_M)
        costGetTime = [
            Expr.dot(R, PriceGetCoef_M)
            #Expr.dot(R.slice([t,0,0], [t+1, R.getShape()[1], R.getShape()[2]]).reshape(R.getShape()[1], R.getShape()[2]), PriceGetCoefTime_M[t]) for t in T
        ]
        #pricePutAndStore = np.add(PricePutCoefCSimple, [PriceStorage[j]*Size_total for j in dest.values()])
        #costPutAndStore = Expr.dot(C, pricePutAndStore)
        if access_cost_heuristic:
            costPutAndStoreTime = []
        else:    
            costPutAndStoreTime = [
                Expr.dot(C.slice([t,0], [t+1, C.getShape()[1]]).reshape(C.getShape()[1]), np.add(PricePutCoefCSimpleTime[t], [PriceStorage[j]*Size_total for j in dest.keys()])) for t in T
            ]

        if withMigrationCosts:
            costMigration = [
                    Expr.dot(Migration.slice([t,0,0], [t+1,Migration.getShape()[1], Migration.getShape()[2]]).reshape(Migration.getShape()[1], Migration.getShape()[2]), PriceMigrationCoef) for t in T
            ]
        else:
            costMigration = []

        M.objective("cost", ObjectiveSense.Minimize, Expr.add(costPutAndStoreTime + costMigration + costGetTime))

        if verbose > 2:
            print(f"Access heuristic: {access_cost_heuristic}")
            print("Model:")
            sys.stdout.flush()
            M.writeTaskStream("recedingHorizon", sys.stdout.buffer)
        try:
            if timer is not None:
                timer.continueComputation()
            M.solve()
            if timer is not None:
                timer.stop()
        except Exception as e:
            raise(e)

        a = np.array(R.level())
        a.shape = (len(T), len(ASdense), len(destDense))
        value=M.primalObjValue()
        if access_cost_heuristic:
            # Compute C from R
            c = np.any(a, axis=1)
            
            # Update value by adding storage costs, akin to objective function
            for t in T:
                for (origD, d) in dest.items():
                    # Put and storage price for choosen object stores
                    value += PricePutCoefCSimpleTime[t][d] + PriceStorage[origD]*Size_total
        else:
            c = np.array(C.level())
            c.shape = (C.getShape()[0], C.getShape()[1])
        migration = np.array(Migration.level())
        migration.shape = (Migration.getShape()[0], Migration.getShape()[1], Migration.getShape()[2])

        boolean_threshold = 0.5
        res = OptimizationResult(
            value=value,
            objectStores=[
                { origD for (origD, d) in dest.items() if c[t,d] > boolean_threshold} for t in T
            ],
            assignments=[
                { aTranslated: {origD for (origD, d) in dest.items() if a[t, aID, d] > boolean_threshold } for (aTranslated, aID) in AStranslate.items()} for t in T
            ],
            migrations=[
                { origJ: {origI: migration[t,i,j] for (origI, i) in dest.items() if migration[t, i, j] > boolean_threshold} for (origJ, j) in dest.items()} for t in T
            ]
        )

        if verbose > 0:
            print(f"Objective value: {M.primalObjValue()}")
            for t in T:
                print(f"Time {t}")
                print(f"C: {c[t]}")
                for (j, jDense) in dest.items():
                    if c[t][jDense] > boolean_threshold:
                        for (i, iDense) in AStranslate.items():
                            if a[t,iDense, jDense] > boolean_threshold:
                                print(f"R[{t},{i},{j}] ={a[t, iDense, jDense]}")
                for (i, iDense) in dest.items():
                    for (j, jDense) in dest.items():
                        if migration[t, iDense, jDense] > boolean_threshold:
                            print(f"Migration[{t},{i},{j}] ={migration[t, iDense, jDense]}")

    return res

def genArgs(*, numberOfClientDCs: int, numberofDestDCs: int, f: int, latency_slo: int = 1000, Size_total: int = 42, PriceMax: float = 0.9, PriceCoefNet: float = 1.0, PriceCoefGet = 1.0, PriceCoefPut = 1.0, PriceCoefStorage = 1.0, GetCoef: float = 1.0, PutCoef: float = 1.0, seed = 1, verbose=0):
    random.seed(a=seed)

    #assert(numberOfClientDCs <= numberofDestDCs)

    def rand(a=0, b=1000):
        #return random.randint(a,b)
        return int(a + (b-a)*random.random())

    def randPrice(a=0.1, b=PriceMax, step=0.001):
        #return a + random.randint(a=0, b=int(b/step))*step
        return (a + ((b-a)/step)*step*random.random())

    AS = [i for i in range(numberOfClientDCs)]
    destDCs = [i for i in range(numberOfClientDCs, numberOfClientDCs + numberofDestDCs)]

    args = {
        # Count of objectes in access set
        'Count' : rand(),
        # Duration of epoch
        'T' : rand(),
        # Source data centers: Set of data centers that issue PUTs and GETs
        'AS' : AS,
        # Number of failures to tolerate
        'min_f' : f,
        'max_f' : f,
        # latency_slo on pth percentile of latencies (PUT and GET)
        # TODO What is meant with latency_slo,p : latency_slo?
        'latency_slo' : latency_slo,
        #'p' : 42,
        # pth percentile latency between VMs in data centers i and j, always within latency_slo
        'L_C' : { j:{ i: rand(b=latency_slo) for i in AS + destDCs} for j in AS + destDCs},
        # pth percentile latency between a VM in data center i and the storage service in data center j
        'L_S' : { j:{ i: rand(b=latency_slo) for i in AS + destDCs} for j in AS + destDCs},
        # Total no. of PUTs and GETs issued at data center i across all objectes with access set AS
        'PUTs' : { i: PutCoef*rand() for i in AS},
        'GETs' : { i: GetCoef*rand() for i in AS},
        # Avg. and total size of objects of access set AS
        'Size_total' : Size_total,
        # Prices at data center i per GET, PUT and storage (byte per hour)
        'PriceGet' : { i: PriceCoefGet*randPrice() for i in destDCs},
        'PricePut' : { i: PriceCoefPut*randPrice() for i in destDCs},
        'PriceStorage' : { i: PriceCoefStorage*randPrice() for i in destDCs},
        # Price per byte of network transfer from data center i to j
        'PriceNet': { j:{ i: PriceCoefNet*randPrice() for i in AS + destDCs} for j in AS + destDCs}
    }

    if verbose > 2:
        print(f"PUTs: {args['PUTs']},\nGETs: {args['GETs']},\nPriceGet: {args['PriceGet']},\nPricePut: {args['PricePut']},\nPriceStorage: {args['PriceStorage']},\nPriceNet: {args['PriceNet']}")

    return args

def subArgs(*, orig, numberOfClientDCs: int, numberofDestDCs: int):

    destDCs = list(orig['PricePut'].keys())[:numberofDestDCs]
    AS = orig['AS'][:numberOfClientDCs]
    args = {
        # Count of objectes in access set
        'Count' : orig['Count'],
        # Duration of epoch
        'T' : orig['T'],
        # Source data centers: Set of data centers that issue PUTs and GETs
        'AS' : AS,
        # Number of failures to tolerate
        'min_f' : orig['min_f'],
        'max_f' : orig['max_f'],
        # latency_slo on pth percentile of latencies (PUT and GET)
        # TODO What is meant with latency_slo,p : latency_slo?
        'latency_slo' : orig['latency_slo'],
        #'p' : 42,
        # pth percentile latency between VMs in data centers i and j, always within latency_slo
        'L_C' : { j:{ i: orig['L_C'][j][i] for i in AS + destDCs} for j in AS + destDCs},
        # pth percentile latency between a VM in data center i and the storage service in data center j
        'L_S' : { j:{ i: orig['L_S'][j][i] for i in AS + destDCs} for j in AS + destDCs},
        # Total no. of PUTs and GETs issued at data center i across all objectes with access set AS
        'PUTs' : { i: orig['PUTs'][i] for i in AS},
        'GETs' : { i: orig['GETs'][i] for i in AS},
        # Avg. and total size of objects of access set AS
        'Size_total' : orig['Size_total'],
        # Prices at data center i per GET, PUT and storage (byte per hour)
        'PriceGet' : { i: orig['PriceGet'][i] for i in destDCs},
        'PricePut' : { i: orig['PricePut'][i] for i in destDCs},
        'PriceStorage' : { i: orig['PriceStorage'][i] for i in destDCs},
        # Price per byte of network transfer from data center i to j
        'PriceNet': { j:{ i: orig['PriceNet'][j][i] for i in AS + destDCs} for j in AS + destDCs}
    }

    return args

def benchmark():
    destDCs = [100, 150, 200, 250, 300, 350]
    destDCs = [2]
    clientDCs = [20, 40]
    clientDCs = [2]
    fs = [1]
    if len(sys.argv) > 1:
        problemSizes = [int(i) for i in sys.argv[1:]]
        fs = [0, 1, 2, 3, 4]

    myArgs = genArgs(f=2, numberOfClientDCs=np.max(clientDCs), numberofDestDCs=np.max(destDCs), PriceCoefNet=1, PriceCoefStorage=0.00001, PriceCoefPut=0.01, PriceCoefGet=1, PutCoef=0.01, verbose=0)

    print("f, numberofDestDCs, numberOfClientDCs, execution-time", file=sys.stderr)
    for f in fs:
        for numberofDestDCs in destDCs:
            if numberofDestDCs > f:
                for numberOfClientDCs in clientDCs:
                    args = subArgs(orig=myArgs, numberOfClientDCs=numberOfClientDCs, numberofDestDCs=numberofDestDCs)
                    args['min_f'] = f
                    args['max_f'] = f
                    print(f"===== {datetime.now()} Start: {f}, {numberOfClientDCs}, {numberofDestDCs} ======================")
                    res = recedingHorizonControl(**args, verbose=1, withRouting = False)
                    print(f"{f}, {numberOfClientDCs}, {numberofDestDCs}, {res.time}", file=sys.stderr)
                    print(f"{f}, {numberOfClientDCs}, {numberofDestDCs}, {res.time}")


testArgs = {
    # Count of objectes in access set
    'Count' : 2,
    # Duration of epoch
    'T' : 42,
    # Source data centers: Set of data centers that issue PUTs and GETs
    'AS' : [0, 2],
    # Number of failures to tolerate
    'min_f' : 0,
    'max_f' : 0,
    # latency_slo on pth percentile of latencies (PUT and GET)
    # TODO What is meant with latency_slo,p : latency_slo?
    'latency_slo' : 1000,
    #'p' : 42,
    # pth percentile latency between VMs in data centers i and j
    'L_C' : {0: {0: 1, 1: 1, 2:1}, 1:{0: 1, 1:1, 2:1}, 2:{0:1, 1:1, 2:1}},
    # pth percentile latency between a VM in data center i and the storage service in data center j
    'L_S' : {0: {0: 1, 1: 1, 2:1}, 1:{0: 1, 1:1, 2:1}, 2:{0:1, 1:1, 2:1}},
    # Total no. of PUTs and GETs issued at data center i across all objectes with access set AS
    'PUTs' : [{0: 42, 2:42}],
    'GETs' : [{0: 420, 2:420}],
    # Avg. and total size of objects of access set AS
    'Size_total' : 42,
    # Prices at data center i per GET, PUT and storage (byte per hour)
    'PriceGet' : {0: 0.0004, 1: 0.000125, 2: 0.0005},
    'PricePut' : {0: 0.005, 1: 0.00125, 2: 0.0065},
    'PriceStorage' : {0: 0.023, 1: 0.02, 2: 0.024},
    # Price per byte of network transfer from data center i to j
    'PriceNet': {0: {0: 0, 1: 0.02, 2:0.09}, 1:{0: 0.02, 1:0, 2:0.09}, 2:{0:0.09, 1:0.09, 2:0}}
}

def simpleUseCase(*, timeSlots=3, verbose=0, storagePriceFileName=None, networkPriceFileName=None, selector="", **kwargs):
    args = copy.deepcopy(testArgs)
    args.update(kwargs)
    args.update({"storagePriceFileName": storagePriceFileName, "networkPriceFileName": networkPriceFileName, "selector": selector, "verbose": verbose})
    p=Problem(**args)

    objects = [
        # Different workloads, each with some object size and PUT/Get number of distinct appliction regions over time
        #(1, {0: 1000, 2:10}, {0:300000, 2:20000}),
        #(10, {0: 1000, 2:10}, {0:300000, 2:20000}),
        (0.1, [{0: 50000, 2:50000}], [{0:100, 2:200}]),
    ]

    if storagePriceFileName is not None:
        # Create random workload for each application region
        objects = [
            (size, [{i: np.random.randint(100) for i in p.AS} ], [{i: np.random.randint(10000) for i in p.AS} ]) for (size, _, _) in objects
        ]
    
    # For testing of time horizon, create more time slots while keeping aggregate workload constant
    #timeSlots = 3
    t = 0
    objectsTime = []
    for (size, PUTs, GETs) in objects:
        PUTsTime = [ {i: PUTs[t][i] / timeSlots for i in PUTs[t]} for _ in range(timeSlots)]
        GETsTime = [ {i: GETs[t][i] / timeSlots for i in GETs[t]} for _ in range(timeSlots)]
        objectsTime.append((size, PUTsTime, GETsTime))
        

    # Calculate for different objects
    for (size, PUTs, GETs) in objectsTime:
        
        # XXX: Skipping adjustment for DynamoDB
        # Adjust prices for store 1 (Dynamo DB) by access granularity
        # Size from GB to KB
        # 1KB write chunks
        #writeChunks = size * 1000 * 1000
        # 4KB read chunkcs
        #readChunks = size * 1000 * 1000 / 4
        #args['PricePut'][1] *= writeChunks
        #args['PriceGet'][1] *= readChunks

        #args["Size_total"] = size * args["Count"]
        #args['PUTs'] = PUTs
        #args['GETs'] = GETs
        p.setSize_total(size * p.Count)
        p.setPUTs(PUTs)
        p.setGETs(GETs)

        res = recedingHorizonControl(**p.__dict__, withRouting = False)
        print(res.time)
        if verbose > 1:
            print(res)

        # Solving with initial state from previous solution
        p.initialState=res
        res2 = recedingHorizonControl(**p.__dict__, withRouting = False)
        print(res2.time)
        if verbose > 0:
            print(res2)


        if storagePriceFileName is None:
            # XXX: Only works for test data
            # Solving with historical state different from optimal solution
            # Be careful to allow a feasible solution!
            historicalState = OptimizationResult(
                objectStores=[
                    {2} # Force object store 2 as historical choice
                ],
                assignments=[
                    {
                        0: {2}, # Assign application region 0 to object store 2
                        2: {2} # Assign application region 2 to object store 2
                    }
                ]
            )

            # Solving with historical state
            p.initialState=historicalState
            res3 = recedingHorizonControl(**p.__dict__, withRouting = False)
            print(res3.time)
            if verbose > 0:
                print(res3)

import json
def simpleTimeResolution():
    #args = copy.deepcopy(testArgs)
    p=Problem(**copy.deepcopy(testArgs))

    objects = [
        # Different workloads, each with some object size and PUT/Get number of distinct appliction regions over time
        #(1, {0: 1000, 2:10}, {0:300000, 2:20000}),
        #(10, {0: 1000, 2:10}, {0:300000, 2:20000}),
        (0.1, [{0: 50000, 2:50000}], [{0:100, 2:200}]),
    ]
    
    # For testing of time horizon, create more time slots while keeping aggregate workload constant
    timeSlots = 3
    t = 0
    objectsTime = []
    for (size, PUTs, GETs) in objects:
        PUTsTime = [ {i: PUTs[t][i] / timeSlots for i in PUTs[t]} for _ in range(timeSlots)]
        GETsTime = [ {i: GETs[t][i] / timeSlots for i in GETs[t]} for _ in range(timeSlots)]
        objectsTime.append((size, PUTsTime, GETsTime))
        
    #with open("../../data/workloadTrace.json", "r") as f:
    with open("./data/workloadTrace.json", "r") as f:
        workloadTraceTimeResolutions = json.load(f)

    resultsTimeResolutions = dict()
    # Calculate for different objects
    for (timeResolution, workloadTrace) in workloadTraceTimeResolutions.items():

        def convertAS(k):
            if "eu-west-1" in k:
                return 0
            elif "us-east-1" in k:
                return 2
            else:
                return -1
        size_total = list(workloadTrace["Size_total"]["0.0"].values())[0]
        PUTs = workloadTrace["PUTs"].values()
        PUTs = [ {convertAS(k): v for k, v in PUT.items()} for PUT in PUTs]
        GETs = workloadTrace["GETs"].values()
        GETs = [ {convertAS(k): v for k, v in GET.items()} for GET in GETs]
        
        p.setSize_total(size_total)
        p.setPUTs(PUTs)
        p.setGETs(GETs)

        res = recedingHorizonControl(**p.__dict__, verbose=0, withRouting = False, withMigrationCosts=False)
        resultsTimeResolutions[timeResolution] = res

    for (timeResolution, res) in resultsTimeResolutions.items():
        print(f"{timeResolution}: {res.value}")


if __name__ == "__main__":
    #simpleTimeResolution()
    verbose = 1
    storagePriceFileName = "data/storage_price_filtered.csv"
    networkPriceFileName = "data/network_cost copy.csv"
    selector = "aws"
    f = 0
    print(f"For replication factor f = {f}")
    simpleUseCase(timeSlots=1, selector="aws", storagePriceFileName=storagePriceFileName, networkPriceFileName=networkPriceFileName, min_f=f, max_f=f, verbose=verbose)

    f = 1
    print(f"For replication factor f = {f}")
    simpleUseCase(timeSlots=1, selector="aws", storagePriceFileName=storagePriceFileName, networkPriceFileName=networkPriceFileName, min_f=f, max_f=f, verbose=verbose)

    #spanStore(**testArgs)
    #spanStore(**genArgs(numberOfClientDCs=2, numberofDestDCs=2, f=0))
    #spanStore(**genArgs(numberOfClientDCs=2, numberofDestDCs=2, f=1))

    #problemSizes = [3, 5, 10, 100, 200, 1000, 2000, 5000, 10000]
    #problemSizes = [500]

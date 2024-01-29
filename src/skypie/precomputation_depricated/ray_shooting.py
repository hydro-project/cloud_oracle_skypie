import torch

def precompute_ray_shooting(*, M: "torch.Tensor", p: "torch.Tensor", verbose=0) -> "torch.Tensor":

    if verbose >= 0:
        assert p[0] == 1, "p[0] must be 1"

    if verbose > 1:
        print("Precomputing ray shooting:")
        print("p = {}".format(p))
    if verbose > 2:
        print("M = {}".format(M))

    T1 = torch.matmul(M,p)

    return T1

def ray_shooting(*, M: "torch.Tensor", r: "torch.Tensor", p: "torch.Tensor" = None, stable: bool=False, verbose=0, T1: "torch.Tensor" = None) -> "torch.Tensor":
    """
    Format: b+Ax>=0
    -Ax <= b
    This function computes the index of the inequality whose halfplane is hit first by the ray from p in direction r.
    Alternatively to passing p it's precomputed matrix multiplication with M can be passed as T1.
    As always, the inequalities are given by b+Ax>=0, where the first column of M is b and the remaining columns are A, i.e., M[:,0] = b and M[:,1:] = A.
    M is a 2D tensor of shape (n, d+1), where n is the number of inequalities and d is the dimension of the space.
    p is a 1D tensor of shape (d,) and r is a 1D tensor of shape (d,).
    The function returns a 1D tensor of shape (1,) containing the index of the inequality whose halfplane is hit first by the ray.
    This function is stable, i.e., it deterministically breaks ties.

    cdd takes a bit of a shortcut for calculating alpha.
    Rather than dividing and then multiplying by -1, they use the inverse.

    XXX: Crucially, p[0] = 1 and r[0] = 0 !
    XXX: Stable does not work for linearly dependent planes!

    XXX: This function does not yet support batching.
    XXX: Check behaviour when the ray is parallel to a hyperplane.

    This algorithm is based on cddlib's ray_shooting function.
    """

    if verbose >= 0:
        if p is not None:
            assert p[0] == 1, "p[0] must be 1"
        assert r[0] == 0, "r[0] must be 0"

    if verbose > 1:
        print("Ray shooting:")
        print("p = {}".format(p))
        print("r = {}".format(r))
    if verbose > 2:
        print("M = {}".format(M))
    
    # Compute T1 if not passed
    # XXX: Remember to maintain precompute_ray_shooting if this changes
    if T1 is None:
        T1 = torch.matmul(M,p)
    T2 = torch.matmul(M,r)
    alpha = T2/T1 # Distance to the intersection of the ray with the halfplane, by some "projection"
    # Traditional distance calculation
    #alpha = -(T1/T2)
        
    insideMask = (T1 > 0) # Point is inside the halfspace of the inequalities

    if verbose >= 0:
        #assert insideMask.any(), "Point is outside the halfspace of all inequalities."
        if not insideMask.all():
            print(insideMask)
        # XXX: For debugging we are very strict
        assert insideMask.all(), "Point is outside the halfspace of some inequalities."
        
        pointOutside = torch.nonzero(~insideMask).squeeze(dim=-1)
        if pointOutside.shape[0] == M.shape[0]:
            print("Point is outside the halfspace of all inequalities.")
        #if pointOutside.shape[0] > 0:
        #    print("Point is outside the halfspace of the following inequalities: {}".format(pointOutside))

    if not stable:
        return alpha[insideMask].argmin(dim=-1) # Return the index of the first hit halfplane. This is not stable, i.e., it does not break ties deterministically.
    else:
        minAlpha = torch.min(alpha[insideMask]) #alpha[insideMask].min(dim=-1) # Find the minimum alpha
        indexMask = (alpha == minAlpha) & (insideMask) # Mask of the minimum alpha(s)
        minIndexes = torch.nonzero(indexMask).squeeze(dim=-1) # Get the indexes of the minimum alpha
        if minIndexes.shape[0] > 1: # Break ties deterministically
            tie = M[minIndexes] / (T1[minIndexes].unsqueeze(dim=-1))
            minTie = 0
            for i in range(1, tie.shape[0]):
                for j in range(tie.shape[1]):
                    if tie[minTie][j] < tie[i][j]: # minTie is lexically smaller than i, done with i and continue keep minTie
                        break
                    elif tie[minTie][j] > tie[i][j]: # min is lexically greater than i, done with i and continue with __i__
                        minTie = i
                        break
                    else: # when equal continue comparing
                        pass

            minIndexes = minIndexes[minTie]
            
        return minIndexes

if __name__ == "__main__":
    
    testCase = 3
    verbose = 3
    no_dim = 3
    no_inequalities = 4
    doNonStable = True
    expected = None

    if testCase in [0,1,2]:
        inequalitiesCube = [
            [1, -1, 0], # 1 - x0 >=0 -> x0 <= 1 Vertical plane at x0 = 1
            [1, 0, -1], # 1 - x1 >=0 -> x1 <= 1 Horizontal plane at x1 = 1
            [1.5, -1, -1], # 1 - x0 - x1 >=0 -> x0 + x1 <= 1 Diagonal plane at x0 + x1 = 1
        ]
        inequalities = torch.tensor(inequalitiesCube, dtype=torch.float64)
        # Point is origin
        p = torch.tensor([1,0,0], dtype=torch.float64)

        if testCase == 0:
            # Diagonal ray
            r = torch.tensor([0,1,1], dtype=torch.float64)
            expected = [2, 2]
        elif testCase == 1:
            # Horizontal ray
            r = torch.tensor([0,1,0], dtype=torch.float64)
            expected = [0, 0]
        elif testCase == 2:
            # Vertical ray
            r = torch.tensor([0,0,1], dtype=torch.float64)
            expected = [1, 1]

    elif testCase == 4:
        # Ax <= b
        inequalities = torch.full(fill_value=-1, size=(no_inequalities, no_dim), dtype=torch.double)
        for i in range(1, no_inequalities+1):
            inequalities[-i, 0] = i
        p = torch.tensor([1, 0, 0], dtype=torch.double)
        r = torch.tensor([0, 1, 0], dtype=torch.double)
        expected = [no_inequalities-1, no_inequalities-1]

    print("Stable ray shooting")
    res = ray_shooting(M=inequalities, p=p, r=r, verbose=verbose, stable=True)
    print("result=", res)
    if expected is not None and expected[0] != res:
        print(f"ERROR: result is not as expected: expected={expected[0]}")

    if doNonStable:
        print("Non stable ray shooting")
        res2 = ray_shooting(M=inequalities, p=p, r=r, stable=False)
        print("result=", res2)
        if expected is not None and len(expected) > 1 and expected[1] != res2:
            print(f"ERROR: result not as expected, expected={expected[1]}")
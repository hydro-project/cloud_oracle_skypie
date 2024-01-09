import numpy as np
from typing import Tuple
from skypie.util.my_dataclasses import NormalizationType

def dataNormalization(*, type: NormalizationType, inequalities: "np.ndarray", interiorPoint: "np.ndarray" = None) -> Tuple["np.ndarray","np.ndarray"]:
    """
    Poor man's data normalization.
    Our data has many orders of magnitude difference, which can cause trouble for the optimizer.
    For our particular use case and data, simply projecting by log10 seems sufficient.
    """

    assert (inequalities >= 0).all, "Inequalities must be non-negative!"

    #normalizedZero = np.zeros_like(inequalities[0])

    if type == NormalizationType.No:
        normalizedInequalities = np.array(inequalities)
        normalizedInteriorPoint = np.array(interiorPoint) if interiorPoint else None

        return normalizedInequalities, normalizedInteriorPoint

    elif type == NormalizationType.log10All:
        normalizedInequalities = np.array(inequalities)
        normalizedInteriorPoint = np.array(interiorPoint) if interiorPoint else None

        mask = normalizedInequalities != 0 # Avoid 0, and only scale x_1, ..., x_n-1, i.e., not x_0 nor b
        normalizedInequalities *= 10 # Avoid 1
        np.log10(normalizedInequalities, where=mask, out=normalizedInequalities)
        minimum = normalizedInequalities.min()
        if minimum < 0:
            normalizedInequalities[mask] -= minimum
        
        if interiorPoint:
            if (interiorPoint > 0).any():
                np.log10(normalizedInteriorPoint, where=normalizedInteriorPoint != 0, out=normalizedInteriorPoint)
            else:
                # XXX: relying on hacked interior point
                pass
    else:
        raise RuntimeError("Normalization type not found!")

    if (normalizedInequalities < 0).any():
        raise RuntimeError("Normalizing inequalities failed: negative values!")
    
    if np.inf in normalizedInequalities or np.NINF in normalizedInequalities or np.nan in normalizedInequalities:
            raise RuntimeError("Normalizing inequalities failed!")
    
    if interiorPoint:
        if np.inf in normalizedInteriorPoint or np.NINF in normalizedInteriorPoint or np.nan in normalizedInteriorPoint:
            raise RuntimeError("Normalizing interior point failed!")

    return normalizedInequalities, interiorPoint
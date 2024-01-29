import unittest
import numpy as np
from skypie.precomputation_depricated.redundancy_elimination_cvxpy import redundancyElimination as redundancyEliminationCVXPY
from skypie.precomputation_depricated.redundancy_elimination_mosek import redundancyElimination as redundancyEliminationMosek

class TestRedundancyElimination(unittest.TestCase):
    def test_redundancy_elimination_cvxpy(self):
        coefficients = [
            [0.5, 1.0], # Non redundant
            [2.0, 2.0],  # Redundant
            [1.0, 0.5], # Non redundant
        ]
        inequalities = np.array([
            [0] + [c * -1 for c in coefficients_i] + [1] for coefficients_i in coefficients
        ])

        res = redundancyEliminationCVXPY(inequalities=inequalities)
        nonredundant = [i for i, (r, _) in enumerate(res) if r == True]

        expected = [0, 2]
        self.assertEqual(nonredundant, expected, f"Expected {expected}, got {nonredundant}")

    def test_redundancy_elimination_mosek(self):
        coefficients = [
            [0.5, 1.0], # Non redundant
            [2.0, 2.0],  # Redundant
            [1.0, 0.5], # Non redundant
        ]
        inequalities = np.array([
            [0] + [c * -1 for c in coefficients_i] + [1] for coefficients_i in coefficients
        ])

        res = redundancyEliminationMosek(inequalities=inequalities)
        nonredundant = [i for i, (r, _) in enumerate(res) if r == True]

        expected = [0, 2]
        self.assertEqual(nonredundant, expected, f"Expected {expected}, got {nonredundant}")

if __name__ == "__main__":
    unittest.main()

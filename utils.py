import numpy as np
from mytypes import ArrayLike


def cross_product_matrix(n: ArrayLike, debug: bool = True) -> np.ndarray:
    assert len(n) == 3, f"utils.cross_product_matrix: Vector not of length 3: {n}"
    vector = np.array(n, dtype=float).reshape(3)
    
    n1 = vector[0]
    n2 = vector[1]
    n3 = vector[2]
    S = np.array([[0, -n3, n2],
                  [n3, 0, -n1],
                  [-n2, n1, 0]])  # Create the cross product matrix
    
    if debug:
        assert S.shape == (
            3,
            3,
        ), f"utils.cross_product_matrix: Result is not a 3x3 matrix: {S}, \n{S.shape}"
        assert np.allclose(
            S.T, -S
        ), f"utils.cross_product_matrix: Result is not skew-symmetric: {S}"

    return S

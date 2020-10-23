import numpy as np
import utils


def quaternion_product(ql: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """Perform quaternion product according to either (10.21) or (10.34).

    Args:
        ql (np.ndarray): Left quaternion of the product of either shape (3,) (pure quaternion) or (4,)
        qr (np.ndarray): Right quaternion of the product of either shape (3,) (pure quaternion) or (4,)

    Raises:
        RuntimeError: Left or right quaternion are of the wrong shape
        AssertionError: Resulting quaternion is of wrong shape

    Returns:
        np.ndarray: Quaternion product of ql and qr of shape (4,)s
    """

    if ql.shape == (4,):
        eta_left = ql[0]
        epsilon_left = ql[1:].reshape((3, 1))
    elif ql.shape == (3,):
        eta_left = 0
        epsilon_left = ql.reshape((3, 1))
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, left quaternion shape incorrect: {ql.shape}"
        )

    if qr.shape == (4,):
        eta_right = qr[0]
        q_right = qr.copy()
    elif qr.shape == (3,):
        epsilon_right = qr[1:].reshape((3,1))
        q_right = np.concatenate(([0], qr))
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, right quaternion wrong shape: {qr.shape}"
        )
    
    epsMatrix = np.zeros((4,4))
    epsMatrix[0,1:] = -epsilon_left.T
    epsMatrix[1:,0] = epsilon_left.T
    epsMatrix[1:,1:] = utils.cross_product_matrix(epsilon_left)
    quaternion = (eta_left * np.eye(4) + epsMatrix) @ q_right # Formula (10.34)

    # Ensure result is of correct shape
    quaternion = quaternion.ravel()
    assert quaternion.shape == (
        4,
    ), f"utils.quaternion_product: Quaternion multiplication error, result quaternion wrong shape: {quaternion.shape}"
    return quaternion


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray, debug: bool = True
) -> np.ndarray:
    """Convert a quaternion to a rotation matrix

    Args:
        quaternion (np.ndarray): Quaternion of either shape (3,) (pure quaternion) or (4,)
        debug (bool, optional): Debug flag, could speed up by setting to False. Defaults to True.

    Raises:
        RuntimeError: Quaternion is of the wrong shape
        AssertionError: Debug assert fails, rotation matrix is not element of SO(3)
 epsilon_right = np.zeros((3,1))
    if qr.shape == (4,):
        eta_right = qr[0]
        q_right = qr.copy()
    elif qr.shape == (3,):
        epsilon_right = qr[1:].reshape((3,1))
        q_right = np.conc
    Returns:
        np.ndarray: Rotation matrix of shape (3, 3)
    """
    if quaternion.shape == (4,):
        eta = quaternion[0]
        epsilon = quaternion[1:]
    elif quaternion.shape == (3,):
        eta = 0
        epsilon = quaternion.copy()
    else:
        raise RuntimeError(
            f"quaternion.quaternion_to_rotation_matrix: Quaternion to multiplication error, quaternion shape incorrect: {quaternion.shape}"
        )
    
    S = utils.cross_product_matrix(epsilon)
    R = np.identity(3) + 2*eta*S + 2*S*S # Convert from quaternion to rotation matrix

    if debug:
        assert np.allclose(
            np.linalg.det(R), 1
        ), f"quaternion.quaternion_to_rotation_matrix: Determinant of rotation matrix not close to 1"
        assert np.allclose(
            R.T, np.linalg.inv(R)
        ), f"quaternion.quaternion_to_rotation_matrix: Transpose of rotation matrix not close to inverse"

    return R


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion into euler angles

    Args:
        quaternion (np.ndarray): Quaternion of shape (4,)

    Returns:
        np.ndarray: Euler angles of shape (3,)
    """

    assert quaternion.shape == (
        4,
    ), f"quaternion.quaternion_to_euler: Quaternion shape incorrect {quaternion.shape}"

    quaternion_squared = quaternion ** 2
    eta_squared = quaternion_squared[0]
    epsilon1_squared = quaternion_squared[1]
    epsilon2_squared = quaternion_squared[2]
    epsilon3_squared = quaternion_squared[3]
    
    eta = quaternion[0]
    epsilon1 = quaternion[1]
    epsilon2 = quaternion[2]
    epsilon3 = quaternion[3]
    
    phi = np.arctan2(2*(epsilon3*epsilon2 + eta*epsilon1), eta_squared - epsilon1_squared - epsilon2_squared + epsilon3_squared)
    theta = np.arcsin(2*(eta*epsilon2 - epsilon1*epsilon3))
    psi = np.arctan2(2*(epsilon1*epsilon2 + eta*epsilon3), eta_squared + epsilon1_squared - epsilon2_squared - epsilon3_squared)

    euler_angles = np.array([phi, theta, psi])
    assert euler_angles.shape == (
        3,
    ), f"quaternion.quaternion_to_euler: Euler angles shape incorrect: {euler_angles.shape}"

    return euler_angles


def euler_to_quaternion(euler_angles: np.ndarray) -> np.ndarray:
    """Convert euler angles into quaternion

    Args:
        euler_angles (np.ndarray): Euler angles of shape (3,)

    Returns:
        np.ndarray: Quaternion of shape (4,)
    """

    assert euler_angles.shape == (
        3,
    ), f"quaternion.euler_to_quaternion: euler_angles shape wrong {euler_angles.shape}"

    half_angles = 0.5 * euler_angles
    c_phi2, c_theta2, c_psi2 = np.cos(half_angles)
    s_phi2, s_theta2, s_psi2 = np.sin(half_angles)

    quaternion = np.array(
        [
            c_phi2 * c_theta2 * c_psi2 + s_phi2 * s_theta2 * s_psi2,
            s_phi2 * c_theta2 * c_psi2 - c_phi2 * s_theta2 * s_psi2,
            c_phi2 * s_theta2 * c_psi2 + s_phi2 * c_theta2 * s_psi2,
            c_phi2 * c_theta2 * s_psi2 - s_phi2 * s_theta2 * c_psi2,
        ]
    )

    assert quaternion.shape == (
        4,
    ), f"quaternion.euler_to_quaternion: Quaternion shape incorrect {quaternion.shape}"

    return quaternion

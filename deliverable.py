from typing import Tuple

def position_distance_penalty(
    position: Tuple[float, float, float],
    target_position: Tuple[float, float, float],
    penalty_scale: float = 1.0
) -> float:
    """
    Compute a position-based distance penalty for a quadcopter.

    The penalty is proportional to the Euclidean distance between the current position
    and the target position, scaled by `penalty_scale`. This can be used as a negative
    reward in reinforcement learning to encourage the quadcopter to minimize its
    distance from the target.

    Args:
        position (Tuple[float, float, float]): The current (x, y, z) position of the quadcopter.
        target_position (Tuple[float, float, float]): The desired (x, y, z) target position.
        penalty_scale (float, optional): Scaling factor for the penalty. Default is 1.0.

    Returns:
        float: The computed penalty (always <= 0).
    """
    dx = position[0] - target_position[0]
    dy = position[1] - target_position[1]
    dz = position[2] - target_position[2]
    distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    penalty = -penalty_scale * distance
    return penalty

# Example usage and simple test cases
if __name__ == "__main__":
    # Test 1: Zero distance
    pos = (0.0, 0.0, 0.0)
    target = (0.0, 0.0, 0.0)
    assert position_distance_penalty(pos, target) == 0.0

    # Test 2: Unit distance
    pos = (1.0, 0.0, 0.0)
    target = (0.0, 0.0, 0.0)
    assert abs(position_distance_penalty(pos, target) + 1.0) < 1e-6

    # Test 3: Arbitrary distance and scale
    pos = (1.0, 2.0, 2.0)
    target = (4.0, 6.0, 2.0)
    expected_dist = ((1-4)**2 + (2-6)**2 + (2-2)**2) ** 0.5  # 5.0
    assert abs(position_distance_penalty(pos, target, penalty_scale=2.0) + 2.0 * expected_dist) < 1e-6

    print("All tests passed.")
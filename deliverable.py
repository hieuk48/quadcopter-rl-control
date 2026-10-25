"""
Quadcopter RL Control - Survival Bonus & Terminal Condition Implementation

Implements survival bonus and terminal condition rules for the quadcopter RL environment.
This patch is intended for integration into the eliezerkirubi-sys/quadcopter-rl-control project,
as per Issue #8: https://github.com/eliezerkirubi-sys/quadcopter-rl-control/issues/8

Assumptions:
- The environment class is named `QuadcopterEnv` and follows OpenAI Gym API.
- The state includes position (x, y, z) and orientation (roll, pitch, yaw).
- The step() method returns (obs, reward, done, info).
- Survival bonus is a constant added each step unless terminal condition is met.
- Terminal conditions: out-of-bounds position or orientation.

No external dependencies are used.
"""

from typing import Tuple, Any, Dict

class QuadcopterEnv:
    """
    Quadcopter Reinforcement Learning Environment with Survival Bonus and Terminal Conditions.
    """

    # Define environment boundaries and orientation limits
    X_LIMIT: float = 10.0  # meters
    Y_LIMIT: float = 10.0  # meters
    Z_LIMIT: float = 20.0  # meters (altitude)
    MIN_Z: float = 0.0     # ground level
    ROLL_LIMIT: float = 1.0  # radians (~57 deg)
    PITCH_LIMIT: float = 1.0  # radians (~57 deg)
    YAW_LIMIT: float = 3.1416  # radians (180 deg)

    SURVIVAL_BONUS: float = 0.1  # reward per step for surviving

    def __init__(self) -> None:
        """
        Initialize the quadcopter environment.
        """
        self.state = self._reset_state()
        self.steps = 0

    def _reset_state(self) -> Dict[str, float]:
        """
        Reset the quadcopter state to initial conditions.

        Returns:
            dict: State dictionary with position and orientation.
        """
        return {
            'x': 0.0,
            'y': 0.0,
            'z': 1.0,  # start 1m above ground
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            # ... other state variables as needed
        }

    def reset(self) -> Dict[str, float]:
        """
        Reset the environment to initial state.

        Returns:
            dict: Initial observation/state.
        """
        self.state = self._reset_state()
        self.steps = 0
        return self.state.copy()

    def step(self, action: Any) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Apply an action to the environment.

        Args:
            action (Any): The action to apply.

        Returns:
            Tuple containing:
                - observation (dict): The next state.
                - reward (float): The reward for this step.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional info.
        """
        # --- Simulate dynamics (placeholder) ---
        self._apply_action(action)
        self.steps += 1

        # --- Check terminal conditions ---
        done, reason = self._check_terminal()

        # --- Compute reward ---
        reward = self._compute_reward(done)

        info = {'steps': self.steps}
        if done:
            info['termination_reason'] = reason

        return self.state.copy(), reward, done, info

    def _apply_action(self, action: Any) -> None:
        """
        Update the state based on the action.
        Placeholder for actual quadcopter dynamics.

        Args:
            action (Any): The action to apply.
        """
        # For demonstration, this is a no-op.
        # In practice, update self.state based on action and physics.
        pass

    def _check_terminal(self) -> Tuple[bool, str]:
        """
        Check if the current state meets any terminal condition.

        Returns:
            Tuple[bool, str]: (done, reason)
        """
        x, y, z = self.state['x'], self.state['y'], self.state['z']
        roll, pitch, yaw = self.state['roll'], self.state['pitch'], self.state['yaw']

        # Out-of-bounds position
        if abs(x) > self.X_LIMIT or abs(y) > self.Y_LIMIT:
            return True, 'out_of_bounds_xy'
        if z < self.MIN_Z or z > self.Z_LIMIT:
            return True, 'out_of_bounds_z'

        # Excessive orientation
        if abs(roll) > self.ROLL_LIMIT:
            return True, 'excessive_roll'
        if abs(pitch) > self.PITCH_LIMIT:
            return True, 'excessive_pitch'
        if abs(yaw) > self.YAW_LIMIT:
            return True, 'excessive_yaw'

        return False, ''

    def _compute_reward(self, done: bool) -> float:
        """
        Compute the reward for the current step.

        Args:
            done (bool): Whether the episode has ended.

        Returns:
            float: The reward.
        """
        if done:
            # No survival bonus if terminal state
            return 0.0
        else:
            # Survival bonus for staying alive
            return self.SURVIVAL_BONUS

    # Optionally, add more methods as needed for the environment

# --- Example test cases ---

def _test_survival_bonus() -> None:
    env = QuadcopterEnv()
    obs = env.reset()
    total_reward = 0.0
    for _ in range(5):
        obs, reward, done, info = env.step(action=None)
        total_reward += reward
        assert not done, "Should not be done in first 5 steps"
        assert reward == env.SURVIVAL_BONUS, "Survival bonus not applied"
    print("Survival bonus test passed.")

def _test_terminal_conditions() -> None:
    env = QuadcopterEnv()
    obs = env.reset()
    # Force out-of-bounds x
    env.state['x'] = env.X_LIMIT + 1.0
    obs, reward, done, info = env.step(action=None)
    assert done, "Should be done when out of bounds"
    assert reward == 0.0, "No reward on terminal state"
    assert info['termination_reason'] == 'out_of_bounds_xy'
    print("Terminal condition (x) test passed.")

    obs = env.reset()
    env.state['z'] = env.MIN_Z - 0.1
    obs, reward, done, info = env.step(action=None)
    assert done, "Should be done when below ground"
    assert info['termination_reason'] == 'out_of_bounds_z'
    print("Terminal condition (z) test passed.")

    obs = env.reset()
    env.state['roll'] = env.ROLL_LIMIT + 0.1
    obs, reward, done, info = env.step(action=None)
    assert done, "Should be done when excessive roll"
    assert info['termination_reason'] == 'excessive_roll'
    print("Terminal condition (roll) test passed.")

if __name__ == "__main__":
    _test_survival_bonus()
    _test_terminal_conditions()
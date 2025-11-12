"""
Modern custom reward functions for Rocket League bot training.
Designed to encourage smart ball control and discourage bad habits.
"""
import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED


class SpeedTowardBallReward(RewardFunction):
    """
    Rewards velocity component toward the ball.
    Encourages approaching and chasing the ball.
    """
    
    def __init__(self, min_distance: float = 100.0):
        """
        Args:
            min_distance: Minimum distance to give reward (prevents wiggling on ball)
        """
        super().__init__()
        self.min_distance = min_distance

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Vector from player to ball
        pos_diff = state.ball.position - player.car_data.position
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # Don't reward when too close (prevents jittering)
        if dist_to_ball < self.min_distance:
            return 0.0
        
        # Direction to ball (normalized)
        dir_to_ball = pos_diff / dist_to_ball
        
        # Project velocity onto direction to ball
        speed_toward_ball = np.dot(player.car_data.linear_velocity, dir_to_ball)
        
        # Only reward positive speed (moving toward ball)
        # Normalize by max car speed to get [0, 1] range
        return max(0.0, speed_toward_ball / CAR_MAX_SPEED)


class FlipDisciplineReward(RewardFunction):
    """
    Discourages flip spam during mid-range ball approach.
    
    Strategy:
    - ALLOW flips when very close (<400uu) - good for shots
    - PENALIZE flips in mid-range (400-2000uu) - loses control
    - ALLOW flips when far (>2000uu) - good for recovery/speed
    """
    
    def __init__(self, close_distance: float = 400, far_distance: float = 2000, penalty: float = 2.0):
        """
        Args:
            close_distance: Distance below which flips are allowed (attacking)
            far_distance: Distance above which flips are allowed (recovering)
            penalty: Penalty magnitude for bad flips
        """
        super().__init__()
        self.close_distance = close_distance
        self.far_distance = far_distance
        self.penalty = penalty
        self.was_on_ground = {}  # Track per-player ground state
    
    def reset(self, initial_state: GameState):
        self.was_on_ground.clear()
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if len(previous_action) < 6:
            return 0.0
        
        jump_pressed = previous_action[5] > 0.5  # Index 5 = jump
        is_on_ground = player.on_ground
        player_id = id(player)
        
        # Initialize tracking for this player
        if player_id not in self.was_on_ground:
            self.was_on_ground[player_id] = is_on_ground
        
        # Detect potential flip: airborne + jump pressed
        if not is_on_ground and jump_pressed:
            # Calculate distance to ball
            pos_diff = state.ball.position - player.car_data.position
            dist_to_ball = np.linalg.norm(pos_diff)
            
            # DANGER ZONE: mid-range approach
            if self.close_distance < dist_to_ball < self.far_distance:
                self.was_on_ground[player_id] = is_on_ground
                return -self.penalty
        
        # Update ground state
        self.was_on_ground[player_id] = is_on_ground
        return 0.0


class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0
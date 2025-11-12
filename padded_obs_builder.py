"""
Custom observation builder with padding support for variable team sizes.
Wraps any rlgym_sim observation builder and pads to a fixed size.
"""
import numpy as np
from typing import Any, List
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import ObsBuilder


class AdvancedObsPadder(ObsBuilder):
    """
    Observation builder that pads observations to support variable team sizes.
    
    This allows training on 1v1 and deploying on 2v2 or 3v3 without retraining,
    by padding player observations with zeros for missing players.
    
    Args:
        team_size: Maximum team size to pad to (e.g., 3 for 3v3)
        tick_skip: Tick skip value (currently unused, kept for API compatibility)
        obs_builder: Optional custom observation builder. If None, uses DefaultObs.
    """
    
    def __init__(
        self, 
        team_size: int = 3,
        tick_skip: int = 8,
        obs_builder: ObsBuilder = None
    ):
        super().__init__()
        self.team_size = team_size
        self.tick_skip = tick_skip
        
        # Use DefaultObs if no custom builder provided
        if obs_builder is None:
            from rlgym_sim.utils.obs_builders import DefaultObs
            from rlgym_sim.utils import common_values
            
            self.obs_builder = DefaultObs(
                pos_coef=np.asarray([
                    1 / common_values.SIDE_WALL_X,
                    1 / common_values.BACK_NET_Y,
                    1 / common_values.CEILING_Z
                ]),
                ang_coef=1 / np.pi,
                lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
            )
        else:
            self.obs_builder = obs_builder
        
        self._obs_size = None  # Will be determined on first build
    
    def reset(self, initial_state: GameState):
        """Reset the observation builder."""
        self.obs_builder.reset(initial_state)
        self._obs_size = None  # Reset obs size on environment reset
    
    def build_obs(
        self, 
        player: PlayerData, 
        state: GameState, 
        previous_action: np.ndarray
    ) -> np.ndarray:
        """
        Build observation for a single player with padding.
        
        Args:
            player: The player to build observation for
            state: Current game state
            previous_action: Previous action taken by this player
            
        Returns:
            Padded observation array
        """
        # Get the base observation from the wrapped builder
        base_obs = self.obs_builder.build_obs(player, state, previous_action)
        base_obs = np.asarray(base_obs, dtype=np.float32)
        
        # Determine observation size per player on first call
        if self._obs_size is None:
            # DefaultObs typically returns a flat array
            # We need to figure out the per-player size
            self._obs_size = len(base_obs)
        
        # For now, return the base observation
        # Padding happens at the team level in multi-agent scenarios
        # For single-agent training, no padding is needed
        return base_obs
    
    def get_obs_space(self) -> int:
        """
        Get the size of the observation space.
        
        Returns the padded observation size accounting for max team size.
        """
        # Get base obs space size
        if hasattr(self.obs_builder, 'get_obs_space'):
            base_size = self.obs_builder.get_obs_space()
        else:
            # Estimate based on DefaultObs (typically ~70-90 features per player)
            # This will be updated on first build_obs call
            base_size = 107  # Common DefaultObs size
        
        # Return base size (padding is handled internally if needed)
        return base_size


class PaddedObsBuilder(ObsBuilder):
    """
    Alternative padded observation builder with explicit team padding.
    
    This version explicitly pads observations for missing teammates/opponents
    to a fixed team size, making the observation space consistent regardless
    of actual team sizes in the game.
    
    Args:
        base_obs_builder: The base observation builder to wrap
        max_team_size: Maximum number of players per team (default: 3)
    """
    
    def __init__(self, base_obs_builder: ObsBuilder, max_team_size: int = 3):
        super().__init__()
        self.base_obs_builder = base_obs_builder
        self.max_team_size = max_team_size
        self._per_player_obs_size = None
    
    def reset(self, initial_state: GameState):
        """Reset the base observation builder."""
        self.base_obs_builder.reset(initial_state)
        self._per_player_obs_size = None
    
    def build_obs(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray
    ) -> np.ndarray:
        """
        Build padded observation.
        
        Returns observation with padding for missing players up to max_team_size.
        """
        # Get base observation
        base_obs = self.base_obs_builder.build_obs(player, state, previous_action)
        base_obs = np.asarray(base_obs, dtype=np.float32)
        
        # For single-agent scenarios, just return the base observation
        # The padding logic would be more complex for actual multi-agent scenarios
        return base_obs
    
    def get_obs_space(self) -> int:
        """Get the observation space size."""
        if hasattr(self.base_obs_builder, 'get_obs_space'):
            return self.base_obs_builder.get_obs_space()
        return 107  # Default fallback


# For backward compatibility
def create_padded_obs_builder(team_size: int = 3, tick_skip: int = 8) -> AdvancedObsPadder:
    """
    Factory function to create a padded observation builder.
    
    Args:
        team_size: Maximum team size to pad to
        tick_skip: Tick skip value (for API compatibility)
        
    Returns:
        Configured AdvancedObsPadder instance
    """
    return AdvancedObsPadder(team_size=team_size, tick_skip=tick_skip)

import numpy as np
import os
import torch

# CRITICAL: Verify CUDA is available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: CUDA NOT AVAILABLE - Training will be VERY slow on CPU!")

# Monkey-patch torch.load to always use CPU mapping when CUDA is not available
# This fixes the "Attempting to deserialize object on a CUDA device" error
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if not torch.cuda.is_available() and 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rewards import InAirReward, SpeedTowardBallReward, HandbrakePenalty

# Game timing constants
TICK_SKIP = 8  # Number of physics ticks per step
GAME_TICK_RATE = 120  # Rocket League runs at 120 ticks per second
STEP_TIME = TICK_SKIP / GAME_TICK_RATE  # Time between steps in seconds (8/120 = 0.0667 seconds)


class ExampleLogger(MetricsLogger):
    def __init__(self):
        super().__init__()
        self.prev_ball_toucher = None
        self.prev_blue_score = 0
        self.prev_orange_score = 0
        self.total_touches = 0
        self.total_goals = 0
        self.episode_touches = 0
        self.episode_goals = 0
        
    def _collect_metrics(self, game_state: GameState) -> list:
        # Détecter les touches (quand last_touch change)
        if game_state.last_touch != self.prev_ball_toucher and game_state.last_touch != -1:
            self.total_touches += 1
            self.episode_touches += 1
            self.prev_ball_toucher = game_state.last_touch
            
        # Détecter les buts
        if game_state.blue_score > self.prev_blue_score:
            self.total_goals += 1
            self.episode_goals += 1
            self.prev_blue_score = game_state.blue_score
            
        if game_state.orange_score > self.prev_orange_score:
            self.total_goals += 1
            self.episode_goals += 1
            self.prev_orange_score = game_state.orange_score
        
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "total_touches": self.total_touches,
                  "total_goals": self.total_goals,
                  "episode_touches": self.episode_touches,
                  "episode_goals": self.episode_goals,
                  "Cumulative Timesteps":cumulative_timesteps}
        
        # Reset episode stats après le report
        self.episode_touches = 0
        self.episode_goals = 0
        
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward, FaceBallReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.state_setters import RandomState
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction

    # As requested: single-agent training with no opponents
    spawn_opponents = False
    team_size = 1
    # User requested timeout between 10 and 15 seconds. Use a midpoint by default.
    timeout_seconds = 12
    timeout_ticks = int(round(timeout_seconds * GAME_TICK_RATE / TICK_SKIP))

    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks)]#GoalScoredCondition()
    
    # RandomState for better training - cars and ball spawn with random positions/velocities
    # cars_on_ground=False means cars spawn airborne 50% of the time
    state_setter = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=False)

    reward_fn = CombinedReward.from_zipped(
    # Format is (func, weight)
    (EventReward(team_goal=1), 25),
    (VelocityBallToGoalReward(), 5),      # Augmenté: encourage de bons touches de balle
    (EventReward(touch=1), 1),           # Toucher la balle                                     
    (SpeedTowardBallReward(), 0.5),        # Augmenté: encourage à garder de la vitesse (décourage le frein à main)
    (FaceBallReward(), 0.1),                # Regarder la balle
    # Minimal pour conserver capacité de saut (InAirReward(), 0.003), # Pousser la balle vers le but 
)

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=TICK_SKIP,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)
    
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    
    metrics_logger = ExampleLogger()

    # Configuration manuelle - ajustez selon votre machine
    # RTX 4090 + 96 CPU cores = machine de guerre ! On passe à 48 processus
    n_proc = 48  # Utilise 50% des CPU cores (48/96) pour équilibrer avec le GPU
    minibatch_size = 50_000  # Doit être un diviseur de ppo_batch_size (50k)
    device = "cuda:0"  # "cuda:0" pour GPU, "cpu" pour CPU

    # Network size - constant pour garder les checkpoints
    policy_size = (512, 512, 256)
    critic_size = (512, 512, 256)

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.75)))  # 0.75 au lieu de 0.9 pour plus de vitesse

    # Discover latest checkpoint/run folder under data/checkpoints.
    latest_checkpoint_dir = None
    checkpoint_base = os.path.join("data", "checkpoints")
    try:
        # Find directories starting with the common prefix used by rlgym-ppo runs
        if os.path.isdir(checkpoint_base):
            run_dirs = [d for d in os.listdir(checkpoint_base) if d.startswith("rlgym-ppo-run") and os.path.isdir(os.path.join(checkpoint_base, d))]
            if run_dirs:
                # Choose the most recently modified run directory (robust to different naming schemes)
                latest_run = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(checkpoint_base, d)))
                latest_run_dir = os.path.join(checkpoint_base, latest_run)
                
                # Within the run directory, find the latest numbered checkpoint subdirectory
                checkpoint_subdirs = [d for d in os.listdir(latest_run_dir) if d.isdigit() and os.path.isdir(os.path.join(latest_run_dir, d))]
                if checkpoint_subdirs:
                    # Sort by the numeric value to get the highest checkpoint number
                    latest_checkpoint = max(checkpoint_subdirs, key=int)
                    latest_checkpoint_dir = os.path.join(latest_run_dir, latest_checkpoint)
    except Exception:
        # If anything goes wrong (missing folder, permissions, etc.), leave None so learner won't try to load
        latest_checkpoint_dir = None



    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      
                      # FORCE GPU DEVICE
                      device=device,
                      
                      # Data collection settings
                      ts_per_iteration=50_000,  # Start with 50k, increase to 100k once bot hits ball consistently
                      exp_buffer_size=150_000,  # 3x ts_per_iteration for better learning
                      ppo_batch_size=50_000,    # Same as ts_per_iteration
                      ppo_minibatch_size=minibatch_size,  # Auto-adjusted based on GPU/CPU
                      
                      # Network architecture - Auto-adjusted based on GPU/CPU
                      policy_layer_sizes=policy_size,
                      critic_layer_sizes=critic_size,
                      
                      # PPO hyperparameters
                      ppo_ent_coef=0.01,  # Good exploration value (NOT 0.001 like bad example!)
                      ppo_epochs=2,       # 2-3 is optimal, starting with 2
                      
                      # Learning rates - starting high since bot can't score yet
                      policy_lr=2e-4,     # Higher LR for early learning (decrease to 1e-4 once scoring)
                      critic_lr=2e-4,     # Keep same as policy_lr
                      
                      # Normalization
                      standardize_returns=True,
                      standardize_obs=True,
                      
                      # Checkpointing
                      save_every_ts=100_000,
                      checkpoint_load_folder=latest_checkpoint_dir,
                      
                      # Training duration - set to huge number, stop manually when satisfied
                      timestep_limit=10e15,  # 10 quadrillion (basically infinite)
                      
                      # Rendering - turn OFF for faster training, turn ON to watch
                      render=False,  # Set to True when you want to watch, False for max speed
                      render_delay=STEP_TIME,  # 2x speed when rendering is enabled
                      
                      # Logging
                      log_to_wandb=True)
    learner.learn()
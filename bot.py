"""
Modern RLGym-PPO training setup for Rocket League bot.
Uses latest best practices: padded observations, proper reward shaping, wandb logging.
"""
import numpy as np
import os
from pathlib import Path
import torch

# CUDA verification
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
else:
    print("⚠️  CPU mode - training will be slow!")

from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rewards import SpeedTowardBallReward, FlipDisciplineReward, InAirReward

# Game timing constants
TICK_SKIP = 8  # Physics ticks per step (8 is standard for most training)
GAME_TICK_RATE = 120  # Rocket League physics rate
STEP_TIME = TICK_SKIP / GAME_TICK_RATE  # 0.0667 seconds per step


class ModernMetricsLogger(MetricsLogger):
    """Enhanced metrics tracking for training progress."""
    
    def __init__(self):
        super().__init__()
        self.prev_ball_toucher = None
        self.prev_blue_score = 0
        self.prev_orange_score = 0
        self.cumulative_touches = 0
        self.cumulative_goals = 0
        self.interval_touches = 0
        self.interval_goals = 0
        
    def _collect_metrics(self, game_state: GameState) -> list:
        """Collect per-step metrics from game state."""
        # Track ball touches
        if game_state.last_touch != self.prev_ball_toucher and game_state.last_touch != -1:
            self.cumulative_touches += 1
            self.interval_touches += 1
            self.prev_ball_toucher = game_state.last_touch
        
        # Track goals scored
        if game_state.blue_score > self.prev_blue_score:
            self.cumulative_goals += 1
            self.interval_goals += 1
            self.prev_blue_score = game_state.blue_score
        
        if game_state.orange_score > self.prev_orange_score:
            self.cumulative_goals += 1
            self.interval_goals += 1
            self.prev_orange_score = game_state.orange_score
        
        # Collect player data (safely handle multiple agents)
        if len(game_state.players) > 0:
            player = game_state.players[0]
            return [
                player.car_data.linear_velocity,
                player.car_data.position,
                game_state.ball.position,
                np.linalg.norm(player.car_data.linear_velocity)  # Speed scalar
            ]
        return [np.zeros(3), np.zeros(3), np.zeros(3), 0.0]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        """Aggregate and report metrics to wandb."""
        if not collected_metrics:
            return
        
        # Calculate averages
        velocities = np.array([m[0] for m in collected_metrics])
        speeds = np.array([m[3] for m in collected_metrics])
        
        avg_velocity = velocities.mean(axis=0)
        avg_speed = speeds.mean()
        max_speed = speeds.max()
        
        report = {
            "performance/avg_speed": avg_speed,
            "performance/max_speed": max_speed,
            "performance/avg_velocity_x": avg_velocity[0],
            "performance/avg_velocity_y": avg_velocity[1],
            "performance/avg_velocity_z": avg_velocity[2],
            "gameplay/cumulative_touches": self.cumulative_touches,
            "gameplay/cumulative_goals": self.cumulative_goals,
            "gameplay/interval_touches": self.interval_touches,
            "gameplay/interval_goals": self.interval_goals,
            "training/timesteps": cumulative_timesteps,
        }
        
        # Reset interval stats
        self.interval_touches = 0
        self.interval_goals = 0
        
        wandb_run.log(report)


def build_rocketsim_env():
    """Build a modern RocketSim environment with best practices."""
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import (
        VelocityBallToGoalReward,
        EventReward,
        FaceBallReward
    )
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        NoTouchTimeoutCondition,
        GoalScoredCondition
    )
    from rlgym_sim.utils.state_setters import RandomState
    from rlgym_sim.utils.action_parsers import ContinuousAction
    from rlgym_ppo.util import AdvancedObsPadder

    # Training configuration
    spawn_opponents = False  # 1v0 training
    team_size = 1
    timeout_seconds = 12
    timeout_ticks = int(timeout_seconds * GAME_TICK_RATE / TICK_SKIP)

    # Modern action parser (continuous control)
    action_parser = ContinuousAction()
    
    # Terminal conditions
    terminal_conditions = [
        NoTouchTimeoutCondition(timeout_ticks),
        GoalScoredCondition()
    ]
    
    # State setter with randomization for robust learning
    state_setter = RandomState(
        ball_rand_speed=True,
        cars_rand_speed=True,
        cars_on_ground=False  # 50% aerial spawns
    )

    # Modern reward function (well-balanced weights)
    reward_fn = CombinedReward.from_zipped(
        (EventReward(touch=1.0), 50),            # Successful touches
        (SpeedTowardBallReward(), 5),           # Approach ball
        (FaceBallReward(), 1),    
        (InAirReward(), 0.05)              # Air reward
    )

    # MODERN: Padded observations for scalability
    # Supports variable team sizes (1v0 → 3v3) without retraining
    obs_builder = AdvancedObsPadder(
        team_size=3,           # Pad for up to 3v3
        tick_skip=TICK_SKIP    # Temporal info
    )

    env = rlgym_sim.make(
        tick_skip=TICK_SKIP,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter
    )
    
    # Optional: RocketSimVis rendering support
    try:
        import rocketsimvis_rlgym_sim_client as rsv
        type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    except ImportError:
        pass  # RocketSimVis not available

    return env

def find_latest_checkpoint() -> str | None:
    """Find the most recent checkpoint directory."""
    checkpoint_base = Path("data/checkpoints")
    
    if not checkpoint_base.exists():
        return None
    
    try:
        # Find all run directories
        run_dirs = [d for d in checkpoint_base.iterdir() 
                   if d.is_dir() and d.name.startswith("rlgym-ppo-run")]
        
        if not run_dirs:
            return None
        
        # Get most recent run
        latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
        
        # Find highest checkpoint number
        checkpoint_subdirs = [d for d in latest_run.iterdir() 
                             if d.is_dir() and d.name.isdigit()]
        
        if not checkpoint_subdirs:
            return None
        
        latest_checkpoint = max(checkpoint_subdirs, key=lambda d: int(d.name))
        return str(latest_checkpoint)
    
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


if __name__ == "__main__":
    from rlgym_ppo import Learner
    
    # Initialize metrics logger
    metrics_logger = ModernMetricsLogger()
    
    # ===== HARDWARE CONFIGURATION =====
    # Adjust based on your system (RTX 4090 + 96 cores)
    N_PROC = 48  # 50% of CPU cores for balanced CPU/GPU usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ===== NETWORK ARCHITECTURE =====
    # Large networks for complex behaviors
    POLICY_LAYERS = (512, 512, 256)
    CRITIC_LAYERS = (512, 512, 256)
    
    # ===== PPO HYPERPARAMETERS =====
    TIMESTEPS_PER_ITERATION = 100_000  # Increased from 50k for better sample efficiency
    BATCH_SIZE = 100_000
    MINIBATCH_SIZE = 50_000  # Must divide BATCH_SIZE evenly
    BUFFER_SIZE = 200_000  # 2x batch size
    
    LEARNING_RATE = 2e-4  # Standard for early training
    ENTROPY_COEF = 0.01   # Good exploration
    PPO_EPOCHS = 2        # 2-3 is optimal
    
    # ===== CHECKPOINTING =====
    SAVE_EVERY_TS = 500_000  # Save every ~5 iterations
    checkpoint_folder = find_latest_checkpoint()
    
    if checkpoint_folder:
        print(f"📂 Resuming from: {checkpoint_folder}")
    else:
        print("🆕 Starting fresh training")
    
    # ===== INFERENCE SETTINGS =====
    min_inference_size = max(1, int(N_PROC * 0.75))
    
    # ===== BUILD LEARNER =====
    learner = Learner(
        build_rocketsim_env,
        
        # Process and device config
        n_proc=N_PROC,
        min_inference_size=min_inference_size,
        device=DEVICE,
        
        # Network architecture
        policy_layer_sizes=POLICY_LAYERS,
        critic_layer_sizes=CRITIC_LAYERS,
        
        # PPO parameters
        ppo_batch_size=BATCH_SIZE,
        ppo_minibatch_size=MINIBATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        ppo_ent_coef=ENTROPY_COEF,
        
        # Data collection
        ts_per_iteration=TIMESTEPS_PER_ITERATION,
        exp_buffer_size=BUFFER_SIZE,
        
        # Learning rates
        policy_lr=LEARNING_RATE,
        critic_lr=LEARNING_RATE,
        
        # Normalization
        standardize_returns=True,
        standardize_obs=True,
        
        # Checkpointing
        save_every_ts=SAVE_EVERY_TS,
        checkpoint_load_folder=checkpoint_folder,
        
        # Training duration
        timestep_limit=1e15,  # Infinite - stop manually
        
        # Rendering
        render=False,
        render_delay=STEP_TIME,
        
        # Logging
        metrics_logger=metrics_logger,
        log_to_wandb=True,
    )
    
    print("\n" + "="*60)
    print("🚀 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Processes: {N_PROC}")
    print(f"Timesteps/Iteration: {TIMESTEPS_PER_ITERATION:,}")
    print(f"Batch Size: {BATCH_SIZE:,}")
    print(f"Network: {POLICY_LAYERS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Save Every: {SAVE_EVERY_TS:,} timesteps")
    print("="*60 + "\n")
    
    # Start training
    learner.learn()
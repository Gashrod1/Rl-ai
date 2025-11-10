import numpy as np
import os
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rewards import InAirReward, SpeedTowardBallReward

# Game timing constants
TICK_SKIP = 8  # Number of physics ticks per step
GAME_TICK_RATE = 120  # Rocket League runs at 120 ticks per second
STEP_TIME = TICK_SKIP / GAME_TICK_RATE  # Time between steps in seconds (8/120 = 0.0667 seconds)


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
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
                  "Cumulative Timesteps":cumulative_timesteps}
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
    (EventReward(touch=1), 50),
    (SpeedTowardBallReward(), 5),
    (FaceBallReward(), 1),
    (InAirReward(), 0.05)
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

    # 8 processes
    n_proc = 4

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

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
                      # training loop sizes (keep as before unless you want to tune)
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      # PPO hyperparameters requested
                      ppo_ent_coef=0.01,
                      ppo_epochs=2,
                      policy_lr=1e-4,
                      critic_lr=1e-4,
                      standardize_returns=True,
                      standardize_obs=True,
                      save_every_ts=100_000,
                      # Use ~200M timesteps as an approximate training budget
                      timestep_limit=200_000_000,
                      # If a previous run exists, load checkpoints from it
                      checkpoint_load_folder=latest_checkpoint_dir,
                      render=True,
                      render_delay=STEP_TIME,  # Normal speed: one state per step at real-time speed
                      log_to_wandb=True)
    learner.learn()
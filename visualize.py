"""
Simple visualization script to watch the bot play on CPU.
No training - just loads a checkpoint and visualizes the bot's behavior.
"""
import numpy as np
import torch
from pathlib import Path
import time

# Force CPU mode
print("🖥️  Running in CPU-only visualization mode")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()} (not used)")

# Import bot configuration
from bot import build_rocketsim_env, find_latest_checkpoint

def load_policy_for_cpu(checkpoint_path: str):
    """Load a policy checkpoint and map it to CPU."""
    from rlgym_ppo.ppo import ContinuousPolicy
    
    # Check for different possible policy file names
    checkpoint_dir = Path(checkpoint_path)
    policy_file = checkpoint_dir / "PPO_POLICY.pt"
    
    if not policy_file.exists():
        # Try alternate name
        policy_file = checkpoint_dir / "policy.pt"
    
    if not policy_file.exists():
        raise FileNotFoundError(f"No policy file found in {checkpoint_path}\nLooked for: PPO_POLICY.pt or policy.pt")
    
    # Load checkpoint with CPU mapping
    print(f"Loading policy from {policy_file}...")
    checkpoint = torch.load(policy_file, map_location=torch.device('cpu'), weights_only=False)
    
    # Get observation and action sizes from checkpoint
    # Look for weight tensors (they have .weight in the key)
    weight_keys = [k for k in checkpoint.keys() if 'weight' in k.lower() and len(checkpoint[k].shape) == 2]
    
    if not weight_keys:
        raise ValueError(f"Could not find weight layers in checkpoint")
    
    # Sort by layer number (model.0.weight, model.2.weight, etc.)
    weight_keys = sorted(weight_keys)
    
    # First layer determines input size (obs_size)
    first_weight = checkpoint[weight_keys[0]]
    obs_size = first_weight.shape[1]  # Input dimension
    
    # Last layer determines output size
    last_weight = checkpoint[weight_keys[-1]]
    output_size = last_weight.shape[0]  # Output dimension
    
    print(f"Detected obs_size={obs_size}, output_size={output_size}")
    
    # Create policy network
    policy = ContinuousPolicy(
        obs_size,
        output_size,
        layer_sizes=(512, 512, 256),  # Match bot.py architecture
        device='cpu'
    )
    
    # Load the weights
    policy.load_state_dict(checkpoint)
    policy.eval()  # Set to evaluation mode
    
    print("✅ Policy loaded successfully")
    return policy


def visualize_bot(checkpoint_path: str = None, episodes: int = None, render_delay: float = None):
    """
    Run the bot in visualization mode.
    
    Args:
        checkpoint_path: Path to checkpoint folder (or None for random policy)
        episodes: Number of episodes to run (None = infinite)
        render_delay: Delay between frames in seconds (None = 1x speed, 0 = no delay)
    """
    # Calculate 1x speed delay based on game settings
    # TICK_SKIP = 8, GAME_TICK_RATE = 120
    # Real-time step delay = 8 / 120 = 0.0667 seconds per step
    if render_delay is None:
        render_delay = 8 / 120  # 1x speed (real-time)
        print(f"ℹ️  Using 1x speed (real-time): {render_delay:.4f}s per step")
    # Build environment
    print("\n🏗️  Building environment...")
    env = build_rocketsim_env()
    
    # Load policy if checkpoint provided
    policy = None
    if checkpoint_path:
        try:
            policy = load_policy_for_cpu(checkpoint_path)
        except Exception as e:
            print(f"⚠️  Failed to load policy: {e}")
            print("⚠️  Using random actions instead")
    else:
        print("ℹ️  No checkpoint provided - using random actions")
    
    # Check if RocketSimVis is available
    try:
        import rocketsimvis_rlgym_sim_client as rsv
        has_visualizer = True
        print("✅ RocketSimVis detected - visualization enabled")
    except ImportError:
        has_visualizer = False
        print("⚠️  RocketSimVis not found - running without visualization")
        print("   Install with: pip install rocketsimvis_rlgym_sim_client")
    
    if episodes is None:
        print(f"\n🎮 Running infinitely...")
        print("   Press Ctrl+C to stop\n")
    else:
        print(f"\n🎮 Running {episodes} episode(s)...")
        print("   Press Ctrl+C to stop\n")
    
    episode = 0
    while episodes is None or episode < episodes:
        obs = env.reset()
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        done = False
        step_count = 0
        episode_reward = 0
        
        episode += 1
        if episodes is None:
            print(f"Episode {episode}")
        else:
            print(f"Episode {episode}/{episodes}")
        
        while not done:
            # Get action from policy or random
            if policy is not None:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
                    action_result = policy.get_action(obs_tensor, deterministic=True)
                    
                    # Handle different return types (could be tensor or tuple)
                    if isinstance(action_result, tuple):
                        action_tensor = action_result[0]  # Usually (action, log_prob, ...)
                    else:
                        action_tensor = action_result
                    
                    action = action_tensor.cpu().numpy()[0]
            else:
                # Random action
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, info = env.step(action)
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            episode_reward += reward
            step_count += 1
            
            # Render if available
            if has_visualizer and hasattr(env, 'render'):
                try:
                    env.render()
                except Exception:
                    pass  # Ignore render errors
            
            # Delay for visualization
            if render_delay > 0:
                time.sleep(render_delay)
            
            # Safety timeout
            if step_count > 10000:
                print("  ⏱️  Timeout reached")
                break
        
        print(f"  Steps: {step_count}, Reward: {episode_reward:.2f}")
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    checkpoint_path = None
    episodes = None  # None = infinite by default
    render_delay = None  # None = 1x speed (real-time)
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            print("Usage: python visualize.py [checkpoint_path] [episodes] [speed]")
            print("\nExamples:")
            print("  python visualize.py                          # Random policy, infinite, 1x speed")
            print("  python visualize.py latest                   # Load latest checkpoint, 1x speed")
            print("  python visualize.py latest 10                # 10 episodes at 1x speed")
            print("  python visualize.py latest 3 2               # 3 episodes at 2x speed")
            print("  python visualize.py latest inf 0.5           # Infinite at 0.5x speed (slow-mo)")
            print("  python visualize.py latest inf 0             # Infinite at max speed (no delay)")
            print("\nSpeed: 1 = real-time (default), 2 = 2x faster, 0.5 = half speed, 0 = no delay")
            print("Use 'inf' or '0' for infinite episodes (default)")
            sys.exit(0)
        
        if sys.argv[1] == "latest":
            checkpoint_path = find_latest_checkpoint()
            if checkpoint_path:
                print(f"Using latest checkpoint: {checkpoint_path}")
            else:
                print("No checkpoints found - using random policy")
        else:
            checkpoint_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        ep_arg = sys.argv[2].lower()
        if ep_arg in ['inf', 'infinite', '0']:
            episodes = None  # Infinite
        else:
            episodes = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        speed = float(sys.argv[3])
        if speed == 0:
            render_delay = 0  # No delay, max speed
        else:
            # Calculate delay based on speed multiplier
            # Base delay is 8/120 = 0.0667s for 1x speed
            base_delay = 8 / 120
            render_delay = base_delay / speed
    
    # Run visualization
    try:
        visualize_bot(checkpoint_path, episodes, render_delay)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

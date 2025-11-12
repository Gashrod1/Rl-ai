"""
Quick visualization launcher - just run this to watch your bot!
"""
import subprocess
import sys
from pathlib import Path

# Check if we're in the right directory
if not Path("bot.py").exists():
    print("❌ Error: Please run this script from the Rl-ai directory")
    print(f"Current directory: {Path.cwd()}")
    sys.exit(1)

print("=" * 60)
print("🎮 RL Bot Visualizer")
print("=" * 60)
print()

# Find checkpoints
from bot import find_latest_checkpoint

checkpoint = find_latest_checkpoint()

if checkpoint:
    print(f"✅ Found checkpoint: {checkpoint}")
    print()
    choice = input("Load this checkpoint? (Y/n): ").strip().lower()
    
    if choice in ['', 'y', 'yes']:
        use_checkpoint = True
    else:
        use_checkpoint = False
        print("Using random policy instead")
else:
    print("ℹ️  No checkpoints found")
    use_checkpoint = False

print()
episodes = input("How many episodes to run? (default: infinite, or enter number): ").strip()
if not episodes:
    episodes = "inf"
elif episodes.lower() in ['inf', 'infinite']:
    episodes = "inf"

print()
print("Starting visualization...")
print("=" * 60)
print()

# Build command
cmd = [sys.executable, "visualize.py"]
if use_checkpoint:
    cmd.append("latest")
cmd.append(episodes)

# Run visualization
try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("\n\nStopped by user")

# Quick Commands for Visualization

# 1x speed (real-time) - default
python visualize.py latest

# 2x speed (faster)
python visualize.py latest inf 2

# 0.5x speed (slow motion)
python visualize.py latest inf 0.5

# Max speed (no delay)
python visualize.py latest inf 0

## Check Setup
```powershell
python check_viz.py
```

## Watch Bot Play
```powershell
# Interactive (easiest)
python watch.py

# Direct with latest checkpoint
python visualize.py latest

# Random policy (no checkpoint needed)
python visualize.py

# Custom: 10 episodes at 30 FPS
python visualize.py latest 10 30
```

## What Each Script Does

| Script | Purpose |
|--------|---------|
| `check_viz.py` | Check if you have everything installed |
| `watch.py` | Interactive launcher - asks you what to do |
| `visualize.py` | Main visualization script - runs bot on CPU |
| `VISUALIZATION.md` | Full guide with troubleshooting |

## First Time Setup

1. **Check requirements:**
   ```powershell
   python check_viz.py
   ```

2. **Install missing packages if needed:**
   ```powershell
   pip install rocketsimvis_rlgym_sim_client
   ```

3. **Run visualization:**
   ```powershell
   python watch.py
   ```

That's it! 🎮

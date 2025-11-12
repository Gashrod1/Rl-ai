# Quick Configuration Guide

## Current Setup: 1v1 MODE (89-dim observations)

Your bot is now configured to train against an opponent!

## How to Switch Modes

### To Train with Opponent (1v1) - CURRENT
In `bot.py`, line ~103:
```python
spawn_opponents = True  # 89-dim observations
```

### To Train without Opponent (1v0)
In `bot.py`, line ~103:
```python
spawn_opponents = False  # 70-dim observations
```

⚠️ **Important**: When you change this setting, you must also ensure your checkpoint matches!

---

## What Happens When You Run Now

1. Script finds your latest checkpoint (trained with 70-dim)
2. **Automatic transfer learning** kicks in
3. First layer expands from 70→89 inputs
4. Training continues with opponents spawned
5. New checkpoints saved with 89-dim

---

## Observation Space Breakdown

### 70-dimensional (1v0 - No Opponents)
- Player position (3)
- Player linear velocity (3)
- Player angular velocity (3)
- Player rotation matrix (9)
- Player boost amount (1)
- Ball position (3)
- Ball linear velocity (3)
- Ball angular velocity (3)
- Inverted data (mirror for team symmetry)
- Boost pad states (34)
- **Total: 70 features**

### 89-dimensional (1v1 - With Opponents)
- Everything from 70-dim above
- **+ Opponent position (3)**
- **+ Opponent linear velocity (3)**
- **+ Opponent angular velocity (3)**
- **+ Opponent rotation matrix (9)**
- **+ Opponent boost amount (1)**
- **Total: 89 features (70 + 19)**

---

## Training Tips for 1v1

1. **Be Patient**: Your bot needs ~1-5M timesteps to adapt to opponent presence
2. **Monitor Metrics**: Watch `total_touches` and `total_goals` in WandB
3. **Expect Initial Drop**: Performance may dip temporarily while learning
4. **Reward Tuning**: Consider adjusting concede penalty if bot becomes too defensive

### Suggested Reward Adjustments for 1v1
```python
# More emphasis on preventing opponent goals
(EventReward(team_goal=1, concede=-1.5), 30),  # Concede penalty increased

# Add optional rewards for competitive play
# (OpponentDistanceToBallReward(), 0.5),  # Reward being closer to ball than opponent
# (EventReward(demo=1), 1),  # Reward demolishing opponent
```

---

## Checkpoint Compatibility

| Checkpoint Type | spawn_opponents=False (70) | spawn_opponents=True (89) |
|----------------|---------------------------|--------------------------|
| 70-dim checkpoint | ✅ Works directly | ✅ Auto-transferred |
| 89-dim checkpoint | ❌ Will crash | ✅ Works directly |

The transfer function handles this automatically!

---

## Need Help?

- Check `TRANSFER_LEARNING_GUIDE.md` for detailed explanation
- Watch console output for transfer learning messages
- Monitor WandB for training progress
- Check checkpoint folders for "_transferred_to_89" suffix

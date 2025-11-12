# Transfer Learning: 1v0 (70-dim) to 1v1 (89-dim)

## What Changed

Your bot was trained in **1v0 mode** with observation size **70**, which includes:
- Your car's state (position, velocity, rotation, etc.)
- Ball state
- Boost pads
- But NO opponent information

To play **1v1** (against an opponent), you need observation size **89**, which adds:
- Opponent car's state (position, velocity, rotation, etc.) = 19 additional features

## What I've Done

### 1. **Updated `bot.py` Configuration**
Changed `spawn_opponents = False` to `spawn_opponents = True` (line 103)

### 2. **Added Transfer Learning Function**
Created `transfer_70_to_89_checkpoint()` that:
- Loads your existing 70-dim trained model
- Expands the first layer from 70 to 89 inputs
- **Preserves** all learned weights for the first 70 features
- **Initializes** new random weights for the 19 opponent features
- Saves the adapted checkpoint to a new folder with "_transferred_to_89" suffix

### 3. **Automatic Transfer on Load**
When you run `bot.py`, it will:
1. Find your latest 70-dim checkpoint
2. Automatically transfer it to 89-dim
3. Continue training with opponents

## How to Use

### Option A: Continue Training (Recommended)
Just run `bot.py` as usual:
```bash
python bot.py
```

The script will:
- Detect your 70-dim checkpoint
- Transfer it to 89-dim automatically
- Continue training with opponents spawned

Your bot will:
- ✅ Keep all learned ball-touching and scoring behaviors
- 🆕 Learn to handle opponent presence from scratch
- 📈 Gradually improve at competitive play

### Option B: Manual Transfer (Advanced)
If you want to transfer a specific checkpoint manually:

```python
from bot import transfer_70_to_89_checkpoint

# Transfer a specific checkpoint
old_checkpoint = "data/checkpoints/rlgym-ppo-run_xxx/12345"
new_checkpoint = transfer_70_to_89_checkpoint(old_checkpoint)
print(f"New checkpoint at: {new_checkpoint}")
```

## What to Expect

### Phase 1: Initial Confusion (First ~1M timesteps)
- Your bot may seem "worse" initially
- It's learning what the opponent data means
- Ball-touching skills are still there, just needs to adapt

### Phase 2: Adaptation (1M-5M timesteps)
- Bot learns to avoid colliding with opponent
- Starts to position itself better
- More consistent ball touches in competitive scenarios

### Phase 3: Competitive Play (5M+ timesteps)
- Bot understands opponent positioning
- Learns to block opponent shots
- Develops competitive strategies

## Important Notes

1. **Observation Space Change**: 
   - Old: 70 features (car + ball + boost)
   - New: 89 features (car + ball + boost + opponent)

2. **Network Structure**: 
   - The policy network's first layer changed from (70 → 512) to (89 → 512)
   - All other layers remain unchanged
   - Only the new 19 features start from scratch

3. **Training Speed**: 
   - May be slightly slower due to opponent AI
   - Each episode is more competitive and valuable

4. **Checkpoints**: 
   - Original 70-dim checkpoints are preserved
   - New checkpoints will be 89-dim
   - Can't switch back to 70-dim without re-training

## Troubleshooting

### Error: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
- Your checkpoint is still 70-dim
- Make sure `spawn_opponents = True` in `bot.py`
- Delete any cached checkpoints and let the transfer function run

### Bot Performance Drops
- **This is normal!** The bot needs to learn what opponent data means
- Keep training, performance will recover and exceed previous levels
- Monitor `total_touches` and `total_goals` in WandB

### Want to Go Back to 1v0?
- Set `spawn_opponents = False`
- Point to your original 70-dim checkpoint
- Or train a new agent from scratch

## Alternative: Training from Scratch

If transfer learning doesn't work well, you can always train from scratch with opponents:
1. Comment out the checkpoint loading in `bot.py`
2. Set `checkpoint_load_folder=None`
3. Run with `spawn_opponents = True`
4. Let it train fresh for better competitive behavior

---

Good luck with your competitive training! Your bot already knows how to touch the ball and score - now it just needs to learn to do it with an opponent in the way! 🚗⚽

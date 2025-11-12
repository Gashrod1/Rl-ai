"""
Check if you have everything needed for visualization.
"""
import sys

print("Checking visualization requirements...\n")

# Check Python version
print(f"✅ Python {sys.version.split()[0]}")

# Check core dependencies
required = {
    "torch": "PyTorch (neural networks)",
    "numpy": "NumPy (data handling)",
    "rlgym_sim": "RLGym-Sim (environment)",
    "rlgym_ppo": "RLGym-PPO (training/policy)"
}

optional = {
    "rocketsimvis_rlgym_sim_client": "RocketSimVis (visualization)"
}

print("\n📦 Required packages:")
missing_required = []
for package, description in required.items():
    try:
        __import__(package)
        print(f"  ✅ {package:30s} - {description}")
    except ImportError:
        print(f"  ❌ {package:30s} - {description} [MISSING]")
        missing_required.append(package)

print("\n🎨 Optional packages:")
missing_optional = []
for package, description in optional.items():
    try:
        __import__(package)
        print(f"  ✅ {package:30s} - {description}")
    except ImportError:
        print(f"  ⚠️  {package:30s} - {description} [NOT INSTALLED]")
        missing_optional.append(package)

# Check for checkpoints
print("\n💾 Checkpoints:")
try:
    from bot import find_latest_checkpoint
    checkpoint = find_latest_checkpoint()
    if checkpoint:
        print(f"  ✅ Found: {checkpoint}")
    else:
        print("  ℹ️  No checkpoints found (will use random policy)")
except Exception as e:
    print(f"  ⚠️  Could not check: {e}")

# Check CUDA
import torch
print(f"\n🖥️  Device:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Will use: CPU (for visualization)")

# Summary
print("\n" + "=" * 60)
if missing_required:
    print("❌ MISSING REQUIRED PACKAGES:")
    for pkg in missing_required:
        print(f"   - {pkg}")
    print("\nInstall with:")
    print(f"   pip install {' '.join(missing_required)}")
    print("\n⚠️  Cannot run visualization until these are installed!")
else:
    print("✅ All required packages installed!")
    
    if missing_optional:
        print("\n⚠️  Optional package missing:")
        for pkg in missing_optional:
            print(f"   - {pkg}")
        print("\nFor visual simulation, install with:")
        print(f"   pip install {' '.join(missing_optional)}")
        print("\n✅ Can run visualization (console-only mode)")
    else:
        print("✅ All optional packages installed!")
        print("\n🎉 Ready to visualize with full graphics!")
    
    print("\nRun visualization with:")
    print("   python watch.py       (interactive)")
    print("   python visualize.py   (direct)")

print("=" * 60)

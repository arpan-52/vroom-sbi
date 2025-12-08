#!/usr/bin/env python
"""Test if all imports work correctly."""

print("Testing imports...")

try:
    print("1. Testing SBI import...")
    from sbi.inference import SNPE
    print("   ✓ SNPE imported")

    print("2. Testing posterior_nn import...")
    try:
        from sbi.utils import posterior_nn
        print("   ✓ posterior_nn from sbi.utils")
    except ImportError:
        try:
            from sbi.neural_nets import posterior_nn
            print("   ✓ posterior_nn from sbi.neural_nets")
        except ImportError:
            print("   ✗ posterior_nn not found (will use default SNPE)")

    print("3. Testing train module import...")
    from src.train import train_all_models
    print("   ✓ train_all_models imported")

    print("\n✅ All critical imports successful!")
    print("You can run: python train_all.py")

except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

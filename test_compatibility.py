def test_compatibility():
    try:
        import cv2
        import numpy as np
        print(f"✅ SUCCESS!")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"OpenCV supports NumPy {np.__version__}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing NumPy 2.x and OpenCV compatibility...")
    print("-" * 50)
    success = test_compatibility()
    print("-" * 50)
    if success:
        print("🎯 Your environment is ready for Holo_Process!")
        print("Run: python app.py")
    else:
        print("🔧 Run the compatibility fix scripts to resolve issues.")
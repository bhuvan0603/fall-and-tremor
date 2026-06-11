"""
Test script to verify webcam is accessible
"""
import cv2
import sys

def test_webcam(camera_id=0):
    """Test if webcam is accessible"""
    print(f"Testing camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        print("\nTroubleshooting:")
        print("1. Check if camera is connected")
        print("2. Close other apps using camera (Zoom, Teams, etc.)")
        print("3. Check Windows Settings → Privacy → Camera")
        return False
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera {camera_id} is accessible!")
    print(f"   Resolution: {int(width)}x{int(height)}")
    print(f"   FPS: {fps:.1f}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"   Frame shape: {frame.shape}")
        print("✅ Successfully captured test frame")
    else:
        print("❌ Failed to capture frame")
        cap.release()
        return False
    
    cap.release()
    print("\n✅ Webcam test passed!")
    print("You can now run: python webcam_fall_detector.py")
    return True

if __name__ == '__main__':
    print("="*60)
    print("🎥 WEBCAM TEST")
    print("="*60)
    print()
    
    # Test default camera
    success = test_webcam(0)
    
    # Try camera 1 if camera 0 fails
    if not success:
        print("\nTrying external camera (ID=1)...")
        test_webcam(1)
    
    print("="*60)

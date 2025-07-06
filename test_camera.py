#!/usr/bin/env python3
"""
Simple camera test script to verify camera functionality.
"""

import cv2
import sys

def test_camera(camera_index=0):
    """Test camera functionality."""
    print(f"Testing camera {camera_index}...")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_index}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Camera {camera_index} opened successfully")
    print(f"📹 Resolution: {width}x{height}")
    print(f"🎯 FPS: {fps}")
    print("💡 Press 'q' to quit, 's' to save a test image")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Add frame counter to display
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(f"Camera {camera_index} Test", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("🛑 Quit requested")
            break
        elif key == ord('s'):
            filename = f"test_frame_{camera_index}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved test image: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Camera {camera_index} test completed")
    return True

if __name__ == "__main__":
    camera_index = 0
    
    # Allow command line argument for camera index
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("❌ Error: Invalid camera index")
            sys.exit(1)
    
    success = test_camera(camera_index)
    if not success:
        sys.exit(1) 
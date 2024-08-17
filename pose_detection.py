import os
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model for person, backpack, and handbag detection
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model: {e}")

# Initialize MediaPipe Pose
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except Exception as e:
    raise RuntimeError(f"Error initializing MediaPipe Pose: {e}")

# Open a video capture stream from the camera
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except Exception as e:
    raise RuntimeError(f"Error opening video stream: {e}")

# Bag midpoints
bags = {
    "red": (100, 380),
    "green": (320, 400),
    "blue": (500, 370)
}

def calculate_direction_ratio(x1, y1, x2, y2):
    try:
        if y2 - y1 != 0:
            return (x2 - x1) / (y2 - y1)
        return None
    except Exception as e:
        print(f"Error in calculating direction ratio: {e}")
        return None

def find_closest_bag(arm_direction, elbow_x, elbow_y):
    closest_bag = None
    min_diff = float('inf')

    try:
        for bag, (bx, by) in bags.items():
            bag_direction = calculate_direction_ratio(elbow_x, elbow_y, bx, by)
            if bag_direction is not None:
                diff = abs(arm_direction - bag_direction)
                if diff < min_diff:
                    min_diff = diff
                    closest_bag = bag
    except Exception as e:
        print(f"Error finding closest bag: {e}")

    return closest_bag, min_diff

def get_main_person(frame):
    results = model(frame, classes=0)  # class 0 is person
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]  # width * height
        main_person_index = areas.argmax()
        return boxes[main_person_index].xyxy[0].cpu().numpy().astype(int)
    return None

def detect_bags(frame):
    results = model(frame, classes=[24, 26])  # class 24 is backpack, 26 is handbag
    backpacks = []
    paper_bags = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            if cls == 24:  # backpack
                backpacks.append(b)
            elif cls == 26:  # handbag (paper bag)
                paper_bags.append(b)
    return backpacks, paper_bags

def draw_bounding_boxes(frame, boxes, color=(0, 255, 0)):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

frame_count = 0
last_print_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    current_time = time.time()
    if current_time - last_print_time >= 1:  # Print debug info every 1 second
        print(f"Processing frame {frame_count}")
        last_print_time = current_time

    # Detect the main person in the frame
    main_person_box = get_main_person(frame)
    
    if main_person_box is not None:
        x1, y1, x2, y2 = main_person_box
        person_frame = frame[y1:y2, x1:x2]
    else:
        person_frame = frame

    # Detect backpacks and paper bags
    backpacks, paper_bags = detect_bags(frame)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect poses
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:
        # Draw the pose annotations on the image
        mp_drawing.draw_landmarks(
            person_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Get landmark positions
        landmarks = pose_results.pose_landmarks.landmark
        h, w, c = person_frame.shape

        joints = {
            "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
            "right": {"shoulder": 12, "elbow": 14, "wrist": 16}
        }

        for side, indices in joints.items():
            # Get coordinates for shoulder, elbow and wrist
            shoulder = landmarks[indices["shoulder"]]
            elbow = landmarks[indices["elbow"]]
            wrist = landmarks[indices["wrist"]]

            # Convert to pixel coordinates
            shoulder_x, shoulder_y = int(shoulder.x * w), int(shoulder.y * h)
            elbow_x, elbow_y = int(elbow.x * w), int(elbow.y * h)
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

            # Draw the line from shoulder to wrist
            cv2.line(person_frame, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (0, 255, 0), 2)
            cv2.line(person_frame, (elbow_x, elbow_y), (wrist_x, wrist_y), (0, 255, 0), 2)
            
            # Extend the line
            extended_x = int(wrist_x + (wrist_x - elbow_x) * 2)
            extended_y = int(wrist_y + (wrist_y - elbow_y) * 2)
            cv2.line(person_frame, (wrist_x, wrist_y), (extended_x, extended_y), (0, 255, 0), 2)

            # Calculate arm direction ratio
            arm_direction = calculate_direction_ratio(shoulder_x, shoulder_y, wrist_x, wrist_y)

            if arm_direction is not None:
                # Find the closest bag for this arm
                bag, diff = find_closest_bag(arm_direction, elbow_x + x1, elbow_y + y1)

                # Check if this arm is closer to a bag than the other and if the wrist is below the elbow
                if bag and wrist_y > elbow_y and diff < 0.5:  # Added threshold for diff
                    print(f"{side.capitalize()} arm is pointing at the {bag} bag!")

        # Paste the person_frame back to the original frame
        if main_person_box is not None:
            frame[y1:y2, x1:x2] = person_frame

    # Draw the bags' midpoint
    for color, (x, y) in bags.items():
        cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255) if color == "red" else (0, 255, 0) if color == "green" else (255, 50, 50), thickness=-1)
    
    # Draw bounding boxes for backpacks only if backpacks or paper bags are detected
    if backpacks or paper_bags:
        draw_bounding_boxes(frame, backpacks, color=(255, 0, 0))  # Draw backpacks in blue
    
    # Display the frame
    cv2.imshow('Pose Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
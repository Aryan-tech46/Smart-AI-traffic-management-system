import cv2
from ultralytics import YOLO
from collections import defaultdict
import csv
import time

# Load YOLO model
model = YOLO("yolo11s.pt")  # Replace with your model if needed
class_list = model.names

# Input video paths
video1 = 'lane2.mp4'
video2 = 'lane1.mp4'
videos = [video1, video2]  # Corrected this line

# Road and vehicle parameters
road_length = 200  # meters
vehicle_length = 5  # meters
road_segments = road_length  # Number of vehicle slots

# CSV file setup
csv_filename = "vehicle_data.csv"
with open(csv_filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Serial Number", "Frame", "Vehicle Count", "Vehicle Type", "Density (vehicles per meter)"])

# Get target resolution from the first frame of video1
cap = cv2.VideoCapture(video1)
ret, frame = cap.read()
if not ret:
    raise Exception("Could not read the first frame from video1")
target_width, target_height = frame.shape[1], frame.shape[0]
cap.release()

# Function to process a video for a given duration and return updated variables
def process_video(video_path, max_duration, serial_number, frame_count, target_width, target_height):
    cap = cv2.VideoCapture(video_path)
    line_y_red = 430
    class_counts = defaultdict(int)
    crossed_ids = set()
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (time.time() - start_time) > max_duration:
            break

        frame = cv2.resize(frame, (target_width, target_height))

        frame_count += 1

        # Run YOLO tracking
        results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7], verbose=False)

        if results[0].boxes.data is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu()

            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = class_list[class_idx]

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if cy > line_y_red and track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    class_counts[class_name] += 1

            total_vehicles = sum(class_counts.values())
            density = ((total_vehicles) / 3 * vehicle_length) / road_segments

            with open(csv_filename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([serial_number, frame_count, total_vehicles, ", ".join(class_counts.keys()), density])

            serial_number += 1

            y_offset = 30
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30

            cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (50, y_offset + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

            cv2.putText(frame, f"Density: {density:.2f} vehicles/segment", (50, y_offset + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.line(frame, (0, line_y_red), (frame.shape[1], line_y_red), (0, 0, 255), 2)
        cv2.imshow("YOLO Object Tracking and Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if 'density' not in locals():
        density = 0

    # Timer logic based on density
    if 0<density < 0.04:
        density_timer = 40
    elif 0.04<=density < 0.07:
        density_timer = 60
    elif 0.07<=density < 0.11:
        density_timer = 80
    else:
        density_timer = 120

    print(f"Finished '{video_path}' | Density: {density:.2f} | Next Timer: {density_timer}s")
    return density_timer, serial_number, frame_count

# === Main loop ===
traffic_timer = 20
serial_number = 1
frame_count = 0
video_index = 0

try:
    while True:
        current_video = videos[video_index]
        density_timer, serial_number, frame_count = process_video(
            current_video, traffic_timer, serial_number, frame_count, target_width, target_height
        )
        traffic_timer = density_timer
        video_index = (video_index + 1) % len(videos)

except KeyboardInterrupt:
    print("\nProgram terminated by user.")

print(f"CSV file '{csv_filename}' saved successfully!")

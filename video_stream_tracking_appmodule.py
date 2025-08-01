import os
import cv2
import torch
import joblib
import gzip
import shutil
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort

from velocity_tracker import VelocityTracker
from zone_utils import define_zones, get_all_zones_for_bbox, draw_zones_on_image
from voice_feedback import VoiceAlertManager


# # Decompress before loading
# if not os.path.exists("best.pt") or os.path.getsize("best.pt") == 0:
#     with gzip.open("best.pt.gz", "rb") as f_in:
#         with open("best.pt", "wb") as f_out:
#             shutil.copyfileobj(f_in, f_out)

from ultralytics import YOLO
# try:
#     model = YOLO("best.pt", task = "detect")
# except Exception as e:
#     raise RuntimeError(f"Failed to load YOLO model from best.pt. "
#                        f"Check if the file is a valid YOLO weight file. Original error: {e}")

# model = YOLO("best.pt")
model = YOLO("runs/train/best.pt")
tracker = DeepSort(max_age=30)
velocity_tracker = VelocityTracker()
voice_alert = VoiceAlertManager()
with gzip.open("random_forest_model.pkl.gz", "rb") as f:
    classifier = joblib.load(f)
    
# Global variables
zones = None
frame_width, frame_height = None, None
frame_count = 0
fps = 30


def init_zones(frame):
    """Always re-initialize detection zones to current frame size."""
    global zones, frame_width, frame_height
    frame_height, frame_width = frame.shape[:2]
    zones = define_zones(frame_width, frame_height)


def process_frame(frame):
    """Process one frame, run detections, classification, and overlay safety info."""
    global frame_count, fps, zones

    frame_count += 1

    # Always refresh zones in case resolution changes
    init_zones(frame)

    # === YOLO Detection ===
    results = model(frame, verbose=False)[0]

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)
        if class_id in [0, 2, 3, 5, 7]:  # person + vehicle classes
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))

    # === Tracking ===
    tracks = tracker.update_tracks(detections, frame=frame)

    # === Feature Variables ===
    num_vehicles = num_pedestrians = 0
    num_in_crossing_zone = num_entering_crossing_zone = 0
    total_speed = vehicle_count_for_speed = 0
    dir_left = dir_right = dir_up = dir_down = 0
    vehicle_distances, vehicle_speeds = [], []

    cz = zones.get('crossing_zone')
    cz_x, cz_y = (
        ((cz[0][0] + cz[1][0]) / 2, (cz[0][1] + cz[1][1]) / 2)
        if cz else (frame_width // 2, frame_height // 2)
    )

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        class_id = getattr(track, "det_class", -1)

        velocity_tracker.update(track_id, (x1, y1, x2, y2))
        speed, direction = velocity_tracker.get_speed_direction(track_id)
        angle = direction.get("angle", 0) if isinstance(direction, dict) else 0

        zones_hit = get_all_zones_for_bbox((x1, y1, x2, y2), zones)

        if class_id in [2, 3, 5, 7]:  # Vehicle
            num_vehicles += 1
            total_speed += speed
            vehicle_count_for_speed += 1

            if "crossing_zone" in zones_hit:
                num_in_crossing_zone += 1
            elif speed > 1.0 and "crossing_zone" not in zones_hit:
                num_entering_crossing_zone += 1

            if 135 < angle <= 225:
                dir_left += 1
            elif angle <= 45 or angle > 315:
                dir_right += 1
            elif 45 < angle <= 135:
                dir_up += 1
            elif 225 < angle <= 315:
                dir_down += 1

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = ((cx - cz_x) ** 2 + (cy - cz_y) ** 2) ** 0.5
            if speed > 0:
                vehicle_distances.append(dist)
                vehicle_speeds.append(speed)

        elif class_id == 0:  # Pedestrian
            num_pedestrians += 1

    # === Compute Features ===
    avg_speed = total_speed / vehicle_count_for_speed if vehicle_count_for_speed else 0.0
    avg_time_to_collision = (
        sum(d / s for d, s in zip(vehicle_distances, vehicle_speeds)) / len(vehicle_distances)
        if vehicle_distances else 999.0
    )

    feature_dict = {
        "num_vehicles": num_vehicles,
        "num_pedestrians": num_pedestrians,
        "num_in_crossing_zone": num_in_crossing_zone,
        "num_entering_crossing_zone": num_entering_crossing_zone,
        "avg_vehicle_speed": avg_speed,
        "avg_time_to_collision": avg_time_to_collision,
        "dir_left": dir_left,
        "dir_right": dir_right,
        "dir_up": dir_up,
        "dir_down": dir_down,
    }

    try:
        feature_vector = pd.DataFrame([feature_dict])
        frame_label = classifier.predict(feature_vector)[0]
    except Exception as e:
        print(f"[Classifier Error] {e}")
        frame_label = 1  # default Unsafe

    is_safe = (frame_label == 0)

    timestamp_sec = frame_count / fps
    voice_alert.update_and_speak(is_safe, timestamp_sec)

    # === Draw Zones + Safety Label ===
    draw_zones_on_image(frame, zones)
    label_text = "Safe" if is_safe else "Unsafe"
    label_color = (0, 255, 0) if is_safe else (0, 0, 255)
    cv2.putText(frame, f'Overall: {label_text}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

    return frame

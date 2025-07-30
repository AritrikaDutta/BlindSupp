import math
from collections import defaultdict

class VelocityTracker:
    def __init__(self, max_history=5):
        self.track_history = defaultdict(list)
        self.max_history = max_history
        self.first_seen = {}
        self.last_bbox = {}

    def update(self, track_id, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.track_history[track_id].append((cx, cy))
        if len(self.track_history[track_id]) > self.max_history:
            self.track_history[track_id].pop(0)

        self.last_bbox[track_id] = bbox
        if track_id not in self.first_seen:
            self.first_seen[track_id] = len(self.track_history[track_id])

    def get_speed_direction(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 2:
            return 0.0, 'unknown'

        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]
        speed = math.sqrt(dx**2 + dy**2)

        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'

        return speed, direction

    def get_time_to_collision(self, track_id, target_zone):
        history = self.track_history[track_id]
        if len(history) < 2:
            return None

        cx, cy = history[-1]
        prev_cx, prev_cy = history[0]
        dx = cx - prev_cx
        dy = cy - prev_cy
        speed = math.sqrt(dx ** 2 + dy ** 2)
        if speed < 1e-5:
            return None

        zx = (target_zone["x1"] + target_zone["x2"]) / 2
        zy = (target_zone["y1"] + target_zone["y2"]) / 2
        dist = math.sqrt((cx - zx) ** 2 + (cy - zy) ** 2)
        return round(dist / speed, 2)

    def is_moving_toward_zone(self, track_id, target_zone):
        history = self.track_history[track_id]
        if len(history) < 2:
            return False

        cx, cy = history[-1]
        prev_cx, prev_cy = history[0]
        dx = cx - prev_cx
        dy = cy - prev_cy

        zx = (target_zone["x1"] + target_zone["x2"]) / 2
        zy = (target_zone["y1"] + target_zone["y2"]) / 2

        vx = zx - cx
        vy = zy - cy

        dot_product = dx * vx + dy * vy
        return dot_product > 0

    def get_features(self, track_id, zones=None):
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return None

        history = self.track_history[track_id]
        cx, cy = history[-1]
        prev_cx, prev_cy = history[0]
        dx = cx - prev_cx
        dy = cy - prev_cy
        speed = math.sqrt(dx**2 + dy**2)
        direction_angle = math.degrees(math.atan2(dy, dx)) if dx != 0 else 90.0
        is_stationary = 1 if speed < 1.0 else 0
        time_visible = len(history)

        bbox = self.last_bbox.get(track_id)
        if bbox is None or zones is None:
            return None

        x1, y1, x2, y2 = bbox
        crossing = zones["CROSSING"]
        zx1, zy1, zx2, zy2 = crossing["x1"], crossing["y1"], crossing["x2"], crossing["y2"]
        iou_crossing = self._iou(bbox, (zx1, zy1, zx2, zy2))

        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        zone_left = zones["LEFT"]
        zone_right = zones["RIGHT"]

        in_left = 1 if self._intersects(bbox, zone_left) else 0
        in_right = 1 if self._intersects(bbox, zone_right) else 0

        crossing_center_x = (crossing["x1"] + crossing["x2"]) / 2
        crossing_center_y = (crossing["y1"] + crossing["y2"]) / 2
        dist_to_crossing = math.sqrt((cx - crossing_center_x) ** 2 + (cy - crossing_center_y) ** 2)

        return {
            "speed": round(speed, 2),
            "direction_angle": round(direction_angle, 2),
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "is_stationary": is_stationary,
            "time_visible": time_visible,
            "distance_to_crossing": round(dist_to_crossing, 2),
            "iou_crossing": round(iou_crossing, 2),
            "in_left_zone": in_left,
            "in_right_zone": in_right
        }

    def _intersects(self, bbox, zone):
        x1, y1, x2, y2 = bbox
        zx1, zy1, zx2, zy2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
        return not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2)

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        if boxAArea + boxBArea - interArea == 0:
            return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

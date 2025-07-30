# zone_utils_1.py

import cv2

def define_zones(image_width, image_height):
    left_w = 0.20
    right_w = 0.20
    crossing_top = 0.4
    crossing_bottom = 1.0

    zones = {
        "CROSSING": {
            "x1": int(image_width * left_w),
            "y1": int(image_height * crossing_top),
            "x2": int(image_width * (1 - right_w)),
            "y2": int(image_height * crossing_bottom)
        },
        "LEFT": {
            "x1": 0,
            "y1": 0,
            "x2": int(image_width * left_w),
            "y2": image_height
        },
        "RIGHT": {
            "x1": int(image_width * (1 - right_w)),
            "y1": 0,
            "x2": image_width,
            "y2": image_height
        },
        "CENTER_DISTANT": {
            "x1": int(image_width * left_w),
            "y1": 0,
            "x2": int(image_width * (1 - right_w)),
            "y2": int(image_height * crossing_top)
        }
    }
    return zones

def get_all_zones_for_bbox(bbox, zones):
    x1, y1, x2, y2 = bbox
    matching_zones = []
    for name, z in zones.items():
        zx1, zy1, zx2, zy2 = z["x1"], z["y1"], z["x2"], z["y2"]
        if not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2):
            matching_zones.append(name)
    return matching_zones

def draw_zones_on_image(image, zones, color_map=None):
    if color_map is None:
        color_map = {
            "CROSSING": (0, 255, 0),
            "LEFT": (255, 0, 0),
            "RIGHT": (0, 0, 255),
            "CENTER_DISTANT": (0, 255, 255)
        }

    for name, z in zones.items():
        color = color_map.get(name, (255, 255, 255))
        cv2.rectangle(image, (z["x1"], z["y1"]), (z["x2"], z["y2"]), color, 2)
        cv2.putText(image, name, (z["x1"] + 5, z["y1"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

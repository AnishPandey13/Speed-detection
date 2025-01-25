import math

import math

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Unique ID count for objects
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and IDs
        objects_bbs_ids = []

        # Get center point of new objects
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if this object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # Adjusted threshold for ground-level videos
                if dist < 50:  # Threshold value can be fine-tuned based on video resolution
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Assign a new ID to the object if no match is found
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Remove old objects not detected in the current frame
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            obj_id = obj_bb_id[4]
            new_center_points[obj_id] = self.center_points[obj_id]
        self.center_points = new_center_points

        return objects_bbs_ids


        # Clean up the dictionary by removing unused IDs
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points
        return objects_bbs_ids

"""
YOLOv8-based building detector (CPU fallback).
Replaces GroundingDinoDetector for CPU/AMD compatibility.
"""

import os
import cv2
import csv
import math
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from .pano2pers import save_bounding_boxes_as_images, _find_image_file


class YoloDetector:
    """
    YOLOv8-based detector for buildings (CPU/AMD compatible).
    """

    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the detector with a YOLOv8 model.
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"YOLOv8 model loaded. Classes: {self.class_names}")

    def filter_boxes_based_on_conditions(self, results, image):
        """
        Filter boxes to keep only the most confident building detection.
        """
        boxes = results[0].boxes
        if len(boxes) == 0:
            return [], []

        # Filter for 'building' class (YOLOv8 class 0 is 'person', adjust as needed)
        building_class_id = None
        for class_id, name in self.class_names.items():
            if "building" in name.lower():
                building_class_id = class_id
                break

        if building_class_id is None:
            return [], []

        # Filter boxes by class and confidence
        filtered_boxes = []
        filtered_conf = []
        for box in boxes:
            if box.cls == building_class_id and box.conf > 0.35:
                filtered_boxes.append(box.xyxy[0].tolist())
                filtered_conf.append(box.conf.item())

        return filtered_boxes, filtered_conf

    def run_detection(
        self,
        geo_aov_csv: str,
        images_folder: str,
        output_folder: str,
        max_aov=130,
        min_aov=10,
        adjust_angle: float = 0.0,
        save_annotated_images: bool = True,
        text_prompt: str = "building",
        max_fov=None
    ):
        """
        Main detection loop. Mimics GroundingDinoDetector.run_detection.
        """
        geo_aov = pd.read_csv(geo_aov_csv, dtype={"pid": str, "building_id": str})
        geo_aov['pid'] = geo_aov['pid'].astype(str)
        geo_aov['left_angle_image'] = (geo_aov['left_angle_geo'] - geo_aov['compass_angle'] + 180) % 360
        geo_aov['right_angle_image'] = (geo_aov['right_angle_geo'] - geo_aov['compass_angle'] + 180) % 360
        geo_aov = geo_aov[(geo_aov['aov_geo'] > min_aov) & (geo_aov['aov_geo'] < max_aov)]

        pid_all = self._check_id(images_folder)
        file_path = os.path.join(output_folder, "building_bbox.csv")
        file_exists = os.path.isfile(file_path)
        already_id = set()
        if file_exists:
            already_id = set(pd.read_csv(file_path, dtype={'pid': str})['pid'].unique())
        remaining_pid = (pid_all & set(geo_aov["pid"])) - already_id
        os.makedirs(output_folder, exist_ok=True)
        if not file_exists:
            with open(file_path, "w", newline="") as file:
                csv.writer(file).writerow(["pid", "building_id", "boxes"])
        print(f"Found {len(pid_all)} images total in {images_folder}, {len(remaining_pid)} remain to process.")

        img_data = []
        for pid in remaining_pid:
            img_path = _find_image_file(pid, images_folder)
            if img_path is not None and os.path.exists(img_path):
                img_data.append({'pid': pid, 'image_url': img_path})

        img_df = pd.DataFrame(img_data, columns=["pid", "image_url"])

        for pid in tqdm(remaining_pid, desc="Processing images", dynamic_ncols=True, leave=False):
            img_row = img_df[img_df['pid'] == pid]
            if img_row.empty:
                continue
            img_path = img_row.iloc[0]['image_url']
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"No image found for PID: {pid}")
                continue

            img_geo_aov = geo_aov[geo_aov['pid'] == pid]

            for _, row in img_geo_aov.iterrows():
                building_id = row['building_id']
                left_bd = row.get('left_angle_image', float("nan"))
                right_bd = row.get('right_angle_image', float("nan"))
                if math.isnan(left_bd) or math.isnan(right_bd):
                    continue

                img_height, img_width = img.shape[:2]
                color = (0, 255, 0)  # Green for YOLOv8

                if right_bd < left_bd:
                    right_bd += 360
                x_left = int(left_bd * img_width / 360 - adjust_angle * img_width)
                x_right = int(right_bd * img_width / 360 + adjust_angle * img_width)
                crop_width = min(x_right - x_left, img_width)
                if crop_width <= 0:
                    continue

                x_left %= img_width
                x_right = x_left + crop_width
                if x_right > img_width:
                    cropped_image = cv2.hconcat([img[:, x_left:], img[:, :x_right - img_width]])
                else:
                    cropped_image = img[:, x_left:x_right]

                # Run YOLOv8 detection
                results = self.model(cropped_image, verbose=False)
                print(f"YOLOv8 results for PID {pid}: {len(results[0].boxes)} boxes")  # Debug
                filtered_boxes, filtered_conf = self.filter_boxes_based_on_conditions(results, cropped_image)
                print(f"Filtered boxes for PID {pid}: {len(filtered_boxes)}")  # Debug
                if not filtered_boxes:
                    # Write a row even if no boxes (for debugging)
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        if not file_exists:
                            writer.writerow(["pid", "building_id", "boxes"])
                            file_exists = True
                        writer.writerow([pid, building_id, "[]"])
                        print(f"Wrote empty box for PID: {pid}, Building ID: {building_id}")  # Debug
                    continue

                # Take the highest-confidence box
                x1, y1, x2, y2 = map(int, filtered_boxes[0])
                conf = filtered_conf[0]

                # Draw box on original image
                x1_orig = x1 + x_left
                x2_orig = x2 + x_left
                if x2_orig > img_width:
                    cv2.rectangle(img, (x1_orig, y1), (img_width - 1, y2), color, 2)
                    cv2.rectangle(img, (0, y1), (x2_orig - img_width, y2), color, 2)
                else:
                    cv2.rectangle(img, (x1_orig, y1), (x2_orig, y2), color, 2)
                text = f"ID: {building_id}, Conf: {conf:.2f}"
                cv2.putText(img, text, (x1_orig, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Normalized box
                cx = ((x1_orig + x2_orig) / 2) % img_width
                cy = (y1 + y2) / 2
                w = x2_orig - x1_orig
                h = y2 - y1
                normalized_box = [cx / img_width, cy / img_height, w / img_width, h / img_height, conf]

                # Write to CSV
                with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(["pid", "building_id", "boxes"])
                        file_exists = True
                    writer.writerow([pid, building_id, normalized_box])
                    print(f"Wrote box for PID: {pid}, Building ID: {building_id}")  # Debug

            if save_annotated_images:
                annotated_dir = os.path.join(output_folder, "annotated_img")
                os.makedirs(annotated_dir, exist_ok=True)
                save_path = os.path.join(annotated_dir, f"{pid}.png")
                cv2.imwrite(save_path, img)

        save_bounding_boxes_as_images(file_path, images_folder, output_folder, max_fov=max_fov)
        print(f"Processing complete. Results saved in {output_folder}.")

    @staticmethod
    def _check_id(folder):
        """
        Retrieve set of IDs from files in the specified folder.
        """
        if not os.path.exists(folder):
            return set()
        return {
            os.path.splitext(name)[0]
            for root, _, files in os.walk(folder)
            for name in files
            if not name.startswith('.') and os.path.isfile(os.path.join(root, name))
        }

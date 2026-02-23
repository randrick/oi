import os
import traceback
from pathlib import Path

import cv2
import numpy as np
import shutil

import database
import license_plate_detection
from utils import File
from config import (
    CONFIDENCE_THRESHOLD,
    INPUT_DIMENSIONS,
    YOLO_MODEL_PATH,
    MOVED_TO_PROCESSED,
    NMS_THRESHOLD,
    OBJECT_DETECTION_OUTPUT_PATH,
    OUTPUT_ROOT_PATH,
    SCORE_START_INDEX,
    YOLO_KEEP_CLASSES,
    YOLO_KEEP_IDS,
    YOLO_RESULT_OFFSET,
    YOLO_VERSION,
)
from face_recognition import recognizeSF
from utils import clip_negative_values, is_label_ignored, load_image, save_image


def initialize_yolo_model():
    if YOLO_VERSION == 8:
        model_file = "yolov8l.onnx"
    elif YOLO_VERSION == 9:
        model_file = "yolov9m.onnx"
    else:
        raise ValueError(f"Unsupported YOLO version: {YOLO_VERSION}")

    model = cv2.dnn.readNetFromONNX(os.path.join(YOLO_MODEL_PATH, model_file))
    if model.empty():
        raise RuntimeError("Failed to load YOLO model")
    return model


yolo_model = initialize_yolo_model()


def detect_objects_in_image(model, image):
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255, INPUT_DIMENSIONS, [0, 0, 0], 1, crop=False
    )
    model.setInput(blob)
    outputs = model.forward(model.getUnconnectedOutLayersNames())

    if YOLO_VERSION == 8:
        outputs = (outputs[0].transpose((0, 2, 1)),)
    return outputs


def process_yolo_output(input_image, yolo_outputs):
    class_ids, confidences, boxes, original_boxes = [], [], [], []
    image_height, image_width = input_image.shape[:2]
    x_factor, y_factor = (
        image_width / INPUT_DIMENSIONS[0],
        image_height / INPUT_DIMENSIONS[1],
    )

    for output in yolo_outputs:
        rows = output.shape[YOLO_RESULT_OFFSET]
        for row in output[0] if YOLO_VERSION == 8 else output:
            scores = row[SCORE_START_INDEX:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id not in YOLO_KEEP_IDS or confidence <= CONFIDENCE_THRESHOLD:
                continue

            confidences.append(float(confidence))
            class_ids.append(class_id)
            boxes.append(scale_bounding_box(row[:4], 1.0, 1.0))
            original_boxes.append(scale_bounding_box(row[:4], x_factor, y_factor))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return class_ids, indices, boxes, original_boxes


def load_and_preprocess_image(image_object):
    image_path = os.path.join(
        image_object.root_path, image_object.file_path, image_object.file_name
    )
    # print(f"Loading: {image_object}")
    # print(f"from path: {image_path}")
    full_size_image = load_image(image_path)
    if full_size_image is None:
        return "", ""
    else:
        reduced_image = cv2.resize(full_size_image.copy(), None, fx=0.4, fy=0.4)

    return full_size_image, reduced_image


def scale_bounding_box(box, x_factor, y_factor):
    cx, cy, w, h = box
    left = int((cx - w / 2) * x_factor)
    top = int((cy - h / 2) * y_factor)
    width = int(w * x_factor)
    height = int(h * y_factor)
    return (left, top, width, height)


def extract_and_process_objects(
    image_object: File,
    image_fullsize,
    class_ids,
    indices,
    boxes,
    original_boxes,
) -> bool:
    output_filename = image_object.file_name_from_datetime()
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    clean_image = image_fullsize.copy()
    for i in indices:
        x, y, w, h = clip_negative_values(boxes[i])
        x_, y_, w_, h_ = clip_negative_values(original_boxes[i])

        cropped_image = clean_image[y_ : y_ + h_, x_ : x_ + w_].copy()
        label = YOLO_KEEP_CLASSES[class_ids[i]]
        color = colors[i]

        if class_ids[i] in range(18, 25):
            label = "deer"

        cv2.rectangle(image_fullsize, (x_, y_), (x_ + w_, y_ + h_), color, 1)
        cv2.putText(image_fullsize, label, (x_, y_ + 20), font, 2, color, 2)

        if not is_label_ignored(label):
            process_detected_object(
                image_object, label, cropped_image, output_filename, i
            )

            if boxes:
                save_image(
                    os.path.join(OBJECT_DETECTION_OUTPUT_PATH, image_object.file_name),
                    image_fullsize,
                )
                return True
    return False

    # if SHOW_PREVIEW:
    #     cv2.imshow("Image", image_fullsize)
    #     cv2.waitKey(1)


def process_detected_object(image_object, label, cropped_image, output_filename, index):

    crop_fn = f"{output_filename}_{index}_{image_object.file_extension}"
    crop_path = os.path.join(OUTPUT_ROOT_PATH, label)
    crop_fqfn = os.path.join(crop_path, crop_fn)

    Path(crop_path).mkdir(parents=True, exist_ok=True)
    save_image(crop_fqfn, cropped_image)

    color, detection_result = "", ""
    if label in ["car", "truck", "bus", "motorcycle", "person"]:
        detection_result = add_car_and_people_insights(
            label=label,
            image_fqfn=crop_fqfn,
            output_fn=f"{label}_{output_filename}_{index}_{image_object.file_extension}",
        )

    database.insert_value(
        image_object.name,
        label,
        image_object.file_path,
        image_object.file_name,
        image_object.file_create_date(),
        crop_fn,
        detection_result,
        color,
    )


def add_car_and_people_insights(label, image_fqfn, output_fn, use_rotation=False):
    return_value = None
    try:
        if label in ["car", "truck", "bus", "motorcycle"]:
            return_value = license_plate_detection.detect_license_plate(image_fqfn)
        elif label == "person":
            return_value = recognizeSF.recognize(image_fqfn, output_fn)
    except Exception as e:
        print(f"Error in object detection: {e}")
        print(traceback.format_exc())
    return return_value


"""Move the image file to the processed folder."""


def move_to_processed(image_record):
    fqfn = str(
        os.path.join(
            image_record.root_path, image_record.file_path, image_record.file_name
        )
    )
    root, _ = os.path.splitext(fqfn)
    path, _ = os.path.split(root)
    dest_path = os.path.join(path, "processed")
    try:
        shutil.move(fqfn, dest_path)
    except Exception as e:
        print(f"Failed to move {fqfn} to {dest_path}")
        print(traceback.format_exc())
        os.remove(fqfn)


def analyze_image(image_object):

    try:

        full_size_image, _ = load_and_preprocess_image(image_object)

        yolo_outputs = detect_objects_in_image(yolo_model, full_size_image)

        class_ids, indices, boxes, original_boxes = process_yolo_output(
            full_size_image, yolo_outputs
        )

        if (
            extract_and_process_objects(
                image_object,
                full_size_image,
                class_ids,
                indices,
                boxes,
                original_boxes,
            )
            and MOVED_TO_PROCESSED
        ):
            move_to_processed(image_object)
        else:
            fqfn = os.path.join(
                image_object.root_path, image_object.file_path, image_object.file_name
            )
            os.remove(fqfn)

    except EOFError as e:
        raise e
    except Exception as e:
        print(f"Error in image analysis: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    # Main execution code (if needed)
    pass

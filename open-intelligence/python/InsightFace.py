import argparse
import os
import pickle
import sys
import time
import traceback

import cv2
import psycopg2

import database
from service_instance import instance

from utils import load_image, save_image, crop_image
from config import INSIGHTFACE_OUTPUT_PATH, OUTPUT_ROOT_PATH, PROCESS_SLEEP_SECONDS
from face_recognition import recognizeSF

# Check does system has GPU support
# print('GPU support available: ' + str(gpu_utils.is_gpu_available()))

# load serialized face detector from disk
model_root_path = os.path.join(os.getcwd(), "models/face_detection_model")
proto_path = os.path.join(model_root_path, "deploy.prototxt")
model_path = os.path.join(model_root_path, "res10_300x300_ssd_iter_140000.caffemodel")

model_root_path = os.path.join(os.getcwd(), "models/YN-SF")
yn_model = os.path.join(model_root_path, "face_detection_yunet_2022mar.onnx")
sf_model = os.path.join(model_root_path, "face_recognition_sface_2021dec.onnx")
faces_db_path = os.path.join(model_root_path, "faces_db.pickle")

# try:
#     output_root_folder_path = os.environ['OUTPUT_FOLDER']
# except KeyError as e:
#     print(e)


def app(detector, recognizer, label_encoder, batch_size):
    # Do work
    work_records = database.get_insight_face_images_to_compute()

    if len(work_records) == 0:
        print("No insightFace images to process")
        return
    for row in work_records:
        # Get db row fields
        id = row[0]
        label = row[1]
        cropped_file_name = row[2]
        detection_result = row[3]

        input_image = os.path.join(OUTPUT_ROOT_PATH, label, cropped_file_name)
        if not os.path.exists(input_image):
            return

        try:
            detection_result = detect_and_recognize_faces(
                input_image, cropped_file_name, recognizer, detector, label_encoder
            )
        except Exception as e:
            database.update_insight_face_as_computed("", id)
            print(e)

        # Save result if specified conditions are true
        if detection_result is None:
            detection_result = ""

        # Write database, set as computed
        database.update_insight_face_as_computed(detection_result, id)


def detect_and_recognize_faces(image_path, file_name, recognizer, detector, faces_db):
    """
    Detect faces in an image and attempt to recognize them.
    """
    try:
        image = load_image(image_path)
        if image is None:
            return None

        faces = recognizeSF.detect_faces(detector, image, image_path)
        # if faces is None or faces[1] is None:
        if faces is None:
            return None

        # for i, face in enumerate(faces[1]):
        for i, face in enumerate(faces):
            face_image = crop_image(image, face[:4].astype(int))
            if face_image is None:
                continue
            save_image(os.path.join(INSIGHTFACE_OUTPUT_PATH, file_name), face_image)
            recognition_result = recognizeSF.recognize_for_insight_face(
                image, faces, recognizer, faces_db, image_path
            )

            print(recognition_result)
            return recognition_result

    except Exception as e:
        print(f"Error in detect_and_recognize_faces: {e}")
        print(traceback.format_exc())

    return None


# ---------------------------------------------------------------------
# Keeps program running


def main_loop(batch_size):
    print("[INFO] loading face detector...")
    detector = cv2.FaceDetectorYN.create(yn_model, "", (320, 320), 0.9, 0.3, 5000)

    recognizer = cv2.FaceRecognizerSF.create(sf_model, "")
    label_encoder = pickle.loads(open(faces_db_path, "rb").read())

    print("[Info] loaded")
    while 1:
        try:
            instance.set_instance_status()
            app(detector, recognizer, label_encoder, batch_size)
            print("... running")
        except psycopg2.OperationalError as e:
            print(e)
        time.sleep(int(PROCESS_SLEEP_SECONDS))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Insight Face - face recognition")
        parser.add_argument(
            "--batch",
            dest="batch_size",
            type=int,
            default=10,
            help="size of batch to process",
        )

        args = parser.parse_args()
        print("Using batch size of " + str(args.batch_size))
        main_loop(args.batch_size)
    except KeyboardInterrupt:
        print >> sys.stderr, "\nExiting by user request.\n"
        sys.exit(0)

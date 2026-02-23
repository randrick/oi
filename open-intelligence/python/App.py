"""Main processing of new image files into db with object detection, etc."""

import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import psycopg2
from filelock import FileLock

import database
import object_detection
import service_instance
from utils import File
from config import (
    CAMERA_FOLDERS_CONFIG,
    CAMERA_NAMES_CONFIG,
    CAMERAS_ROOT_PATH,
    INSIGHTFACE_OUTPUT_PATH,
    MOVED_TO_PROCESSED,
    OUTPUT_ROOT_PATH,
    PROCESS_SLEEP_SECONDS,
    TEST_MOVE_PATH,
)
from face_recognition import extract_embeddings, train_model
from utils import get_time_sorted_files

MAX_FILES_TO_TAKE = 100

Path(INSIGHTFACE_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(TEST_MOVE_PATH).mkdir(parents=True, exist_ok=True)


"""Grab all camera folders and their names."""


def camera_folder_setup() -> zip:
    names = []
    folders = []
    for camera_name, camera_folder in zip(CAMERA_NAMES_CONFIG, CAMERA_FOLDERS_CONFIG):
        names.append(camera_name)
        folders.append(camera_folder)
        if MOVED_TO_PROCESSED:
            Path(os.path.join(CAMERAS_ROOT_PATH, camera_folder, "processed/")).mkdir(
                parents=True, exist_ok=True
            )
    return zip(names, folders)


"""Detect objects in all camera images."""


def detect_objects(image_record: File):
    fqfn = str(
        os.path.join(
            image_record.root_path, image_record.file_path, image_record.file_name
        )
    )
    lock_path = fqfn + ".lock"
    lock = FileLock(lock_path, timeout=1)

    with lock.acquire():
        try:
            object_detection.analyze_image(image_record)

        except EOFError as e:
            handle_invalid_image(image_record, e)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        finally:
            lock.release()

    if not lock.is_locked and os.path.exists(lock_path):
        os.remove(lock_path)


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


"""Handle exceptions when processing an image."""


def handle_invalid_image(sorted_file, error):
    print("Invalid image file or storage may be run out", error)
    database.insert_notification(
        f"Invalid image file or camera storage may be run out for camera {sorted_file.name}"
    )
    os.remove(
        os.path.join(
            sorted_file.root_path, sorted_file.file_path, sorted_file.file_name
        )
    )


"""Main function to process images."""


def process_images():
    camera_folders = camera_folder_setup()
    image_files = get_time_sorted_files(camera_folders, target_extention=".jpg")[
        :MAX_FILES_TO_TAKE
    ]

    for image_file in image_files:
        # print(f"process_image {image_file}")
        detect_objects(image_file)


"""Check for tasks to be performed."""


def check_for_tasks():
    # Face model training commands
    if database.bool_run_train_face_model():
        print("[INFO] Training face model")

        # Extract embeddings
        extract_embeddings.extract_embeddings(
            cwd_path=os.getcwd(), input_confidence=0.5
        )

        # Train model
        train_model.train_model(cwd_path=os.getcwd())

    # Detection tasks for re processing
    detection_tasks = database.get_detection_tasks()
    for row in detection_tasks:
        # Get db row fields
        id = row[0]
        label = row[1]
        image_name = row[2]
        print("[INFO] Processing re-recognition task " + image_name)

        image_fqfn = os.path.join(OUTPUT_ROOT_PATH, label)

        detection_result = object_detection.add_car_and_people_insights(
            label=label, image_fqfn=image_fqfn, output_file_name=None
        )

        # Write new detection result
        database.update_detection_task_result(id, detection_result)


"""Main loop to process images and perform tasks."""


def main_loop():
    while True:
        try:
            service_instance.instance.set_instance_status()
            check_for_tasks()
            # process_images()
            print("... running")
        except psycopg2.OperationalError as e:
            print(e)
        time.sleep(PROCESS_SLEEP_SECONDS)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Exiting by user request.", file=sys.stderr)
        sys.exit(0)

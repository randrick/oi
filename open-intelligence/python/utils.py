import os
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from srFile import SrFile

import database

from config import (
    CAMERAS_ROOT_PATH,
    DELETE_FILES,
    IGNORED_LABELS,
    IMAGE_SIMILARITY_THRESHOLD,
    OUTPUT_ROOT_PATH,
    TEST_MOVE_PATH,
)


class File:
    def __init__(self, name, root_path, file_path, file_name):
        fqfn = os.path.join(root_path, file_path, file_name)
        if not os.path.exists(fqfn):
            raise FileNotFoundError(
                f"File {self.file_name} not found at {self.root_path + self.file_path}"
            )
        self.name = name  # Known as camera name
        self.root_path = root_path
        self.file_path = file_path
        self.file_name = file_name
        self.file_extension = self.get_file_extension(root_path, file_path, file_name)
        self.time_stamp = self.file_create_date()

    def __str__(self):
        return f"{self.name} - {self.root_path} - {self.file_path} -  {self.file_name}"

    def file_name_from_datetime(self):

        dt = self.file_create_date()
        return dt.strftime("%Y_%m_%d_%H_%M_%S")

    def file_create_date(self) -> datetime:

        # 20241227103016001
        #   %Y%m%d%H%M%S%f
        if re.search("_\d{17}_", self.file_name):

            dt_str = self.file_name.split("_")[2]
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S%f")

        else:

            dt = datetime.fromtimestamp(
                os.path.getmtime(
                    os.path.join(self.root_path, self.file_path, self.file_name)
                )
            )

        return dt

    def get_file_extension(self, root_path, file_path, file_name):
        filename, file_extension = os.path.splitext(
            os.path.join(root_path, file_path, file_name)
        )
        return file_extension


def is_label_ignored(label):
    return label in IGNORED_LABELS


def clip_negative_values(values):
    return [max(0, value) for value in values]


def load_image(fqfn, record_id=None):

    image = None

    if os.path.exists(fqfn):
        try:
            image = cv2.imread(fqfn)
        except IOError as e:
            print(f"An I/O error occurred reading {fqfn}: {e}")
            print(traceback.format_exc())
            try:
                os.remove(fqfn)
            except IOError as e2:
                print(f"An I/O error occurred deleting {fqfn}: {e2}")
                print(traceback.format_exc())

    if image is None:
        print(f"No image found for {fqfn}")

    # print('loaded: '+fqfn)

    return image


def save_image(fqfn, image):
    try:
        path, _ = os.path.split(fqfn)
        Path(path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(fqfn, image)
    except IOError as e:
        print(f"An I/O error occurred: {e}")
        print(traceback.format_exc())


def crop_image(image: np.ndarray, box: np.ndarray):
    """Extract a face from the image using the given coordinates."""
    # coords = box[:-1].astype(np.int32)
    print(box)
    start_x, start_y, width, height = box
    end_x, end_y = start_x + width, start_y + height
    cropped_image = image[start_y:end_y, start_x:end_x]

    if image.shape[0] * image.shape[1] < 1024:
        return None

    return cropped_image


def get_images(folder: str) -> List[str]:
    files = []
    print(CAMERAS_ROOT_PATH)
    path = os.path.join(CAMERAS_ROOT_PATH, folder)
    print("path: " + path)
    for file_name in os.listdir(path):
        try:
            # print("Camera_root_path: " + str(CAMERAS_ROOT_PATH))
            fqfn = os.path.join(CAMERAS_ROOT_PATH, folder, file_name)
            if (
                os.path.isdir(fqfn)
                or file_name == "Thumbs.db"
                or file_name.find(".lock") != -1
                or os.path.getsize(fqfn) == 0
            ):
                continue

            # print("Processing fqfn:  " + fqfn)
            files.append(file_name)
        except Exception as e:
            print(e)
            print("Error writing " + str(fqfn))
            print(traceback.format_exc())
    return files


def get_time_sorted_files(
    names_folders: zip, target_extention: str = "all"
) -> List[File]:
    time_sorted_files: List[File] = []

    for name, folder in names_folders:
        print("Processing camera " + name)
        for file_name in get_images(folder):
            _, source_extension = os.path.splitext(file_name)
            if target_extention == "jpg" and source_extension == "jpeg":
                pass
            else:
                if target_extention != "all" and source_extension != target_extention:
                    continue

            file_object = File(name, "/input", folder, file_name)
            time_sorted_files.append(file_object)

    sz = len(time_sorted_files)
    # for (seed, fo) in zip(random.sample(range(10 * sz), sz), time_sorted_files):
    #     fo.seed = seed
    time_sorted_files.sort(key=sort_function)
    print("returning " + str(len(time_sorted_files)) + " files.")
    return time_sorted_files


def sort_function(e):
    return e.time_stamp


def read_and_preprocess_image(image_name):
    if os.path.exists(image_name):
        try:
            img = load_image(image_name)
            img = cv2.resize(img, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        except Exception as e:
            print(f"Error reading image {image_name}: {e}")
    return None


def compare_images(image_a, image_b):
    if image_a is not None and image_b is not None:
        s = ssim(image_a, image_b)
        return s > IMAGE_SIMILARITY_THRESHOLD
    return False


def handle_similar_image(similar_object, original_object, sim):
    try:
        if DELETE_FILES:
            os.remove(similar_object.input_image)
        else:
            shutil.move(
                similar_object.input_image,
                str(os.path.join(TEST_MOVE_PATH, original_object.image_name)),
            )
            fqfn = f"{OUTPUT_ROOT_PATH}/{similar_object.label}/super_resolution/{similar_object.input_image}"
            if os.path.exists(fqfn):
                os.remove(fqfn)
        similar_object.output_image = 0
    except Exception as e:
        print(e)
        print(traceback.format_exc())


def remove_image_and_record(sr_record: SrFile):
    raise NotImplementedError


def process_image_objects(similarity_image_objects, threshold):
    """
    Processes similarity image objects to identify and handle similar images.

    Args:
        similarity_image_objects (list): List of SimilarityObject instances to process.

    Returns:
        int: Count of processed records.
    """
    size = len(similarity_image_objects)
    count = 0
    if size > 1:
        for i in range(size - 1):
            so1 = similarity_image_objects[i]
            if so1.output_image == 0:
                continue

            curr = read_and_preprocess_image(so1.input_image)
            if curr is None:
                print(
                    "Could not read {}, removing record: {}".format(
                        so1.input_image, so1
                    )
                )
                database.delete_row(so1.id)
                if os.path.exists(so1.input_image):
                    os.remove(so1.input_image)
                count += 1
                continue

            last_similar = None
            la = 200  # lookahead limit

            for j in range(i + 1, size):
                so2 = similarity_image_objects[j]
                if should_break_comparison(so2, so1, j, i, last_similar, la):
                    break
                if so2.output_image == 0:
                    continue

                next_img = read_and_preprocess_image(so2.input_image)
                if next_img is None:
                    print(
                        "Could not read {}, removing record: {}".format(
                            so2.input_image, so2.id
                        )
                    )
                    so2.output_image = 0
                    database.delete_row(so2.id)
                    if os.path.exists(so2.input_image):
                        os.remove(so2.input_image)
                    count += 1
                    continue

                sim = ssim(curr, next_img)
                if sim > threshold:
                    count += 1
                    last_similar = j
                    handle_similar_image(so2, so1, sim)

            database.update_similarity_check_row_checked(so1.id)

    return count


def read_and_preprocess_image(image_name):
    if os.path.exists(image_name):
        try:
            img = load_image(image_name)
            img = cv2.resize(img, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        except Exception as e:
            print(f"Error reading image {image_name}: {e}")
    return None


def should_break_comparison(so2, so1, j, i, last_similar, la):
    return (
        (so1.label != so2.label)
        or (last_similar is None and j > i + la)
        or (last_similar is not None and j > last_similar + la)
        or ()
    )


def handle_similar_image(so2, so1, sim):
    try:
        if DELETE_FILES:
            os.remove(so2.input_image)
        else:
            shutil.move(so2.input_image, TEST_MOVE_PATH + so1.image_name)
            fqfn = f"{OUTPUT_ROOT_PATH}/{so2.label}/super_resolution/{so2.input_image}"
            if os.path.exists(fqfn):
                os.remove(fqfn)

        database.delete_row(so2.id)
        so2.output_image = 0
    except Exception as e:
        print(e)
        print(traceback.format_exc())


def is_null_empty_or_whitespace(input_variable) -> bool:
    return input_variable is None or input_variable == "" or input_variable == " "

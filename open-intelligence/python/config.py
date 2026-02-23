import os
from pathlib import Path
from typing import List, Set

import database

# App configuration
APP_CONFIG = database.get_application_config()
# print(APP_CONFIG)
# flags, booleans, oh my.
MOVED_TO_PROCESSED: bool = (
    database.find_config_value(APP_CONFIG, "move_to_processed") == "True"
)
DELETE_FILES: bool = database.find_config_value(APP_CONFIG, "delete_files") == "True"
SHOW_PREVIEW: bool = (
    database.find_config_value(APP_CONFIG, "cv2_imshow_enabled") == "True"
)
USE_GPU: bool = database.find_config_value(APP_CONFIG, "use_gpu") == "True"

PROCESS_SLEEP_SECONDS: int = int(
    database.find_config_value(APP_CONFIG, "process_sleep_seconds")
)
SR_MAX_WIDTH: int = int(database.find_config_value(APP_CONFIG, "max_width"))
SR_MAX_HEIGHT: int = int(database.find_config_value(APP_CONFIG, "max_height"))

# Camera configuration
# CAMERAS_ROOT_PATH: Path = Path(database.find_config_value(APP_CONFIG, "cameras_root_path"))
CAMERAS_ROOT_PATH: Path = Path("/input")
CAMERA_NAMES_CONFIG: List[str] = database.find_config_value(
    APP_CONFIG, "camera_names"
).split(",")
CAMERA_FOLDERS_CONFIG: List[str] = database.find_config_value(
    APP_CONFIG, "camera_folders"
).split(",")

# Define paths
OUTPUT_ROOT_PATH: Path = database.find_config_value(APP_CONFIG, "output_folder")

MODEL_ROOT_PATH: Path = os.path.join(os.getcwd(), "models")
YOLO_MODEL_PATH: Path = os.path.join(MODEL_ROOT_PATH, "yolo")

INSIGHTFACE_OUTPUT_PATH: Path = os.path.join(OUTPUT_ROOT_PATH, "insightface", "faces")
OBJECT_DETECTION_OUTPUT_PATH: Path = os.path.join(OUTPUT_ROOT_PATH, "object_detection")
TEST_MOVE_PATH: Path = os.path.join(OUTPUT_ROOT_PATH, "recycle")
PERSON_OUTPUT_PATH: Path = os.path.join(OUTPUT_ROOT_PATH, "person")
FACES_OUTPUT_PATH: Path = os.path.join(OUTPUT_ROOT_PATH, "faces")

FACES_DB_PATH: Path = os.path.join(os.getcwd(), "models", "YN-SF")
FACES_TRAINING_PATH: Path = os.path.join(OUTPUT_ROOT_PATH, "faces_dataset")

FILE_NAME_PREFIX: str = str(database.find_config_value(APP_CONFIG, "file_name_prefix"))

IMAGE_SIMILARITY_THRESHOLD: float = 0.3

# YOLO configuration
YOLO_VERSION: int = 8
SCORE_START_INDEX: int = 4 if YOLO_VERSION >= 8 else 5
YOLO_RESULT_OFFSET: int = 0 if YOLO_VERSION == 4 else 1
INPUT_DIMENSIONS: tuple = (640, 640) if YOLO_VERSION >= 8 else (608, 608)
NMS_THRESHOLD: float = 0.45
CONFIDENCE_THRESHOLD: float = 0.5


YOLO_KEEP_CLASSES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

YOLO_KEEP_IDS: Set = {0, 1, 2, 3, 5, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24}

IGNORED_LABELS: Set = set(
    database.find_config_value(APP_CONFIG, "ignored_labels").split(",")
)

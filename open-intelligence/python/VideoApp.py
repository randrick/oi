import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import psycopg2
from filelock import FileLock

import database
from utils import get_time_sorted_files, save_image

# Config
app_config = database.get_application_config()
move_to_processed = (
    database.find_config_value(app_config, "move_to_processed") == "True"
)
process_sleep_seconds = int(
    database.find_config_value(app_config, "process_sleep_seconds")
)

# Get current time offset
ts = time.time()

utc_offset = (
    datetime.fromtimestamp(ts) - datetime.utcfromtimestamp(ts)
).total_seconds()
time_offset_hours = int(utc_offset / 60 / 60)
print("Time offset: " + str(time_offset_hours))

# Specify your names and folders at config.ini
# split them by a,b,c,d
names = []  # ['App1']
folders = []  # [os.getcwd() + '/images/']

# Process arguments
# parser = ArgumentParser()
# parser.add_argument('--bool_slave_node', type=str,
#                     help='Multi node support, give string True as input if slave.')
# args = parser.parse_args()

# Parse camera name and folder config
cameras_root_path = database.find_config_value(app_config, "cameras_root_path")
camera_names_config = database.find_config_value(app_config, "camera_names").split(",")
camera_folders_config = database.find_config_value(app_config, "camera_folders").split(
    ","
)

# Append in names and folders
for n, f in zip(camera_names_config, camera_folders_config):
    names.append(n)
    folders.append(f)
    Path(os.path.join(cameras_root_path, f, "processed/")).mkdir(
        parents=True, exist_ok=True
    )

max_files_to_take = 1

# Return source folder files sorted by create date


def extract_images_from_video(video_object):
    video_path = os.path.join(video_object.root_path, video_object.file_path)
    video_fqfn = os.path.join(video_path, video_object.file_name)
    video = cv2.VideoCapture(video_fqfn)

    base_name, _ = os.path.splitext(video_object.file_name)
    count = 0
    success = 1

    while success:

        success, image = video.read()
        if success:
            # Camera is set to 20 fps

            if count % 15 == 0:
                # Saves the frames with frame-count

                save_image(os.path.join(video_path , base_name + '-' + str(count)), image)

            count += 1


# Pick files for processing, lock and process them
def app():

    # Videos for processing (.mp4)
    sorted_files = get_time_sorted_files(
        cameras_root_path, names, folders, type=".mp4"
    )[:max_files_to_take]

    # Create image objects
    processed_files = 0
    for sorted_file in sorted_files:

        # File locking check and locking
        fqfn = os.path.join(
            sorted_file.root_path, sorted_file.file_path, sorted_file.file_name
        )
        lock_path = fqfn + ".lock"
        lock = FileLock(lock_path, timeout=1)
        with lock.acquire():
            try:
                processed_files += 1
                extract_images_from_video(sorted_file)
                if move_to_processed:
                    # shutil.move(fqfn, sorted_file.root_path +
                    #             sorted_file.file_path + 'processed/' + sorted_file.file_name)
                    os.remove(fqfn)
                else:
                    os.remove(fqfn)

            except EOFError as e:
                print("Invalid image file or storage may be run out", e)
                database.insert_notification(
                    "Invalid image file or camera storage may be run out for camera "
                    + sorted_file.name
                )
                # try remove the file
                os.remove(
                    os.path.join(
                        sorted_file.root_path,
                        sorted_file.file_path,
                        sorted_file.file_name,
                    )
                )
            except Exception as e:
                print(e)
                print(traceback.format_exc())
            finally:
                lock.release()

            if not lock.is_locked and os.path.exists(lock_path):
                os.remove(lock_path)


# ---------------------------------------------------------------------
# Keeps program running


def main_loop():
    while 1:
        try:
            app()
            print("Video-App ... running")
        except psycopg2.OperationalError as e:
            print(e)
        time.sleep(process_sleep_seconds)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print >> sys.stderr, "\nExiting by user request.\n"
        sys.exit(0)

# ---------------------------------------------------------------------

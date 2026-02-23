import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

import database
from utils import save_image

# Paths
grab_image_output_path = os.getcwd() + "/images/"

# Check folder existence
Path(grab_image_output_path).mkdir(parents=True, exist_ok=True)

# Config
app_config = database.get_application_config()
sleep_seconds = float(database.find_config_value(app_config, "sleep_seconds"))
jpeg_stream_names = database.find_config_value(app_config, "jpeg_stream_names").split(
    ","
)
steam_urls_config = database.find_config_value(app_config, "jpeg_streams").split(",")


def grab():
    for stream_name, stream_url in zip(jpeg_stream_names, steam_urls_config):
        cap = cv2.VideoCapture(stream_url)
        ret, img = cap.read()

        # Create file name from date time
        now = datetime.now()
        output_file_name = str(
            stream_name + "_" + now.strftime("%d_%m_%Y_%H_%M_%S") + ".jpg"
        )

        print(output_file_name)

        # Write output image
        save_image(   os.path.join(grab_image_output_path, output_file_name), img )


# ---------------------------------------------------------------------
# Keeps program running


def main_loop():
    while 1:
        grab()
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print >> sys.stderr, "\nExiting by user request.\n"
        sys.exit(0)

# ---------------------------------------------------------------------

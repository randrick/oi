import os
from os import environ
from pathlib import Path

import cv2
import imutils
import numpy as np

from libraries.openalpr_64.openalpr import Alpr
import database
from utils import load_image, save_image

# Custom config
app_config = database.get_application_config()

output_root_folder_path = database.find_config_value(app_config, "output_folder")
alpr_enabled = (
    database.find_config_value(app_config, "enabled") == "True"
)  # Todo, rename this variable
region = database.find_config_value(app_config, "region")
plate_char_length = database.find_config_value(app_config, "plate_char_length")
use_plate_char_length = (
    database.find_config_value(app_config, "use_plate_char_length") == "True"
)

# Paths
car_labels_path = output_root_folder_path + "/car/"
rotation_temp_images_path = output_root_folder_path + "/rotation_temp/"
alpr_dir = os.getcwd() + "/libraries/openalpr_64"
open_alpr_conf = os.getcwd() + "/libraries/openalpr_64/openalpr.conf"
open_alpr_runtime_data = os.getcwd() + "/libraries/openalpr_64/runtime_data"


class Plate:

    # Class constructor
    def __init__(self, plate, confidence):
        self.plate = plate
        self.confidence = confidence


def detect_license_plate(image_fqfn):
    try:

        if alpr_enabled and os.path.exists(image_fqfn):

            result_plates = []  # From here we pick one with highest confidence
            result_plate = None

            # Set path for alpr
            environ["PATH"] = alpr_dir + ";" + environ["PATH"]

            all_images = [image_fqfn]
            input_image = load_image(image_fqfn)
            # Todo.. make settings about this below feature, this is making process very very slow
            rotation_images = []
            get_rotation_images(input_image, image_fqfn)
            all_images = all_images + rotation_images  # append together

            # Initialize openalpr
            alpr = Alpr(region, open_alpr_conf, open_alpr_runtime_data)
            if not alpr.is_loaded():
                print("Error loading OpenALPR")
                return ""
            alpr.set_top_n(7)
            alpr.set_default_region("md")

            for image in all_images:

                # Image file is loaded here
                results = alpr.recognize_file(image)
                # print("Run ALPR for: " + image)

                i = 0
                for plate in results["results"]:
                    i += 1
                    print("Plate #%d" % i)
                    print("   %12s %12s" % ("Plate", "Confidence"))
                    for candidate in plate["candidates"]:
                        prefix = "-"
                        if candidate["matches_template"]:
                            prefix = "*"

                        print(
                            "  %s %12s%12f"
                            % (prefix, candidate["plate"], candidate["confidence"])
                        )
                        license_plate = candidate["plate"]
                        confidence = candidate["confidence"]

                        if use_plate_char_length:
                            if len(license_plate) == plate_char_length:
                                # Take specified length one
                                result_plates.append(
                                    Plate(
                                        region_filter(license_plate, region), confidence
                                    )
                                )
                                break
                        else:
                            # Take first one (highest confidence)
                            result_plates.append(
                                Plate(region_filter(license_plate, region), confidence)
                            )
                            break

                    # Call when completely done to release memory
                    try:
                        alpr.unload()
                    except Exception as e:
                        print(e)

                # Delete temporary rotation images
                if len(rotation_images) > 0:
                    for ri in rotation_images:
                        os.remove(ri)

                # Sort array
                result_plates.sort(key=lambda x: x.confidence, reverse=True)

                # Take first if has one
                if len(result_plates) > 0:
                    result_plate = result_plates[0].plate

                # Final result
                # if result_plate is not None:
                #     print('Result plate: ' + result_plate)
                # else:
                #     print('Did not recognize any license plate.')

                return result_plate

    except AssertionError as e:
        print(e)

    return ""


# Rotate image to boost plate finding probability
def get_rotation_images(input_image, image_name):

    rotation_images = []

    # Check temp folder existence
    Path(rotation_temp_images_path).mkdir(parents=True, exist_ok=True)

    i = 0
    for angle in np.arange(-30, 30, 4):
        rotated = imutils.rotate_bound(input_image, angle)
        file_name = (
            rotation_temp_images_path + "rotation_" + image_name + "_" + str(i) + ".jpg"
        )
        save_image(file_name, rotated)
        rotation_images.append(file_name)
        i = i + 1

    return rotation_images


# Filter plate based on region
def region_filter(license_plate, region="eu"):
    # Europe
    if region == "eu":
        if len(license_plate) >= 6:
            a = license_plate[0:3]
            b = (
                license_plate[3 : len(license_plate)]
                .replace("I", "1")
                .replace("O", "0")
                .replace("S", "5")
                .replace("B", "8")
                .replace("D", "0")
                .replace("Z", "2")
            )
            license_plate = a + b
    # More region specific rules?
    return license_plate

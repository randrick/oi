import os
import sys
import time
import gc
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import psycopg2

import database
from service_instance import instance as service_instance
from config import OUTPUT_ROOT_PATH, PROCESS_SLEEP_SECONDS, SR_MAX_HEIGHT, SR_MAX_WIDTH
from object_detection import add_car_and_people_insights
from utils import is_null_empty_or_whitespace
from vehicle_color import vehicle_color_detect
from srFile import SrFile


# Running the SR model

# This is a model of Enhanced Super Resolution GAN Model
# The link given here is a model of ESRGAN model
esrgn_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(esrgn_path)


# Model to preprocess the images
def preprocessing(img):
    imageSize = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    cropped_image = tf.image.crop_to_bounding_box(img, 0, 0, imageSize[0], imageSize[1])
    preprocessed_image = tf.cast(cropped_image, tf.float32)
    return tf.expand_dims(preprocessed_image, 0)


def srmodel(img):
    preprocessed_image = preprocessing(img)  # Preprocess the LR Image
    # returns the size of the original argument that is given as input
    return tf.squeeze(model(preprocessed_image)) / 255.0


def app():
    # Do work
    sr_work_records = database.get_super_resolution_images_to_compute()
    sr_image_objects = []
    for row in sr_work_records:
        # Get db row fields
        id = row[0]
        label = row[1]
        cropped_file_name = row[2]
        detection_result = row[3]

        # Construct paths
        input_image_fqfn = os.path.join(OUTPUT_ROOT_PATH, label, cropped_file_name)
        output_image_path = os.path.join(OUTPUT_ROOT_PATH, label, "super_resolution/")
        output_image_fqfn = os.path.join(output_image_path, cropped_file_name)

        # Check path existence
        Path(output_image_path).mkdir(parents=True, exist_ok=True)

        # Make objects
        sr_image_object = SrFile(
            id,
            label,
            cropped_file_name,
            input_image_fqfn,
            output_image_fqfn,
            detection_result,
            "",
        )
        sr_image_objects.append(sr_image_object)

    # Super resolution image
    if len(sr_image_objects) > 0:
        # Process super resolution images
        sr_image_objects = process_super_resolution_images(
            sr_image_objects=sr_image_objects,
            max_width=SR_MAX_WIDTH,
            max_height=SR_MAX_HEIGHT,
        )

        # Process results
        for sr_image_object in sr_image_objects:
            # Label based detection if not detected earlier
            if not is_null_empty_or_whitespace(sr_image_object.detection_result):

                sr_image_object.detection_result = add_car_and_people_insights(
                    label=sr_image_object.label,
                    image_fqfn=sr_image_object.output_image,
                    output_fn=sr_image_object.label + "_" + sr_image_object.image_name,
                    use_rotation=True,
                )

                # Try to detect color
                try:
                    sr_image_object.color = vehicle_color_detect.detect_color(
                        sr_image_object.output_image
                    )
                except Exception as e:
                    print(e)

                # Write database, row no longer processed later
                database.update_super_resolution_row_result(
                    sr_image_object.detection_result,
                    sr_image_object.color,
                    sr_image_object.image_name,
                    sr_image_object.id,
                )
        else:
            print("No new sr image objects to process")


def process_super_resolution_images(sr_image_objects, max_width, max_height):

    # Loop over all images
    # Input and output image is full path + filename including extension
    for sr_image_object in sr_image_objects:

        print("Processing file: " + os.path.basename(sr_image_object.input_image))

        # We may not have image available at all, pass
        try:
            # Read image
            low_res = cv2.imread(sr_image_object.input_image, 1)

            # Get image size details
            original_h, original_w, original_c = low_res.shape
            # print("Sr processing img size: " + str(original_h) + ":" + str(original_w))

            # Check if image is not too big
            # Original size was 1200 but testing with smaller
            if original_w < max_width and original_h < max_height:
                # Convert to RGB (opencv uses BGR as default)
                low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

                sr = srmodel(low_res).numpy()

                # Convert back to BGR for opencv
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                # Save the results:
                cv2.imwrite(sr_image_object.output_image, sr)

                # Save sr image data to object
                sr_image_object.set_sr_image_data(sr)

                # Clear sr object
                sr = None
            else:
                # Save original image
                sr_image_object.set_sr_image_data(low_res)

        except Exception as e:
            print(e)

        gc.collect()

    return sr_image_objects


# ---------------------------------------------------------------------
# Keeps program running


def main_loop():
    while 1:
        try:
            service_instance.set_instance_status()
            app()
            print("... running")
        except psycopg2.OperationalError as e:
            print(e)
        time.sleep(int(PROCESS_SLEEP_SECONDS))


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print >> sys.stderr, "\nExiting by user request.\n"
        sys.exit(0)

# ---------------------------------------------------------------------

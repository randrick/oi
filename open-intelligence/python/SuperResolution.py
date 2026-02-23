import os
import sys
import time
from pathlib import Path

import psycopg2

import database
from service_instance import instance as service_instance
from config import OUTPUT_ROOT_PATH, PROCESS_SLEEP_SECONDS, SR_MAX_HEIGHT, SR_MAX_WIDTH
from libraries.fast_srgan import infer_oi
from object_detection import add_car_and_people_insights
from utils import is_null_empty_or_whitespace
from vehicle_color import vehicle_color_detect
from srFile import SrFile


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
        sr_image_objects = infer_oi.process_super_resolution_images(
            sr_image_objects=sr_image_objects,
            max_width=SR_MAX_WIDTH,
            max_height=SR_MAX_HEIGHT,
        )

        # Process results
        for sr_image_object in sr_image_objects:
            # Label based detection if not detected earlier
            if is_null_empty_or_whitespace(sr_image_object.detection_result):

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

import os
import sys
import time
import traceback
from datetime import date, timedelta
import argparse
import psycopg2

import database
from srFile import SrFile
from utils import process_image_objects
from service_instance import instance as service_instance
from config import PROCESS_SLEEP_SECONDS, OUTPUT_ROOT_PATH, IMAGE_SIMILARITY_THRESHOLD


def create_sr_objects_from_records(records):
    """
    Constructs SrFile objects from database records.

    Args:
        records (list): List of database records.

    Returns:
        list: List of SrFile objects.
    """
    objects = []
    for row in records:
        # Get db row fields
        id = row[0]
        label = row[1]
        cropped_file_name = row[2]

        # Construct paths
        input_image_fqfn = os.path.join(OUTPUT_ROOT_PATH, label, cropped_file_name)

        # Make objects
        objects.append(
            SrFile(id, label, cropped_file_name, input_image_fqfn, None, None, None)
        )

    return objects


def find_and_remove_similar_images(after_date, reverse, threshold):
    """
    Main processing logic:
    1. Retrieve recent images for similarity check
    2. Process similar images
    3. Handle remaining images
    """

    records = database.get_images_for_similarity_check_process_after(after_date, reverse,)
    if not records:
        return

    similarity_objects = create_sr_objects_from_records(records)
    if len(similarity_objects) < 2:
        return

    total_records = len(records)
    try:
        removed_count = process_image_objects(similarity_objects, threshold)
        print(f"{removed_count} of {total_records} records removed!")

    except Exception as e:
        print(e)
        print(traceback.format_exc())


def main_loop(after_date, reverse, threshold, run_once):
    """
    Keeps the program running continuously, executing the main processing function
    and handling database connection errors.
    """
    while True:
        try:
            service_instance.set_instance_status()
            find_and_remove_similar_images(after_date, reverse, threshold)
            
            if run_once:
                break
            
            print("... running")
        except psycopg2.OperationalError as e:
            print(e)
        time.sleep(int(PROCESS_SLEEP_SECONDS))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Find and remove similar images")
        parser.add_argument(
            "-d",
            type=str,
            dest="after_date",
            help="Starting Date YYYY-MM-DD format. Default: 2 days ago",
            default=str(date.today() - timedelta(days=2))
        )
        parser.add_argument(
            "-t",
            type=float,
            dest="threshold",
            default=IMAGE_SIMILARITY_THRESHOLD,
            help="Threshold for similarity.  Default = 0.3",
        )
        parser.add_argument(
            "-r",
            dest="r",
            action="store_true",
            help="Process in reverse chronological order.  ie newest to oldest",
        )
        parser.add_argument(
            "-run_once",
            action="store_true",
            dest="run_once",
            help="Run only once, then exit",
        )


        args = parser.parse_args()
        print(args)

        main_loop(args.after_date, args.r, args.threshold, args.run_once)

    except KeyboardInterrupt:
        print >> sys.stderr, "\nExiting by user request.\n"
        sys.exit(0)

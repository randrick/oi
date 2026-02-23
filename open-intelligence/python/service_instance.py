import os
import sys

import database


class Instance:

    # Class constructor
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def set_instance_status(self):
        # Clean ghost instances
        database.clean_instances()

        # No instance, create new
        if instance.id is None:
            process_name = os.path.basename(sys.argv[0])
            self.id = database.new_instance(process_name)
            self.name = process_name
            print("[INFO] new instance id " + str(instance.id))
        # Keep it updated
        elif instance.id is not None:
            database.update_instance(instance.id)


instance = Instance(None, None)

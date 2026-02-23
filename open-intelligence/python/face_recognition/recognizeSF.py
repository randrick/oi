import os
import pickle
from pathlib import Path
import imutils
import shutil
import traceback
import numpy as np
import cv2

from retinaface import RetinaFace

from utils import load_image, save_image
from config import FILE_NAME_PREFIX, FACES_OUTPUT_PATH, FACES_TRAINING_PATH


cs_thresh = 0.363
l2_thresh = 1.128

models_path = os.getcwd() + "/models/YN-SF/"
# yn_model_fqfn = model_root_path + 'face_detection_yunet_2023mar_int8bq.onnx'
yn_model_fqfn = os.path.join(models_path, "face_detection_yunet_2022mar.onnx")
sf_model_fqfn = os.path.join(models_path, "face_recognition_sface_2021dec.onnx")
faces_db_fqfn = os.path.join(models_path, "/faces_db.pickle")

# load face detector, recognizer, and faces_db
# start = timeit.default_timer()
detector = cv2.FaceDetectorYN.create(yn_model_fqfn, "", (320, 320), 0.9, 0.3, 10)

recognizer = cv2.FaceRecognizerSF.create(sf_model_fqfn, "")
faces_db = pickle.loads(open("./models/YN-SF/faces_db.pickle", "rb").read())
# end = timeit.default_timer()


def recognize(image_path, output_file_name=None):
    # Output field
    detection_name_and_probability = None
    try:
        if not (os.path.exists(image_path) and os.path.getsize(image_path) > 0):
            return None

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions

        image = load_image(image_path)
        _, w = image.shape[:2]
        # image = imutils.resize(image, width=1280)

        # Calculate aspect ratio
        # r = 800 / image.shape[1]
        # dim = (800, int(image.shape[0] * r))
        # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        try:
            faces = detect_faces(detector, image, image_path)
        except Exception as e:
            print(e)

        # if faces[1] is not None:
        # for idx, face in enumerate(faces[1]):
        if faces is not None:
            for idx, face in enumerate(faces):
                tup = best_face_match(image, recognizer, faces_db, face)
                if tup is not None:
                    (name, conf) = tup
                    detection_name_and_probability = (
                        " {}: {:.2f}%".format(name, conf) + " [IF]"
                    )  # Added 'if' to the end to detect
                    # that this is from insightface
                    coords = face[:-1].astype(np.int32)
                    (startX, startY, endX, endY) = (
                        coords[0],
                        coords[1],
                        coords[0] + coords[2],
                        coords[1] + coords[3],
                    )

                    y = startY - 10 if startY - 10 > 0 else startY
                    cv2.rectangle(
                        image, (startX, startY), (endX, endY), (0, 255, 255), 2
                    )
                    cv2.putText(
                        image,
                        detection_name_and_probability,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        2,
                    )
                    image_crop = image[startY:endY, startX:endX]
                    (fH, fW) = image_crop.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        # print('[INFO-RWA] - too small skipping ' +
                        #       str(fW) + 'x' + str(fH))
                        continue

                    # Write small face images
                    if output_file_name is not None:
                        Path(FACES_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

                        save_image(
                            os.path.join(
                                FACES_OUTPUT_PATH, FILE_NAME_PREFIX + output_file_name
                            ),
                            image_crop,
                        )
                    else:
                        # Copy original image to faces dataset for later training,
                        # also create image having bounding box for front end ui validation
                        Path(FACES_TRAINING_PATH).mkdir(parents=True, exist_ok=True)

                        shutil.copy(image_path, FACES_TRAINING_PATH + output_file_name)
                        visualize(image, faces)
                        image = imutils.resize(image, width=w)

                        save_image(
                            os.path.join(
                                FACES_TRAINING_PATH, "RECT_" + output_file_name
                            ),
                            image,
                        )
    except Exception as e:
        print(traceback.format_exc())
        print(e)

    # Return result
    return detection_name_and_probability


# Customized to suit for insight face
def recognize_for_insight_face(
    input_image, input_faces, recognizer, faces_db, image_path
):
    # Output field
    detection_name_and_probability = ""

    # if input_faces[1] is not None:
    #     for idx, face in enumerate(input_faces[1]):
    if input_faces is not None:
        for idx, face in enumerate(input_faces):
            tup = best_face_match(input_image, recognizer, faces_db, face)
            if tup is not None:
                (name, conf) = tup
                detection_name_and_probability += (
                    " {}: {:.2f}%".format(name, conf) + " [IF]"
                )  # Added 'if' to the end to detect

    return detection_name_and_probability


def best_face_match(input_image, recognizer, faces_db, face1):
    # (file, name, image, faces)
    (matches, names) = initialize(faces_db)
    for tup in faces_db:
        (file, name, image2, faces2) = tup
        face1_align = recognizer.alignCrop(input_image, face1)
        face2_align = recognizer.alignCrop(image2, faces2[1][0])
        # Extract features
        face1_feature = recognizer.feature(face1_align)
        face2_feature = recognizer.feature(face2_align)

        cosine_score = recognizer.match(
            face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_COSINE
        )
        l2_score = recognizer.match(
            face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_NORM_L2
        )
        if cosine_score >= cs_thresh and l2_score < l2_thresh:
            # break
            matches[name] += 1
            # only checking one face per image right now
            # break

    max = 0
    match = None
    for name in names:
        val = matches[name] / names[name] * 100
        if val > max:
            match = name
            max = val
    if max > 30:
        return (match, max)
    return None


def initialize(faces_db):
    matches = {}
    names = {}
    for tup in faces_db:
        (file, name, image2, faces2) = tup
        if name not in names:
            names[name] = 1
            matches[name] = 0
        else:
            names[name] += 1

    return (matches, names)


def visualize(image, faces, thickness=1):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
            #     idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(
                image,
                (coords[0], coords[1]),
                (coords[0] + coords[2], coords[1] + coords[3]),
                (0, 255, 0),
                thickness,
            )
            cv2.circle(image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)


def detect_faces(detector: cv2.FaceDetectorYN, image, image_path):

    retina_faces = RetinaFace.detect_faces(image_path)
    if retina_faces:
        return map_result_to_cv2(retina_faces)
    else:
        return None


def map_result_to_cv2(results):

    mapped = []
    for val in results.values():
        record = []
        for v in val["facial_area"]:
            record.append(v)
        for v in val["landmarks"]["right_eye"]:
            record.append(v)
        for v in val["landmarks"]["left_eye"]:
            record.append(v)
        for v in val["landmarks"]["nose"]:
            record.append(v)
        for v in val["landmarks"]["mouth_right"]:
            record.append(v)
        for v in val["landmarks"]["mouth_left"]:
            record.append(v)
        record.append(val["score"])
        # print(len(record))
        mapped.append(np.array(record))
    a = np.array(mapped)
    # print(a.shape)
    # print(mapped)
    return mapped


# faces	detection results stored in a 2D cv::Mat of shape [num_faces, 15]

#     0-1: x, y of bbox top left corner
#     2-3: width, height of bbox
#     4-5: x, y of right eye (blue point in the example image)
#     6-7: x, y of left eye (red point in the example image)
#     8-9: x, y of nose tip (green point in the example image)
#     10-11: x, y of right corner of mouth (pink point in the example image)
#     12-13: x, y of left corner of mouth (yellow point in the example image)
#     14: face score

# {
#     "face_1": {
#         "score": 0.9993440508842468,
#         "facial_area": [155, 81, 434, 443],
#         "landmarks": {
#           "right_eye": [257.82974, 209.64787],
#           "left_eye": [374.93427, 251.78687],
#           "nose": [303.4773, 299.91144],
#           "mouth_right": [228.37329, 338.73193],
#           "mouth_left": [320.21982, 374.58798]
#         }
#   }
# }

# x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm,
# where x1, y1, w, h are the top-left coordinates, width and height of the face bounding box, {x, y}_{re, le, nt, rcm, lcm}
# stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively.

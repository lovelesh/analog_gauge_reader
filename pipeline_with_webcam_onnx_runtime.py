import argparse
import os
import logging
import time
import json

import cv2
import numpy as np
from PIL import Image

from plots import RUN_PATH, Plotter
from gauge_detection.detection_inference import detection_gauge_face, find_center_bbox
from key_point_detection.key_point_inference import KeyPointInference, detect_key_points
from geometry.ellipse import fit_ellipse, cart_to_pol, get_line_ellipse_point, \
    get_point_from_angle, get_polar_angle, get_theta_middle, get_ellipse_error
from angle_reading_fit.angle_converter import AngleConverter
from angle_reading_fit.line_fit import line_fit, line_fit_ransac
from segmentation.segmenation_inference import get_start_end_line, segment_gauge_needle, \
    get_fitted_line, cut_off_line
# pylint: disable=no-name-in-module
# pylint: disable=no-member
from evaluation import constants
import metadata

OCR_THRESHOLD = 0.7
RESOLUTION = (
    448, 448
)  # make sure both dimensions are multiples of 14 for keypoint detection

# Several flags to set or unset for pipeline
WRAP_AROUND_FIX = True
RANSAC = False

WARP_OCR = True

# if random_rotations true then random rotations.
RANDOM_ROTATIONS = False
ZERO_POINT_ROTATION = True

OCR_ROTATION = RANDOM_ROTATIONS or ZERO_POINT_ROTATION

WINDOW_NAME = "Gauge Reading"


def crop_image(img, box, flag=False, two_dimensional=False):
    """
    crop image
    :param img: orignal image
    :param box: in the xyxy format
    :return: cropped image
    """
    img = np.copy(img)
    if two_dimensional:
        cropped_img = img[box[1]:box[3],
                          box[0]:box[2]]  # image has format [y, x]
    else:
        cropped_img = img[box[1]:box[3],
                          box[0]:box[2], :]  # image has format [y, x, rgb]  

    height = int(box[3] - box[1])
    width = int(box[2] - box[0])

    # want to preserve aspect ratio but make image square, so do padding
    if height > width:
        delta = height - width
        left, right = delta // 2, delta - (delta // 2)
        top = bottom = 0
    else:
        delta = width - height
        top, bottom = delta // 2, delta - (delta // 2)
        left = right = 0

    pad_color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(cropped_img,
                                 top,
                                 bottom,
                                 left,
                                 right,
                                 cv2.BORDER_CONSTANT,
                                 value=pad_color)

    if flag:
        return new_img, (top, bottom, left, right)
    return new_img


def move_point_resize(point, original_resolution, resized_resolution):
    new_point_x = point[0] * resized_resolution[0] / original_resolution[0]
    new_point_y = point[1] * resized_resolution[1] / original_resolution[1]
    return new_point_x, new_point_y


# here assume that both resolutions are squared
def rescale_ellipse_resize(ellipse_params, original_resolution,
                           resized_resolution):
    x0, y0, ap, bp, phi = ellipse_params

    # move ellipse center
    x0_new, y0_new = move_point_resize((x0, y0), original_resolution,
                                       resized_resolution)

    # rescale axis
    scaling_factor = resized_resolution[0] / original_resolution[0]
    ap_x_new = scaling_factor * ap
    bp_x_new = scaling_factor * bp

    return x0_new, y0_new, ap_x_new, bp_x_new, phi


def process_image(image, detection_model_path, key_point_model_path,
                  segmentation_model_path, run_path, debug, eval_mode, image_is_raw=False):

    result = []
    errors = {}
    result_full = {}

    if not image_is_raw:
        logging.info("Start processing image at path %s", image)
        image = Image.open(image).convert("RGB")
        image = np.asarray(image)
    else:
        logging.info("Start processing image")
        image = np.asarray(image)
        # add a path tweak in the base path to save all the gauge images
        run_parent_path = run_path

    
    plotter = Plotter(run_parent_path, image)

    if eval_mode:
        result_full[constants.IMG_SIZE_KEY] = {
            'width': image.shape[1],
            'height': image.shape[0]
        }

    if debug:
        plotter.save_img()

    # ------------------Gauge detection-------------------------
    if debug:
        print("-------------------")
        print("Gauge Detection")

    logging.info("Start Gauge Detection")

    all_boxes = detection_gauge_face(image, detection_model_path, conf=0.25, optimized=False)

    if debug:
        plotter.plot_bounding_box_img(all_boxes)


    for index, box in enumerate(all_boxes):
        if debug:
            print(f"index: {index}")
        
        try:
            # add an index subfolder to the parent path to save individual debug / eval images
            run_path = os.path.join(run_parent_path, f"{index}")
            
            if debug:
                print(f"run_path: {run_path}")
            
            plotter.set_run_path(run_path)
            # crop image to only gauge face
            cropped_img = crop_image(image, box)
            # result.append({'reading': 25.89, 'unit' : "psi"}) # only for testing
            
            # resize
            cropped_resized_img = cv2.resize(cropped_img,
                                            dsize=RESOLUTION,
                                            interpolation=cv2.INTER_CUBIC)

            if eval_mode:
                result_full[constants.GAUGE_DET_KEY] = {
                    'x': box[0].item(),
                    'y': box[1].item(),
                    'width': box[2].item() - box[0].item(),
                    'height': box[3].item() - box[1].item(),
                }

            if debug:
                plotter.set_image(cropped_resized_img)
                plotter.plot_image('cropped')

            logging.info("Finish Gauge Detection")

            # ------------------Key Point Detection-------------------------

            if debug:
                print("-------------------")
                print("Key Point Detection")

            logging.info("Start key point detection")

            key_point_inferencer = KeyPointInference(key_point_model_path, optimized=True)
            # print(f"type: {type(cropped_resized_img)}")
            heatmaps = key_point_inferencer.predict_heatmaps(cropped_resized_img, optimized=True)
            key_point_list = detect_key_points(heatmaps)

            key_points = key_point_list[1]
            start_point = key_point_list[0]
            end_point = key_point_list[2]

            if eval_mode:
                if start_point.shape == (1, 2):
                    result_full[constants.KEYPOINT_START_KEY] = {
                        'x': start_point[0][0],
                        'y': start_point[0][1]
                    }
                else:
                    result_full[constants.KEYPOINT_START_KEY] = constants.FAILED
                if end_point.shape == (1, 2):
                    result_full[constants.KEYPOINT_END_KEY] = {
                        'x': end_point[0][0],
                        'y': end_point[0][1]
                    }
                else:
                    result_full[constants.KEYPOINT_END_KEY] = constants.FAILED
                result_full[constants.KEYPOINT_NOTCH_KEY] = []
                for point in key_points:
                    result_full[constants.KEYPOINT_NOTCH_KEY].append({
                        'x': point[0],
                        'y': point[1]
                    })

            if debug:
                plotter.plot_heatmaps(heatmaps)
                plotter.plot_key_points(key_point_list)

            logging.info("Finish key point detection")

            # ------------------Ellipse Fitting-------------------------

            if debug:
                print("-------------------")
                print("Ellipse Fitting")

            logging.info("Start ellipse fitting")

            try:
                coeffs = fit_ellipse(key_points[:, 0], key_points[:, 1])
                ellipse_params = cart_to_pol(coeffs)
            except :
                logging.error("Ellipse parameters not an ellipse.")
                errors[constants.NOT_AN_ELLIPSE_ERROR_KEY] = True
                result.append({
                    constants.ID_KEY: constants.ID_FAILED,
                    constants.READING_KEY: constants.READING_FAILED,
                    constants.MEASURE_UNIT_KEY: constants.FAILED
                })
                result_full[constants.OCR_NUM_KEY] = constants.FAILED
                result_full[constants.NEEDLE_MASK_KEY] = constants.FAILED
                write_files(result, result_full, errors, run_path, eval_mode)
                raise Exception("Ellipse parameters not an ellipse")

            ellipse_error = get_ellipse_error(key_points, ellipse_params)
            errors["Ellipse fit error"] = ellipse_error

            if debug:
                plotter.plot_ellipse(key_points, ellipse_params, 'key_points')

            logging.info("Finish ellipse fitting")

            # calculate zero point

            # Find bottom point to set there the zero for wrap around
            if WRAP_AROUND_FIX and start_point.shape == (1, 2) \
                and end_point.shape == (1, 2):
                theta_start = get_polar_angle(start_point.flatten(), ellipse_params)
                theta_end = get_polar_angle(end_point.flatten(), ellipse_params)
                theta_zero = get_theta_middle(theta_start, theta_end)
            else:
                bottom_middle = np.array((RESOLUTION[0] / 2, RESOLUTION[1]))
                theta_zero = get_polar_angle(bottom_middle, ellipse_params)

            zero_point = get_point_from_angle(theta_zero, ellipse_params)
            if debug:
                plotter.plot_zero_point_ellipse(np.array(zero_point),
                                                np.vstack((start_point, end_point)),
                                                ellipse_params)

            # ------------------Segmentation-------------------------
            
            if debug:
                print("-------------------")
                print("Segmentation")

            logging.info("Start segmentation")

            try:
                needle_mask_x, needle_mask_y = segment_gauge_needle(
                    cropped_resized_img, segmentation_model_path)
            except AttributeError:
                logging.error("Segmentation failed, no needle found")
                errors[constants.SEGMENTATION_FAILED_KEY] = True
                result.append({
                    constants.ID_KEY: constants.ID_FAILED,
                    constants.READING_KEY: constants.READING_FAILED,
                    constants.MEASURE_UNIT_KEY: constants.FAILED
                })
                result_full[constants.NEEDLE_MASK_KEY] = constants.FAILED
                write_files(result, result_full, errors, run_path, eval_mode)
                raise Exception("Segmentation failed, no needle found")

            if eval_mode:
                result_full[constants.NEEDLE_MASK_KEY] = {
                    'x': needle_mask_x.tolist(),
                    'y': needle_mask_y.tolist()
                }

            needle_line_coeffs, needle_error = get_fitted_line(needle_mask_x,
                                                            needle_mask_y)
            needle_line_start_x, needle_line_end_x = get_start_end_line(needle_mask_x)
            needle_line_start_y, needle_line_end_y = get_start_end_line(needle_mask_y)

            needle_line_start_x, needle_line_end_x = cut_off_line(
                [needle_line_start_x, needle_line_end_x], needle_line_start_y,
                needle_line_end_y, needle_line_coeffs)

            errors["Needle line residual variance"] = needle_error

            if debug:
                plotter.plot_segmented_line(needle_mask_x, needle_mask_y,
                                            (needle_line_start_x, needle_line_end_x),
                                            needle_line_coeffs)

            logging.info("Finish segmentation")

            # ------------------Project Needle to ellipse-------------------------

            point_needle_ellipse = get_line_ellipse_point(
                needle_line_coeffs, (needle_line_start_x, needle_line_end_x),
                ellipse_params)

            if point_needle_ellipse.shape[0] == 0:
                print("Needle line and ellipse do not intersect!")
                logging.error("Needle line and ellipse do not intersect!")
                errors[constants.OCR_NONE_DETECTED_KEY] = True
                result.append({
                    constants.ID_KEY: constants.ID_FAILED,
                    constants.READING_KEY: constants.READING_FAILED,
                    constants.MEASURE_UNIT_KEY: constants.FAILED
                })
                write_files(result, result_full, errors, run_path, eval_mode)
                raise Exception("Needle line and ellipse do not intersect")

            if debug:
                plotter.plot_ellipse(point_needle_ellipse.reshape(1, 2),
                                    ellipse_params, 'needle_point')
            
            # ------------------Fit line to angles and get reading of needle-------------------------

            # Find angle of needle ellipse point
            needle_angle = get_polar_angle(point_needle_ellipse, ellipse_params)

            angle_converter = AngleConverter(theta_zero)

            try:
                # Find the gauge index from the bounding box
                gauge_index = metadata.get_gauge_details(box)  

                # Extract units from the metadata
                unit = metadata.meter_config[gauge_index]['unit']

                # Extract id from metadata
                id = metadata.meter_config[gauge_index]['id']

                if debug:
                    print(f"Gauge Index: {gauge_index}")

                # make list of start and end angles along with values
                angle_number_list = [(angle_converter.convert_angle(theta_start), metadata.meter_config[gauge_index]['start']), 
                                        (angle_converter.convert_angle(theta_end), metadata.meter_config[gauge_index]['end'])]
                
                if debug:
                    print(f"Angle Number List: {angle_number_list}")
                
                angle_number_arr = np.array(angle_number_list)

                if RANSAC:
                    reading_line_coeff, inlier_mask, outlier_mask = line_fit_ransac(
                        angle_number_arr[:, 0], angle_number_arr[:, 1])
                else:
                    reading_line_coeff = line_fit(angle_number_arr[:, 0],
                                                angle_number_arr[:, 1])

                reading_line = np.poly1d(reading_line_coeff)
                reading_line_res = np.sum(
                    abs(
                        np.polyval(reading_line_coeff, angle_number_arr[:, 0]) -
                        angle_number_arr[:, 0]))
                reading_line_mean_err = reading_line_res / len(angle_number_arr)
                errors["Mean residual on fitted angle line"] = reading_line_mean_err

                needle_angle_conv = angle_converter.convert_angle(needle_angle)

                reading = reading_line(needle_angle_conv)

                result.append({
                    constants.ID_KEY: id,
                    constants.READING_KEY: reading,
                    constants.MEASURE_UNIT_KEY: unit
                })

                if debug:
                    if RANSAC:
                        plotter.plot_linear_fit_ransac(angle_number_arr,
                                                    (needle_angle_conv, reading),
                                                    reading_line, inlier_mask,
                                                    outlier_mask)
                    else:
                        plotter.plot_linear_fit(angle_number_arr,
                                                (needle_angle_conv, reading), reading_line)

                    print(f"Final reading is: {id} {reading} {unit}")
                    plotter.plot_final_reading_ellipse([], point_needle_ellipse,
                                                    round(reading, 1), ellipse_params)

                # ------------------Write result to file-------------------------
                if debug:
                    write_files(result, result_full, errors, run_path, eval_mode)
            
            except Exception as err:
                err_msg = f"Unexpected {err=}, {type(err)=}"
                print(err_msg)
                result.append({
                    constants.ID_KEY: constants.ID_FAILED,
                    constants.READING_KEY: constants.READING_FAILED,
                    constants.MEASURE_UNIT_KEY: constants.FAILED
                })

        except Exception as err:
                err_msg = f"Unexpected {err=}, {type(err)=}"
                print(err_msg)
                # logging.error(err_msg)
                pass

    return result, all_boxes

def write_files(result, result_full, errors, run_path, eval_mode):
    result_path = os.path.join(run_path, constants.RESULT_FILE_NAME)
    write_json_file(result_path, result)

    error_path = os.path.join(run_path, constants.ERROR_FILE_NAME)
    write_json_file(error_path, errors)

    if eval_mode:
        result_full_path = os.path.join(run_path,
                                        constants.RESULT_FULL_FILE_NAME)
        write_json_file(result_full_path, result_full)

def write_json_file(filename, dictionary):
    file_json = json.dumps(dictionary, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(file_json)
    outfile.close()

def read_json_file(filename):
    with open(filename, "r") as infile:
        dictionary = json.load(infile)
    infile.close()
    return dictionary

def capture_xy(action, x, y, flags, *userdata):

    if action == cv2.EVENT_LBUTTONDBLCLK:
        metadata.save_metadata(x, y)         

def main():
    args = read_args()

    input_path = args.input
    detection_model = args.detection_model
    key_point_model = args.key_point_model
    segmentation_model = args.segmentation_model
    base_path = args.base_path
    metadata_path = args.metadata

    time_str = time.strftime("%Y%m%d%H%M%S")
    base_path = os.path.join(base_path, RUN_PATH + '_' + time_str)
    os.makedirs(base_path)

    args_dict = vars(args)
    file_path = os.path.join(base_path, "arguments.json")
    write_json_file(file_path, args_dict)

    log_path = os.path.join(base_path, "run.log")

    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    try:
        metadata.meter_config = read_json_file(metadata_path)
        if args.debug:
            print(f"Read METER Data: {metadata.meter_config}")
    except:
        print("No metadata file found")
        pass

    if input_path.isdigit():
        index = int(input_path)
        cap = cv2.VideoCapture(index)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        time.sleep(1)

        image_base_name = "webcam"
        
        i = 0
        # Break the loop if q is pressed
        if args.debug:
            wait_time = 0
        else:
            wait_time = 20
        # Loop through Video Frames
        while cap.isOpened():

            start = time.time()
            # Read a frame from video
            success, frame = cap.read()
            # image = np.asarray(frame)
            # print(image)

            if not success:
                break
            # Process the image with the pipeline
            print("processing the image")
            if args.debug:
                image_name = f"{image_base_name}_{i}"
            else:
                image_name = image_base_name
            run_path = os.path.join(base_path, image_name)

            all_boxes = []
            gauge_readings = []

            frame = cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            print(f"original image size: {frame.shape}")
            try:
                gauge_readings, all_boxes = process_image(frame,
                                            detection_model,
                                            key_point_model,
                                            segmentation_model,
                                            run_path,
                                            debug=args.debug,
                                            eval_mode=args.eval,
                                            image_is_raw=True)
                # print(f"All boxes: {all_boxes}")
                print(f"Gauge Readings: {gauge_readings}")
                # box, all_boxes, results = detection_gauge_face(image, detection_model)
                
            except Exception as err:
                err_msg = f"Unexpected {err=}, {type(err)=}"
                print(err_msg)
                logging.error(err_msg)

            
            finally:
                # Set format for the overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                box_color = (0, 255, 0)
                text_color = (0, 0, 255)
                box_thickness = 2
                text_thickness = 2

                # Overlay text on captured image
                for (box, gauge_reading) in zip(all_boxes, gauge_readings):
                    cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                    (int(box[2]), int(box[3])), box_color, box_thickness)
                    cv2.putText(frame, f"#{gauge_reading[constants.ID_KEY]} {gauge_reading[constants.READING_KEY]:.2f} {gauge_reading[constants.MEASURE_UNIT_KEY]}",
                                (int(box[0]), int(box[1]) + 25), font, font_scale, text_color, text_thickness)
                
                end = time.time()
                inference_time = np.round(end - start, 2)
                print(f"Inference time: {inference_time} seconds")
            
                # Set callback for mouse events
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(WINDOW_NAME, capture_xy)
                # Display the frame
                cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(wait_time)  & 0xFF
            if key == ord('r'):
                metadata.meter_config.clear()
                if args.debug:
                    print("Metadata resetted")
                    print(metadata.meter_config)

            elif key == ord('d'):
                if args.debug:
                    print(f"Current METER Data: {metadata.meter_config}")
            
            elif key == ord('s'):
                write_json_file(metadata_path, metadata.meter_config)
                if args.debug:
                    print(f"Saving METER Data: {metadata.meter_config}")
                
            elif key == ord('o'):
                metadata.meter_config = read_json_file(metadata_path)
                if args.debug:
                    print(f"Read METER Data: {metadata.meter_config}")

            elif key == ord('c'):
                wait_time = 0

            elif key == ord('v'):
                wait_time = 20
            
            elif key == ord('q'):
                break
            print("--------------------Inference complete--------------------------")
            i += 1
            # time.sleep(2)
        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Error: Enter correct Camera index")
        logging.error("Invalid camera index")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help="Enter the index of camera input")
    parser.add_argument('--detection_model',
                        type=str,
                        required=False,
                        default="models/gauge_detection_model_custom_trained.onnx",
                        help="Path to detection model")
    parser.add_argument('--key_point_model',
                        type=str,
                        required=False,
                        default="models/key_point_model.onnx",
                        help="Path to key point model")
    parser.add_argument('--segmentation_model',
                        type=str,
                        required=False,
                        default="models/segmentation_model_custom_trained.onnx",
                        help="Path to segmentation model")
    parser.add_argument('--base_path',
                        type=str,
                        required=False,
                        default="results",
                        help="Path where run folder is stored")
    parser.add_argument('--metadata',
                        type=str,
                        required=False,
                        default="metadata.json",
                        help="Path to metadata file")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    main()

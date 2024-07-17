import os
import time
import datetime
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, UnidentifiedImageError
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tflite_support.task import core, processor, vision

def get_detections(detection_result):
    """get moth detections from the detection result and convert to list of dicts

    Args:
        detection_result (tensorflow_lite_support.python.task.processor.proto.detections_pb2.DetectionResult): The object detection results from the model

    Returns:
        list: a list of dictionaries with the detection information
    """
    detections_list = []

    for counter, detection in enumerate(detection_result.detections, start=1):
        bounding_box = detection.bounding_box
        origin_x, origin_y, width, height = bounding_box.origin_x, bounding_box.origin_y, bounding_box.width, bounding_box.height

        category_info = detection.categories[0]
        category_dict = {
            'index': category_info.index,
            'score': category_info.score,
            'display_name': category_info.display_name,
            'category_name': category_info.category_name,
        }

        detection_dict = {
            'counter': counter,
            'bounding_box': {
                'origin_x': origin_x,
                'origin_y': origin_y,
                'width': width,
                'height': height,
            },
            'categories': [category_dict],
        }

        detections_list.append(detection_dict)

    return detections_list

def species_inference(image, interpreter):
    """Perform species classification on an image

    Args:
        image (numpy.ndarray): A numpy array representing the image
        interpreter (tensorflow.lite.python.interpreter.Interpreter): The tflite interpreter

    Returns:
        prediction_tf (numpy.int64): The index of the predicted class
        confidence (numpy.float64): The confidence of the prediction
        time (str): The time taken for inference (microseconds)
    """

    a = datetime.datetime.now()
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    input_data = np.transpose(input_data, (0, 3, 1, 2))
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    outputs_tf = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    prediction_tf = np.squeeze(outputs_tf)
    confidence = np.exp(prediction_tf) / np.sum(np.exp(prediction_tf))
    prediction_tf = prediction_tf.argsort()[::-1][0]
    c = datetime.datetime.now()
    c = c - a

    return prediction_tf, max(confidence) * 100, str(c.microseconds)

def handle_file_creation(event):
    """handle file creation events: perform moth detection and species
    classification. Save results to csv and annotated image.

    Args:
        event (watchdog.events.FileCreatedEvent): File creation event
    """

    # When image is added, load
    if event.is_directory:
        return
    print(f"Performing inferences on: {event.src_path}")
    image_path = event.src_path
    max_loops = 20  # Wait for max 2 seconds for image to be written to disk
    loop_counter = 0
    while True:
        try:
            print("Waiting for image to be written to disk...")
            time.sleep(0.1)  # Give the image time to be written to disk
            image = np.asarray(Image.open(image_path))
            break
        except UnidentifiedImageError:
            if loop_counter > max_loops:
                print("Timeout reached. Unable to open image.")
                return
    image = np.asarray(Image.open(image_path))
    print("Opened image...")
    annot_image = image.copy()
    annotated_image_path = os.path.join('/home/pi/Desktop/model_data_bookworm/annotated_images/',
                                        "most_recent_annotated_image.jpg")

    # Perform moth detecion
    input_tensor = vision.TensorImage.create_from_array(image)
    a = datetime.datetime.now()
    detection_result = detector.detect(input_tensor)
    detections_list = get_detections(detection_result)
    b = datetime.datetime.now()
    c = b - a

    print(f"{len(detections_list)} detections")
    for detection in detections_list:
        bounding_box = detection['bounding_box']
        origin_x, origin_y, width, height = bounding_box['origin_x'], bounding_box['origin_y'], bounding_box['width'], bounding_box['height']

        # Crop the image to the bounding box
        cropped_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
        category_name = detection['categories'][0]['category_name']
        insect_score = detection['categories'][0]['score']
        resized_image = Image.fromarray(cropped_image).convert("RGB").resize((300, 300))
        img = np.array(resized_image) / 255.0
        img = (img - 0.5) / 0.5

        # Perform species classification
        species_inf, conf, inf_time = species_inference(img, interpreter)

        # If insect at image boundary move the label
        im_width, im_height = resized_image.size
        ymax = origin_y - 10 if origin_y - 10 >= 5 else origin_y + height + 30

        # Add bounding box annotation to the image
        bbox_color = (46, 139, 87) if category_name == 'moth' else (238, 75, 43)
        ann_label = f"{species_names[species_inf]}, {conf:.2f}"
        cv2.rectangle(annot_image,
                      (origin_x, origin_y),
                      (origin_x + width, origin_y + height),
                      bbox_color, 4)
        cv2.putText(annot_image, text=ann_label,
                    org=(origin_x, ymax),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.2, color=bbox_color, thickness=4)

        # Save inference results to csv
        df = pd.DataFrame({
            'image_path': [image_path],
            'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'moth_class': [category_name],
            'insect_score': [insect_score],
            'detection_time': [str(c.microseconds)],
            'bounding_box': ['; '.join(map(str, bounding_box.values()))],
            # 'annot_path': [],  # [annotated_image_path],
            'species_inference_time': [inf_time],
            # 'truth': [' '.join(image_path.split('/')[-1].split('_')[0:2])],
            'pred': [species_names[species_inf]],
            'confidence': [conf],
            'model': [region]
        })
        # df['correct'] = np.where(df['pred'] == df['truth'], 1, 0)
        df.to_csv(f'{results_path}/predictions.csv', index=False, mode='a', header=False)

        # Check if the json file exists, if not create it and populate it with an 
        if os.path.exists(f'{results_path}/predictions.json'):

            # Load in the existing json
            with open(f'{results_path}/predictions.json', 'r') as file:
                data = json.load(file)

        else:

            # Create a new json file
            data = {}

        try:

            # Add the new record to the json (append)
            json_df = pd.DataFrame.from_dict(data, orient='index')
            json_df = pd.concat([json_df, df])

            records = json_df.to_dict(orient='records')
            master_dict = {}
            for index, record in enumerate(records):
                master_dict[f'record_{index}'] = record

        except Exception as e:

            print(f"Error: {e}")
            master_dict = {}

        # Write the master dictionary to a JSON file
        output_file_path = f'{results_path}/predictions.json'
        with open(output_file_path, 'w') as outfile:
            json.dump(master_dict, outfile, indent=4)

    cv2.imwrite(annotated_image_path, cv2.cvtColor(annot_image, cv2.COLOR_BGR2RGB))

def monitor_directory(path):
    """monitor a directory for file creation events

    Args:
        path (str): the path to the directory to monitor
    """
    event_handler = FileSystemEventHandler()
    event_handler.on_created = handle_file_creation
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Configuration
    model_path = './models/gbif_model_metadata.tflite'
    enable_edgetpu = False
    num_threads = 1
    region = 'thailand'
    directory_to_watch = "/media/pi/PiImages"
    results_path = f"/media/pi/PiImages/results_{datetime.datetime.now().isoformat().replace(':', '_')}"

    # Create results directory
    os.makedirs(results_path, exist_ok=True)

    # Create a link to the file in directory_to_watch called predictions.json
    os.system(f'ln -sf {results_path} ./results')

    # Moth Detection Setup
    base_options = core.BaseOptions(file_name=model_path,
                                    use_coral=enable_edgetpu,
                                    num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=20,
                                                   score_threshold=0.1)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Species Classification Setup
    interpreter = tf.lite.Interpreter(model_path=f"./models/resnet_{region}.tflite")
    interpreter.allocate_tensors()
    species_names = json.load(open(f'./models/01_{region}_data_numeric_labels.json', 'r'))['species_list']

    monitor_directory(directory_to_watch)

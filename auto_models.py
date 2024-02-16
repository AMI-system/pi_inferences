import os
import time
import datetime
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tflite_support.task import core, processor, vision

# Function to get detections from the detection result
def get_detections(detection_result):
    detections_list = []

    for counter, detection in enumerate(detection_result.detections, start=1):
        bounding_box = detection.bounding_box
        origin_x, origin_y, width, height = bounding_box.origin_x, bounding_box.origin_y, bounding_box.width, bounding_box.height

        category_info = detection.categories[0]  # Assuming one category per detection
        category_dict = {
            'index': category_info.index,
            'score': category_info.score,
            'display_name': category_info.display_name,
            'category_name': category_info.category_name,
        }

        detection_dict = {
            'counter': counter,
            'bounding_box': {
                'origin_x': origin_x, 'origin_y': origin_y,
                'width': width, 'height': height,
            },
            'categories': [category_dict],
        }

        detections_list.append(detection_dict)

    return detections_list

# Function for TensorFlow Lite inference
def tflite_inference(image, interpreter):
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

# Function to handle file creation events
def handle_file_creation(event):
    if event.is_directory:
        return
    print(f"Performing inferences on: {event.src_path}")
    time.sleep(0.5)  # Give the image time to be written to disk
    image_path = event.src_path

    image = np.asarray(Image.open(image_path))
    annot_image = image.copy()
    annotated_image_path = os.path.join('/home/pi/Desktop/model_data_bookworm/annotated_images/', 
                                        os.path.basename(image_path))

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

        cropped_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
        category_name = detection['categories'][0]['category_name']

        resized_image = Image.fromarray(cropped_image).convert("RGB").resize((300, 300))
        img = np.array(resized_image) / 255.0
        img = (img - 0.5) / 0.5

        tflite_inf, conf, inf_time = tflite_inference(img, interpreter)

        im_width, im_height = resized_image.size
        ymax = origin_y - 10 if origin_y - 10 >= 0 else origin_y + height + 20

        bbox_color = (46, 139, 87) if category_name == 'moth' else (238, 75, 43)
        cv2.rectangle(annot_image, 
                      (origin_x, origin_y), 
                      (origin_x + width, origin_y + height), 
                      bbox_color, 4)
        cv2.putText(annot_image, text=species_names[tflite_inf], 
                    org=(origin_x, ymax), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5, color=bbox_color, thickness=4)

        df = pd.DataFrame({
            'image_path': [image_path], 
            'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'moth_class': [category_name],
            'detection_time': [str(c.microseconds)],
            'bounding_box': ['; '.join(map(str, bounding_box))], 
            'annot_path': [annotated_image_path],
            'species_inference_time': [inf_time], 
            'truth': [' '.join(image_path.split('/')[-1].split('_')[0:2])], 
            'pred': [species_names[tflite_inf]], 
            'confidence': [conf],
            'model': [region]
        })
        df['correct'] = np.where(df['pred'] == df['truth'], 1, 0)
        df.to_csv(f'./results/{region}_predictions.csv', index=False, mode='a', header=False)

    cv2.imwrite(annotated_image_path, cv2.cvtColor(annot_image, cv2.COLOR_BGR2RGB))

# Function to monitor a directory for file creation events
def monitor_directory(path):
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
    region = 'uk'
    directory_to_watch = "/home/pi/Desktop/model_data_bookworm/watch_folder"

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

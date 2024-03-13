import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import os
import datetime
import json
import pandas as pd
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""

    img = tf.io.read_file(image_path)

    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""

    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def run_moth_detection(image_path, m_interpreter, threshold=0.5, label=True, width=12):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = m_interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
        )

    a = datetime.datetime.now()
    # Run object detection on the input image
    results = detect_objects(m_interpreter, preprocessed_image, threshold=threshold)
    b = datetime.datetime.now()
    c=b-a

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)

    detections_list = []
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        category_dict = {
            'index': int(obj['class_id']),
            'score': obj['score'],
            'display_name':class_labels[int(obj['class_id'])],
            'category_name': class_labels[int(obj['class_id'])]
        }

        detection_dict = {
            'counter': 1, #counter,
            'bounding_box': {
                'origin_x': xmin,
                'origin_y': ymin,
                'width': xmax - xmin,
                'height': ymax-ymin,
            },
            'categories': [category_dict],
        }

        detections_list.append(detection_dict)
    return detections_list, original_image, str(c.microseconds)

def species_inference(crop_image, species_interpreter):
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
    # input_data = np.expand_dims(image, axis=0).astype(np.float32)
    #
    # input_data = input_data.astype(np.uint8)


    print(crop_image.shape)
    #Resize the cropped image and convert it to FLOAT32
    input_data_resized = cv2.resize(crop_image, (300, 300)).astype(np.float32)

    # Normalize the pixel values to the range [0, 1]
    input_data_resized /= 255.0

    # Expand dimensions to match the expected shape [1, 640, 640, 3]
    input_data = np.expand_dims(input_data_resized, axis=0)
    input_data = np.transpose(input_data, (0, 3, 1, 2))

    # Set the input tensor to the interpreter
    species_interpreter.set_tensor(species_interpreter.get_input_details()[0]['index'],
                                   input_data)



    # Resize input image to match expected shape [1, 640, 640, 3]
    # input_data_resized = cv2.resize(crop_image, (640, 640)).astype(np.uint8)

    # # Expand dimensions to match expected shape [1, 640, 640, 3]
    # input_data = np.expand_dims(input_data_resized, axis=0)

    # species_interpreter.set_tensor(species_interpreter.get_input_details()[0]['index'],
    #                     input_data)
    species_interpreter.invoke()
    outputs_tf = species_interpreter.get_tensor(species_interpreter.get_output_details()[0]['index'])
    prediction_tf = np.squeeze(outputs_tf)
    confidence = np.exp(prediction_tf) / np.sum(np.exp(prediction_tf))
    prediction_tf = prediction_tf.argsort()[::-1][0]
    c = datetime.datetime.now()
    c = c - a

    return prediction_tf, max(confidence) * 100, str(c.microseconds)


def perform_inference(image_path, moth_interpreter, species_interpreter):
    """Performs moth detection and species classification on the input image."""
    # Load input image
    image = np.asarray(Image.open(image_path))
    annot_image = image.copy()

    # Run moth detection
    detections_list, original_image, det_time = run_moth_detection(image_path, moth_interpreter, threshold=moth_threshold)
    original_image_np = original_image.numpy().astype(np.uint8)

    # Process each detected moth
    for detection in detections_list:
        bounding_box = detection['bounding_box']
        origin_x, origin_y, width, height = bounding_box['origin_x'], bounding_box['origin_y'], \
                                            bounding_box['width'], bounding_box['height']

        # Convert bounding box coordinates to pixels
        xmin = int(origin_x)# * original_image_np.shape[1])
        xmax = int((origin_x + width))# * original_image_np.shape[1])
        ymin = int(origin_y)# * original_image_np.shape[0])
        ymax = int((origin_y + height))# * original_image_np.shape[0])

        # slice the image to the bounding box
        cropped_image = original_image_np[ymin:ymax, xmin:xmax]

        # Perform species classification
        species_inf, conf, inf_time = species_inference(cropped_image, species_interpreter)

        # Add bounding box annotation to the image
        bbox_color = (46, 139, 87) if detection['categories'][0]['category_name'] == 'moth' else (238, 75, 43)
        ann_label = f"{species_names[species_inf]}, {conf:.2f}"
        cv2.rectangle(annot_image,
                      (xmin, ymin),
                      (xmax, ymax),
                      bbox_color, 4)
        cv2.putText(annot_image, text=ann_label,
                    org=(xmin, ymin),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.2, color=bbox_color, thickness=4)

        # Save inference results to CSV
        df = pd.DataFrame({
            'image_path': [image_path],
            'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'moth_class': [detection['categories'][0]['category_name']],
            'insect_score': [detection['categories'][0]['score']],
            'detection_time': [det_time],
            'bounding_box': ['; '.join(map(str, [xmin, ymin, xmax, ymax]))],
            'annot_path': [image_path],  # Same as image path for now
            'species_inference_time': [inf_time],
            'truth': [' '.join(os.path.basename(image_path).split('_')[0:2])],
            'pred': [species_names[species_inf]],
            'confidence': [conf],
            'model': [region]
        })
        df['correct'] = np.where(df['pred'] == df['truth'], 1, 0)
        df.to_csv(output_csv_path, index=False, mode='a', header=False)

    annotated_image_path = os.path.join('./annotated_images/', os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, cv2.cvtColor(annot_image, cv2.COLOR_BGR2RGB))
    print('Saved annotated image to: ', annotated_image_path)

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
    time.sleep(0.5)  # Give the image time to be written to disk
    image_path = event.src_path
    #image = np.asarray(Image.open(image_path))

    perform_inference(image_path, moth_interpreter, species_interpreter)


def main():


    # Define the directory to monitor
    directory_to_watch = './watch_folder/'

    monitor_directory(directory_to_watch)

    # Initialize observer and event handler
    # observer = Observer()
    # event_handler = ImageHandler(moth_interpreter, species_interpreter)

    # # Schedule the event handler to watch the directory for new files
    # observer.schedule(event_handler, directory_to_watch, recursive=False)

    # # Start the observer
    # observer.start()
    # print(f"Watching directory: {directory_to_watch}")

    # try:
    #     while True:
    #         # Keep the script running
    #         pass
    # except KeyboardInterrupt:
    #     # Stop the observer if the script is interrupted
    #     observer.stop()

    # # Wait for the observer to join
    # observer.join()



# Define global variables
moth_model_path = './models/gbif_model_metadata.tflite'
region = 'uk'
species_model_path = f"./models/resnet_{region}.tflite"
species_labels = f'./models/01_{region}_data_numeric_labels.json'
output_csv_path = f'./results/{region}_predictions.csv'
class_labels = ['moth', 'nonmoth']
moth_threshold = 0.1

# Load moth detection model
moth_interpreter = tf.lite.Interpreter(model_path=moth_model_path)
moth_interpreter.allocate_tensors()

# Load species classification model
species_interpreter = tf.lite.Interpreter(model_path=species_model_path)
species_interpreter.allocate_tensors()

# Load species names
species_names = json.load(open(species_labels, 'r'))['species_list']


if __name__ == "__main__":
    main()

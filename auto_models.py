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
from concurrent.futures import ThreadPoolExecutor
import threading

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
    try:

        start_time = datetime.datetime.now()

        # When image is added, load
        if event.is_directory:
            return
        print(f"Performing inferences on: {event.src_path} using thread {threading.get_ident()}")
        image_path = event.src_path
        max_loops = 20  # Wait for max 2 seconds for image to be written to disk
        loop_counter = 0
        while True:
            try:
                print(f"Waiting for image {event.src_path} to be written to disk...")
                time.sleep(0.1)  # Give the image time to be written to disk
                image = np.asarray(Image.open(image_path))
                break
            except UnidentifiedImageError:
                if loop_counter > max_loops:
                    print(f"Timeout reached. Unable to open image {event.src_path}.")
                    return
        image = np.asarray(Image.open(image_path))
        print(f"Opened image {event.src_path}...")
        annot_image = image.copy()
        # If target path does not exist, create it
        if not os.path.exists('/media/pi/PiImages/annotated_images'):
            os.makedirs('/media/pi/PiImages/annotated_images')
        annotated_image_path = os.path.join('/media/pi/PiImages/annotated_images',
                                            os.path.basename(image_path))

        # Perform moth detection
        input_tensor = vision.TensorImage.create_from_array(image)
        a = datetime.datetime.now()
        detection_result = detector.detect(input_tensor)
        detections_list = get_detections(detection_result)
        b = datetime.datetime.now()
        c = b - a

        print(f"{len(detections_list)} detections on {event.src_path}")
        idx = 0
        for detection in detections_list:
            idx += 1
            print(f"{event.src_path}: {idx}/{len(detections_list)}")
            bounding_box = detection['bounding_box']
            origin_x, origin_y, width, height = bounding_box['origin_x'], bounding_box['origin_y'], bounding_box['width'], bounding_box['height']

            # Crop the image to the bounding box
            cropped_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
            category_name = detection['categories'][0]['category_name']
            insect_score = detection['categories'][0]['score']
            resized_image = Image.fromarray(cropped_image).convert("RGB").resize((300, 300))
            img = np.array(resized_image) / 255.0
            img = (img - 0.5) / 0.5

            # Ensure each worker uses a different interpreter
            thread_id = threading.get_ident()
            if thread_id not in interpreters:
                # print(f"Creating interpreter for thread {thread_id} and image {event.src_path}")
                interpreter = tf.lite.Interpreter(model_path=f"./models/resnet_{region}.tflite")
                interpreter.allocate_tensors()
                interpreters[thread_id] = interpreter
            else:
                # print(f"Using existing interpreter for thread {thread_id} and image {event.src_path}")
                interpreter = interpreters[thread_id]

            # Perform species classification
            species_inf, conf, inf_time = species_inference(img, interpreter)

            # print(f"Species inference on {event.src_path}: {species_names[species_inf]}, {conf:.2f}")

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

            # print(f"+1 interference on {event.src_path}")

            # Lock for thread-safe file access
            # Create a lock object to ensure that only one thread can access the output files at a time.
            file_lock = threading.Lock()

            # The 'with' statement is used to acquire the lock before entering the block of code.
            # This ensures that the code within the 'with' block is executed by only one thread at a time.
            # When the block is exited, the lock is automatically released, even if an exception occurs.
            with file_lock:
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

                # print(f"+1 inference from {event.src_path} added to output")

        cv2.imwrite(annotated_image_path, cv2.cvtColor(annot_image, cv2.COLOR_BGR2RGB))

        print(f"Done processing {event.src_path} in {datetime.datetime.now() - start_time}")

    except Exception as e:

        print()
        print(f"Error in handle_file_creation: {e}")
        print()

def monitor_directory(path):
    """Monitor a directory for file creation events

    Args:
        path (str): the path to the directory to monitor
    """
    # Create an event handler that will handle file creation events
    event_handler = FileSystemEventHandler()
    
    # Assign a lambda function to handle file creation events by submitting the handle_file_creation function to the executor
    event_handler.on_created = lambda event: executor.submit(handle_file_creation, event)
    
    # Create an observer to monitor the directory
    observer = Observer()
    
    # Schedule the observer to watch the specified path for file creation events
    observer.schedule(event_handler, path, recursive=False)
    
    # Start the observer
    observer.start()

    # Create a ThreadPoolExecutor to handle up to 3 concurrent file creation events
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            # Keep the script running to continuously monitor the directory
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # Stop the observer if the script is interrupted
            observer.stop()
        
        # Wait for the observer to finish
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
    try:
        os.system(f'rm /home/pi/Desktop/model_data_bookworm/results')
    except:
        pass
    os.system(f'ln -sf {results_path} /home/pi/Desktop/model_data_bookworm/results')

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
    # interpreter = tf.lite.Interpreter(model_path=f"./models/resnet_{region}.tflite")
    # interpreter.allocate_tensors()
    interpreters = {}
    species_names = json.load(open(f'./models/01_{region}_data_numeric_labels.json', 'r'))['species_list']

    monitor_directory(directory_to_watch)

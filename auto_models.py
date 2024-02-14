import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# For object detection
import datetime
import numpy as np
import os
from PIL import Image
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import numpy as np

# For species detection
import json
import tensorflow as tf
import pandas as pd

#########################################
# Moth Detection
#########################################

model = './gbif_model_metadata.tflite'
enable_edgetpu = False
num_threads = 1

base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
detection_options = processor.DetectionOptions(max_results=20, score_threshold=0.1)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

def get_detections(detection_result):
    detections_list = []

    for counter, detection in enumerate(detection_result.detections, start=1):
        bounding_box = detection.bounding_box
        origin_x, origin_y, width, height = (
            bounding_box.origin_x, bounding_box.origin_y,
            bounding_box.width, bounding_box.height,
        )

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
        
    return(detections_list)
    
#########################################
# Species Classification
#########################################

region='uk'

# Species Inference
interpreter = tf.lite.Interpreter(model_path=f"./resnet_{region}.tflite")
species_names = json.load(open(f'./01_{region}_data_numeric_labels.json', 'r'))
species_names = species_names['species_list']

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

def tflite_inference(image, interpreter, print_time=False):
    a = datetime.datetime.now()
    
    input_data = np.expand_dims(image, axis=0)
    input_data = input_data.astype(np.float32)
    input_data = np.transpose(input_data, (0, 3, 1, 2))
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs_tf = interpreter.get_tensor(output_details[0]['index'])
    prediction_tf = np.squeeze(outputs_tf)
    confidence = np.exp(prediction_tf) / np.sum(np.exp(prediction_tf))
    prediction_tf = prediction_tf.argsort()[::-1][0]

    b = datetime.datetime.now()
    c = b - a
    if print_time: print(str(c.microseconds) + "\u03bcs")
    return prediction_tf, max(confidence) * 100, str(c.microseconds)

    
######################################
# Run
######################################

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        print(f"Performing inferences on: {event.src_path}")
        image_path = event.src_path
        
        #print(image_dir + image_path)
        image = np.asarray(Image.open(image_path))

		# Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(image)
		
		# Get the inferences 
        a = datetime.datetime.now()
        detection_result = detector.detect(input_tensor)    
        detections_list = get_detections(detection_result)
        b = datetime.datetime.now()
        c = b - a
		
        counter = 1
		
        raw_image_paths = []
        crop_image_paths = []
        moth_class = []
        boxes = []
        detection_time = []
        		
		# Draw bounding boxes on the image
        for detection in detections_list:
            bounding_box = detection['bounding_box']
            origin_x, origin_y, width, height = (bounding_box['origin_x'], bounding_box['origin_y'], bounding_box['width'], bounding_box['height'])

			# Crop the original image
            cropped_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
            category_name = detection['categories'][0]['category_name']
			
			# Save the cropped image
            basepath = os.path.splitext(os.path.basename(image_path))[0]
            
            save_path = os.path.join('/home/pi/Desktop/model_data_bookworm/cropped_common_species/', f'{basepath}_{counter}_{category_name}.jpg')  
            print(save_path)
            Image.fromarray(cropped_image).save(save_path)
			
            counter += 1
            
            class_image = Image.open(save_path).convert("RGB")

            # Resize the image
            resized_image = class_image.resize((300, 300))

            # Convert to NumPy array and normalize
            img = np.array(resized_image) / 255.0
            img = (img - 0.5) / 0.5

            tflite_inf, conf, inf_time = tflite_inference(img, interpreter)

            df = pd.DataFrame({'image_path': [image_path], 
                            'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                            'moth_class': [category_name],
                            'detection_time': [str(c.microseconds)],
                            'bounding_box': ['; '.join(map(str, bounding_box))], 
                            'crop_path': [save_path],
                            'species_inference_time': [inf_time], 
                            'truth': [' '.join(save_path.split('/')[-1].split('_')[0:2])], 
                            'pred': [species_names[tflite_inf]], 
                            'confidence': [conf],
                            'model': [region]
                            })                         
                            
            df['correct'] = np.where(df['pred'] == df['truth'], 1, 0)
            
            # append this df 
            df.to_csv(f'./results/{region}_predictions.csv', index=False, mode='a', header=False)
                    

def monitor_directory(path):
    event_handler = NewFileHandler()
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
    directory_to_watch = "/home/pi/Desktop/model_data_bookworm/watch_folder"
    monitor_directory(directory_to_watch)


       



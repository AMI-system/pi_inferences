import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# For object detection
import cv2
import datetime
import time
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

from PIL import ImageDraw

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
        time.sleep(0.5) # need to give the image time to be written to disk
        image_path = event.src_path
        
        image = np.asarray(Image.open(image_path))
        annot_image = image.copy()
        annotated_image_path = os.path.join('/home/pi/Desktop/model_data_bookworm/annotated_images/', 
                                            os.path.basename(image_path))  

		# Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(image)
		
		# Get the inferences 
        a = datetime.datetime.now()
        detection_result = detector.detect(input_tensor)    
        detections_list = get_detections(detection_result)
        b = datetime.datetime.now()
        c = b - a
		
        counter = 1
        
        print(str(len(detections_list)) + " detections")
		# Draw bounding boxes on the image
        for detection in detections_list:
            bounding_box = detection['bounding_box']
            origin_x, origin_y, width, height = (bounding_box['origin_x'], 
                                                 bounding_box['origin_y'], 
                                                 bounding_box['width'], 
                                                 bounding_box['height'])

			# Crop the original image
            cropped_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
            category_name = detection['categories'][0]['category_name']
            
            
			# If you want to save the cropped image
            #basepath = os.path.splitext(os.path.basename(image_path))[0]
            #save_path = os.path.join('/home/pi/Desktop/model_data_bookworm/cropped_common_species/', f'{basepath}_{counter}_{category_name}.jpg')  
            #print(save_path)
            #Image.fromarray(cropped_image).save(save_path)
			
            counter += 1
            
            #class_image = Image.open(save_path).convert("RGB")

            class_image = Image.fromarray(cropped_image)
            class_image = class_image.convert("RGB")

            # Resize the image
            resized_image = class_image.resize((300, 300))

            # Convert to NumPy array and normalize
            img = np.array(resized_image) / 255.0
            img = (img - 0.5) / 0.5

            tflite_inf, conf, inf_time = tflite_inference(img, interpreter)
            
            
            # Get image dimensions
            im_width, im_height = class_image.size
            ymax = origin_y - 10
            if ymax < 0: ymax = origin_y + height + 20                
            
            # Draw bounding box on the original image          
            if category_name == 'moth':
                cv2.rectangle(annot_image, 
                            (origin_x, origin_y), 
                            (origin_x + width, origin_y + height), 
                            (46, 139, 87), 4)  
                
                cv2.putText(annot_image, 
                            text=species_names[tflite_inf], 
                            org=(origin_x, ymax), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1.5, 
                            color=(46, 139, 87), 
                            thickness=4)
            else:
                cv2.rectangle(annot_image, 
                            (origin_x, origin_y), 
                            (origin_x + width, origin_y + height), 
                            (238, 75, 43), 4)
                
                cv2.putText(annot_image, 
                            text=species_names[tflite_inf], 
                            org=(origin_x, ymax), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1.5, 
                            color=(238, 75, 43), 
                            thickness=4)

            df = pd.DataFrame({'image_path': [image_path], 
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
            
            # append this df 
            df.to_csv(f'./results/{region}_predictions.csv', index=False, mode='a', header=False)
            
        # Save annotated image
        cv2.imwrite(annotated_image_path, cv2.cvtColor(annot_image, cv2.COLOR_BGR2RGB))

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

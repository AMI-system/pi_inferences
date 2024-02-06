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


# list images in path
image_dir = './common_species/'
images = os.listdir(image_dir)
output_directory = './cropped_common_species/'

raw_image_paths = []
crop_image_paths = []
moth_class = []
boxes = []

for image_path in images:
    print(image_dir + image_path)
    image = np.asarray(Image.open(image_dir + image_path))

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(image)
    
    # Get the inferences 
    detection_result = detector.detect(input_tensor)    
    detections_list = get_detections(detection_result)
    
    counter = 1
    
    # Draw bounding boxes on the image
    for detection in detections_list:
        bounding_box = detection['bounding_box']
        origin_x, origin_y, width, height = (
                bounding_box['origin_x'],
                bounding_box['origin_y'],
                bounding_box['width'],
                bounding_box['height'],
            )

        # Crop the original image
        cropped_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
        category_name = detection['categories'][0]['category_name']
        
         # Save the cropped image
        basepath = os.path.basename(image_path)
        save_path = os.path.join(output_directory, f'{basepath}_{counter}_{category_name}.jpg')  
        
        Image.fromarray(cropped_image).save(save_path)

        raw_image_paths = raw_image_paths + [image_path]
        crop_image_paths = crop_image_paths + [save_path]
        moth_class = moth_class + [category_name]
        boxes = boxes + [bounding_box]
        
        counter += 1
        

       

#########################################
# Species Classification
#########################################

print('Now running species classification!!!')

region='singapore'

# Species Inference
interpreter = tf.lite.Interpreter(model_path=f"./resnet_{region}.tflite")
species_names = json.load(open(f'./01_{region}_data_numeric_labels.json', 'r'))
species_names = species_names['species_list']

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# list the files in the common species folder
all_images = crop_image_paths #os.listdir('./cropped_common_species')

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

truth = []
pred = []
confidence = []
time = []

for image_file in all_images:
    image = Image.open(image_file).convert("RGB")  # Convert to RGB mode

    # Resize the image
    resized_image = image.resize((300, 300))

    # Convert to NumPy array and normalize
    img = np.array(resized_image) / 255.0
    img = (img - 0.5) / 0.5

    tflite_inf, conf, inf_time = tflite_inference(img, interpreter)

    truth = truth + [' '.join(image_file.split('/')[-1].split('_')[0:2])]
    pred = pred + [species_names[tflite_inf]]
    confidence = confidence + [conf]
    time = time + [inf_time]

df = pd.DataFrame({'image_path': raw_image_paths, 
                  'moth_class': moth_class,
                  'bounding_box': ['; '.join(map(str, x)) for x in boxes], 
                  'crop_path': crop_image_paths,
                  'species_inference_time':time, 
                  'truth':truth, 
                  'pred':pred, 
                  'confidence':confidence
                  })
                  
                 
df['correct'] = np.where(df['pred'] == df['truth'], 1, 0)

df = df.sort_values('confidence', ascending=False)

df.to_csv(f'./results/{region}_predictions.csv', index=False)


print(df)

import sys
print(sys.path)

# For object detection
import datetime
import numpy as np
import os

from PIL import Image
# ~ import pandas as pd
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import cv2
import numpy as np

# For sepcies detection
import json
import tensorflow as tf
import pandas as pd
# ~ from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

for image_path in images:
    print(image_dir + image_path)
    image = cv2.imread(image_dir + image_path)
    image = np.array(image, dtype=np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = Image.open(image_dir + image_path)

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

        # Save the cropped image
        category_name = detection['categories'][0]['category_name']
        basepath = os.path.basename(image_path)
        save_path = os.path.join(output_directory, f'{basepath}_{counter}_{category_name}.jpg')  
        counter += 1
        Image.fromarray(cropped_image).save(save_path)

# ~ # Object Detection
# ~ interpreter = tf.lite.Interpreter(model_path='./gbif_model.tflite')
# ~ interpreter.allocate_tensors()

# ~ # Set up the functions
# ~ def preprocess_image(image_path, input_size):
    # ~ """Preprocess the input image to feed to the TFLite model"""
    # ~ img = tf.io.read_file(image_path)
    # ~ img = tf.io.decode_image(img, channels=3)
    # ~ img = tf.image.convert_image_dtype(img, tf.uint8)
    # ~ original_image = img
    # ~ resized_img = tf.image.resize(img, input_size)
    # ~ resized_img = resized_img[tf.newaxis, :]
    # ~ resized_img = tf.cast(resized_img, dtype=tf.uint8)
    # ~ return resized_img, original_image


# ~ def detect_objects(interpreter, image, threshold):
    # ~ """Returns a list of detection results, each a dictionary of object info."""

    # ~ signature_fn = interpreter.get_signature_runner()
    # ~ print(signature_fn)

    # ~ # Feed the input image to the model
    # ~ output = signature_fn(images=image)

    # ~ # Get all outputs from the model
    # ~ count = int(np.squeeze(output['output_0']))
    # ~ scores = np.squeeze(output['output_1'])
    # ~ classes = np.squeeze(output['output_2'])
    # ~ boxes = np.squeeze(output['output_3'])

    # ~ results = []
    # ~ for i in range(count):
        # ~ if scores[i] >= threshold:
            # ~ result = {
            # ~ 'bounding_box': boxes[i],
            # ~ 'class_id': classes[i],
            # ~ 'score': scores[i]
            # ~ }
            # ~ results.append(result)
    # ~ return results
    
# ~ def set_input_tensor(interpreter, image):
  # ~ """Sets the input tensor."""
  # ~ tensor_index = interpreter.get_input_details()[0]['index']
  # ~ input_tensor = interpreter.tensor(tensor_index)()[0]
  # ~ input_tensor[:, :] = image


# ~ def get_output_tensor(interpreter, index):
  # ~ """Returns the output tensor at the given index."""
  # ~ output_details = interpreter.get_output_details()[index]
  # ~ tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  # ~ return tensor
  

# ~ def detect_objects3(interpreter, image, threshold):
  # ~ """Returns a list of detection results, each a dictionary of object info."""
  # ~ output = interpreter.get_output_details()  # Model has single output.
  # ~ input = interpreter.get_input_details()  # Model has single input.
  # ~ #input_data = tf.constant(1., shape=[1, 1])
  # ~ interpreter.set_tensor(input['index'], image)
  # ~ interpreter.invoke()
  # ~ print(interpreter.get_tensor(output['index']))

  # ~ # Get all output details
  # ~ boxes = get_output_tensor(interpreter, 0)
  # ~ classes = get_output_tensor(interpreter, 1)
  # ~ scores = get_output_tensor(interpreter, 2)
  
  # ~ print(boxes)
  # ~ print(classes)
  
  # ~ #count = int(get_output_tensor(interpreter, 3))
  


  # ~ results = []
  # ~ for i in range(1):
    # ~ if scores[i] >= threshold:
      # ~ result = {
          # ~ 'bounding_box': boxes[i],
          # ~ 'class_id': classes[i],
          # ~ 'score': scores[i]
      # ~ }
      # ~ results.append(result)
  # ~ return results
    
# ~ def detect_objects3(interpreter, image, threshold):
    # ~ """Returns a list of detection results, each a dictionary of object info."""
    
    # ~ # Set input tensor
    # ~ interpreter.set_tensor(interpreter.get_input_details()[0]['index'], image)

    # ~ # Run inference
    # ~ interpreter.invoke()

    # ~ # Get output details
    # ~ output_details = interpreter.get_output_details()
    # ~ print(output_details)

    # ~ # Extract output tensors
    # ~ boxes = interpreter.get_tensor(output_details[0]['index'])
    # ~ classes = interpreter.get_tensor(output_details[1]['index'])
    # ~ scores = interpreter.get_tensor(output_details[2]['index'])
    # ~ count = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Extract the scalar value

    # ~ results = []
    # ~ for i in range(count):
        # ~ if scores[0, i] >= threshold:
            # ~ result = {
                # ~ 'bounding_box': boxes[0, i],
                # ~ 'class_id': classes[0, i],
                # ~ 'score': scores[0, i]
            # ~ }
            # ~ results.append(result)
    # ~ return results



# ~ def run_odt_and_draw_results(image_path, interpreter, threshold=0.5, label=True, width=12):
    # ~ """Run object detection on the input image and draw the detection results"""
    # ~ # Load the input shape required by the model
    # ~ _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    # ~ basepath = image_path.split('/')[-1]

    # ~ # Load the input image and preprocess it
    # ~ preprocessed_image, original_image = preprocess_image(
        # ~ image_path,
        # ~ (input_height, input_width)
    # ~ )

    # ~ # Run object detection on the input image
    # ~ results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # ~ # Plot the detection results on the input image
    # ~ original_image_np = original_image.numpy().astype(np.uint8)
    # ~ counter = 1
    # ~ for obj in results:
        # ~ # Convert the object bounding box from relative coordinates to absolute
        # ~ # coordinates based on the original image resolution
        # ~ ymin, xmin, ymax, xmax = obj['bounding_box']
        # ~ xmin = int(xmin * original_image_np.shape[1])
        # ~ xmax = int(xmax * original_image_np.shape[1])
        # ~ ymin = int(ymin * original_image_np.shape[0])
        # ~ ymax = int(ymax * original_image_np.shape[0])

        # ~ # Find the class index of the current object
        # ~ class_id = int(obj['class_id'])
        # ~ class_str = ['moth', 'non_moth'][class_id]
        
        # ~ # crop the image to the bounding box
        # ~ crop_img = original_image_np[ymin:ymax, xmin:xmax]
        # ~ # save the image
        # ~ Image.fromarray(crop_img).save('./cropped_common_species/' + class_str + '_' + str(counter) + '_' + basepath)
        # ~ counter = counter + 1

# ~ all_images = os.listdir('./common_species')
# ~ for image in all_images:
   # ~ image_file = './common_species/' + image
   # ~ detection_result_image = run_odt_and_draw_results(
       # ~ image_file,
       # ~ interpreter,
       # ~ threshold=0.3, width=2
   # ~ )

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
all_images = os.listdir('./cropped_common_species')

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

file_path = []
truth = []
pred = []
confidence = []
time = []

for image_file in all_images:
    image_path = './cropped_common_species/' + image_file
    image = Image.open(image_path).convert("RGB")  # Convert to RGB mode

    # Resize the image
    resized_image = image.resize((300, 300))

    # Convert to NumPy array and normalize
    img = np.array(resized_image) / 255.0
    img = (img - 0.5) / 0.5

    tflite_inf, conf, inf_time = tflite_inference(img, interpreter)

    file_path = file_path + [image_file]
    truth = truth + [' '.join(image_file.split('/')[-1].split('_')[-3:-1])]
    pred = pred + [species_names[tflite_inf]]
    confidence = confidence + [conf]
    time = time + [inf_time]

df = pd.DataFrame({'path':file_path, 'inference_time':time, 'truth':truth, 'pred':pred, 'confidence':confidence})
df['correct'] = np.where(df['pred'] == df['truth'], 1, 0)

df = df.sort_values('confidence', ascending=False)

df.to_csv(f'./results/{region}_predictions.csv', index=False)


print(df)

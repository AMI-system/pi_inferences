import datetime
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import pandas as pd


# Object Detection
interpreter = tf.lite.Interpreter(model_path='./gbif_model.tflite')
interpreter.allocate_tensors()

# Set up the functions
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

    signature_fn = interpreter.get_signature_list()#runner()
    print(signature_fn)

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

def run_odt_and_draw_results(image_path, interpreter, threshold=0.5, label=True, width=12):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    basepath = image_path.split('/')[-1]

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    counter = 1
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj['class_id'])
        class_str = ['moth', 'non_moth'][class_id]
        
        # crop the image to the bounding box
        crop_img = original_image_np[ymin:ymax, xmin:xmax]
        # save the image
        Image.fromarray(crop_img).save('./cropped_common_species/' + class_str + '_' + str(counter) + '_' + basepath)
        counter = counter + 1

all_images = os.listdir('./common_species')
for image in all_images:
   image_file = './common_species/' + image
   detection_result_image = run_odt_and_draw_results(
       image_file,
       interpreter,
       threshold=0.3, width=2
   )

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

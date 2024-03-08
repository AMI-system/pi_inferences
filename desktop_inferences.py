import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import os
import datetime
import json
import pandas as pd

def preprocess_image(image_path, input_size):
    """Preprocesses the input image for object detection."""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image

def detect_objects(interpreter, image, threshold):
    """Detects objects in the input image using the provided interpreter."""
    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    detections_list = []

    for i in range(count):
        bounding_box = boxes[i]

        origin_y, origin_x, max_y, max_x = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        width, height = max_x-origin_x, max_y-origin_y

        if scores[i] > threshold:
            category_dict = {
                'index': i,
                'score': scores[i],
                'display_name': classes[i],
                'category_name': class_labels[int(classes[i])],
            }

            detection_dict = {
                'counter': i,
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

def run_moth_detection(image_path, m_interpreter, threshold=0.5):
    """Runs object detection on the input image and returns the detection results."""
    # Load the input shape required by the model
    _, input_height, input_width, _ = m_interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    a = datetime.datetime.now()
    results = detect_objects(m_interpreter, preprocessed_image, threshold=threshold)
    b = datetime.datetime.now()
    c = b - a

    return results, original_image, str(c.microseconds)

def species_inference(crop_image, species_interpreter):
    """Performs species classification on a cropped image."""
    a = datetime.datetime.now()
    input_data = np.expand_dims(crop_image, axis=0).astype(np.float32)
    input_data = np.transpose(input_data, (0, 3, 1, 2))
    species_interpreter.set_tensor(species_interpreter.get_input_details()[0]['index'], input_data)
    species_interpreter.invoke()
    outputs_tf = species_interpreter.get_tensor(species_interpreter.get_output_details()[0]['index'])
    prediction_tf = np.squeeze(outputs_tf)
    confidence = np.exp(prediction_tf) / np.sum(np.exp(prediction_tf))
    prediction_tf = prediction_tf.argsort()[::-1][0]

    # Calculate inference time
    c = datetime.datetime.now() - a

    return prediction_tf, max(confidence) * 100, str(c.microseconds)

# Configuration
nclass = 2
DETECTION_THRESHOLD = 0.1
label_map = {1: 'moth', 2: 'nonmoth'}
class_labels = ['moth', 'nonmoth']
region = 'uk'

# Load moth detection model
moth_interpreter = tf.lite.Interpreter(model_path='./models/gbif_model_metadata.tflite')
moth_interpreter.allocate_tensors()

# Load species classification model
species_interpreter = tf.lite.Interpreter(model_path=f"./models/resnet_{region}.tflite")
species_interpreter.allocate_tensors()
species_names = json.load(open(f'./models/01_{region}_data_numeric_labels.json', 'r'))['species_list']

# Load input image
image_path = './example_images/ami_ami_20230722000010-00-35.jpg'
image = np.asarray(Image.open(image_path))
annot_image = image.copy()
annotated_image_path = os.path.join('./annotated_images/',
                                        os.path.basename(image_path))

# Run moth detection
detections_list, original_image, det_time = run_moth_detection(image_path, moth_interpreter, threshold=DETECTION_THRESHOLD)
original_image_np = original_image.numpy().astype(np.uint8)

# Process each detected moth
for detection in detections_list:
    bounding_box = detection['bounding_box']

    origin_x, origin_y, width, height = bounding_box['origin_x'], bounding_box['origin_y'], bounding_box['width'], bounding_box['height']

    # convert these to pixels
    xmin = int(origin_x * original_image_np.shape[1])
    xmax = int((origin_x + width) * original_image_np.shape[1])
    ymin = int(origin_y * original_image_np.shape[0])
    ymax = int((origin_y + height) * original_image_np.shape[0])

    # Slice the image using integer indices
    cropped_image = image[ymin:ymax, xmin:xmax]
    category_name = detection['categories'][0]['category_name']
    insect_score = detection['categories'][0]['score']
    resized_image = Image.fromarray(cropped_image).convert("RGB").resize((300, 300))
    img = np.array(resized_image) / 255.0
    img = (img - 0.5) / 0.5

    # Perform species classification
    species_inf, conf, inf_time = species_inference(crop_image=img, species_interpreter=species_interpreter)

    # Add bounding box annotation to the image
    bbox_color = (46, 139, 87) if category_name == 'moth' else (238, 75, 43)
    ann_label = f"{species_names[species_inf]}, {conf:.2f}"
    cv2.rectangle(annot_image,
                    (xmin, ymin),
                    (xmax, ymax),
                    bbox_color, 4)
    cv2.putText(annot_image, text=ann_label,
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2, color=bbox_color, thickness=4)

    # Save inference results to csv
    df = pd.DataFrame({
        'image_path': [image_path],
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'moth_class': [category_name],
        'insect_score': [insect_score],
        'detection_time': [det_time],
        'bounding_box': ['; '.join(map(str, [xmin, ymin, xmax, ymax]))],
        'annot_path': [annotated_image_path],
        'species_inference_time': [inf_time],
        'truth': [' '.join(image_path.split('/')[-1].split('_')[0:2])],
        'pred': [species_names[species_inf]],
        'confidence': [conf],
        'model': [region]
    })
    df['correct'] = np.where(df['pred'] == df['truth'], 1, 0)
    df.to_csv(f'./results/{region}_predictions.csv', index=False, mode='a', header=False)

print('Saved annotated image to: ', annotated_image_path)
cv2.imwrite(annotated_image_path, cv2.cvtColor(annot_image, cv2.COLOR_BGR2RGB))

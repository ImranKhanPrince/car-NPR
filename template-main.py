# Import necessary libraries
import cv2
import numpy as np
import easyocr
from urllib.request import urlretrieve
from PIL import Image
import requests
from io import BytesIO

# Define function for license plate detection
def detect_license_plate(image_path, conf_threshold=0.5, nms_threshold=0.5):
    # Load YOLOv4 model
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    # Get class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    # Set output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load input image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Detect objects using YOLOv4
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize variables
    class_ids = []
    confidences = []
    boxes = []

    # Loop over all detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Check if object detected is a license plate
            if classes[class_id] == "license_plate" and confidence > conf_threshold:
                # Get coordinates of object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Get top-left corner of bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Append results to lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-max suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding box around license plate and save cropped image
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            roi = img[y:y+h, x:x+w]
            cv2.imwrite("license_plate.jpg", roi)
            return "license_plate.jpg"
    else:
        return "License plate not detected"

# Define function for super resolution using ESRGAN
def super_resolution(image_path):
    # Load ESRGAN model
    model_url = "https://github.com/xinntao/ESRGAN/releases/download/v0.3.1/RRDB_ESRGAN_x4.pth"
    model_file = BytesIO(requests.get(model_url).content)
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(model_file.getvalue())
    model.setModel("esrgan", 4)

    # Load input image
    img = cv2.imread(image_path)

    # Apply super resolution
    result = model.upsample(img)

    # Save output image
    cv2.imwrite("super_resolution.jpg", result)
    return "super_resolution.jpg"

# Define function for character recognition using EasyOCR

def recognize_characters(image):
    """
    Recognizes the characters in a license plate image using EasyOCR.
    
    Args:
        image (PIL Image): The license plate image.
    
    Returns:
        A list of tuples representing the recognized characters and their bounding boxes.
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['bn'])
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Recognize characters in the image
    result = reader.readtext(image_array, detail=0)
    
    return result

detect_license_plate("./test_image")
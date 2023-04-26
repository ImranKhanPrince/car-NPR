import numpy as np
import easyocr
import matplotlib.pyplot as plt 
import cv2

IMAGE_PATH = "./images/test-image_cropped.png"

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

def cv_plot_image(image, index=0):
  plt.imshow(image)
  # window_name = 'image-'+index
  # cv2.imshow(window_name, image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
   

def path_to_image_array(path):
   npimage = cv2.imread(path)
   cv_plot_image(npimage)


def image_to_text(image_path):
  path_to_image_array(image_path)

if __name__ == "__main__":
  image_to_text(IMAGE_PATH)
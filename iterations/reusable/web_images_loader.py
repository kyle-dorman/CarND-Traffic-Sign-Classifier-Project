import cv2
import os
import reusable.file_loader as fl
from reusable.file_loader import ProjectDataSet

def load():
    name = "web_images.p"
    print("Loading web image data.")
    
    if os.path.isfile(fl.data_file_path(name)) == False:
        print("Creating web images file.")
        create_web_image_file(name)
    else:
        print("Already created web image file.")
         
    images = fl.open_pickle_file(name)

    print("Returning web_images as ProjectDataSet(features, labels).")
    return ProjectDataSet(images)
    
def create_web_image_file(name):
    images = [resize_image(load_image("g_{}".format(i))) for i in range(43)]
    labels = [i for i in range(43)]
    fl.save_pickle_file(name, {'features': images, 'labels': labels})

def load_image(name):
    img_name = image_file(name)
    img = cv2.imread(img_name)
    b,g,r = cv2.split(img)
    return cv2.merge((r,g,b))

def image_file(name):
    extensions = ['.jpg', '.jpeg']
    for ext in extensions:
        full_path = fl.absolute_base_dir("web_images") + "/" + name + ext
        if os.path.isfile(full_path):
            return full_path
    return None
                                   
def resize_image(img):
    return cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)

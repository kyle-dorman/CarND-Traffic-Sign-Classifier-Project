import cv2
import numpy as np
import random
import pickle
from sklearn.utils import shuffle

from reusable.load import load_project_data
import reusable.file_loader as fl
from reusable.file_loader import ProjectDataSet
import os

## Based on this paper, http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
## Samples are randomly perturbed 
##  in position ([-2,2] pixels), 
##  in rotation ([-15,+15] degrees)
##  in brightness ([0.7, 2.0] gamma)
def transform_image(img, min_gamma=0.7, max_gamma=2.0, shift=2, angle=15.0):
	img = change_image_gamma(img, min_gamma=min_gamma, max_gamma=max_gamma)
	img = shift_image(img, shift=shift)
	return rotate_image(img, angle=angle)
  
def shift_image(img, shift=4):
	rows,cols,channels = img.shape
	x_shift = random.randint(-shift, shift)
	y_shift = random.randint(-shift, shift)
	M = np.float32([[1,0,x_shift],[0,1,y_shift]])
	return cv2.warpAffine(img,M,(cols,rows))

def rotate_image(img, angle=20.0):
	rows,cols,channels = img.shape
	rotation_angle = random.uniform(-angle, angle)
	M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_angle,1)
	return cv2.warpAffine(img,M,(cols,rows))

def change_image_gamma(img, min_gamma=0.7, max_gamma=2.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	gamma = random.uniform(min_gamma, max_gamma)
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
	for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)

def append_jiggered_data(X_train, y_train):
	print("Creating jiggered data...")
	unique, counts = np.unique(y_train, return_counts=True)
	# dict of original images
	train_dict = {u: [] for u in unique}
	# dict where we where add jiggered data
	jiggered_dict = {u: [] for u in unique}

	for i in range(len(X_train)):
		train_dict[y_train[i]].append(X_train[i])
		jiggered_dict[y_train[i]].append(X_train[i])

	max_img_count = max(counts)
	for k in train_dict.keys():
		# add (max_image_count - kth_image_count) jiggered images
		for i in range(counts[k], max_img_count):
			old_i = random.randint(0, len(train_dict[k]) - 1)
			jiggered_dict[k].append(transform_image(train_dict[k][old_i]))

	X_train_jiggered = np.concatenate([v for v in jiggered_dict.values()])
	y_train_jiggered = np.concatenate([[i for j in range(max_img_count)] for i in jiggered_dict.keys()])
	print("Finished creating jiggered data...")
	return shuffle(X_train_jiggered, y_train_jiggered)

def load_jiggered_data():
	jiggered_file = "train_jiggered.p"

	if os.path.isfile(fl.data_file_path(jiggered_file)) == False:
		data = load_project_data()
		X_train_jiggered, y_train_jiggered = append_jiggered_data(data.train.features, data.train.labels)
		fl.save_pickle_file(jiggered_file, (X_train_jiggered, y_train_jiggered))
	else:
		X_train_jiggered, y_train_jiggered = fl.open_pickle_file(jiggered_file)
		X_train_jiggered, y_train_jiggered = shuffle(X_train_jiggered, y_train_jiggered)
	return ProjectDataSet({'features': X_train_jiggered, 'labels': y_train_jiggered })

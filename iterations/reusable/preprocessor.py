# load preprocessed pickled data. If data isnt there, load it. 
import reusable.file_loader as fl
from reusable.file_loader import ProjectData
from reusable.file_loader import ProjectDataSet
import os
from sklearn.utils import shuffle
from reusable.load import load_project_data
from reusable.load_jiggered import load_jiggered_data
import numpy as np

def preprocess(features, labels, name="name"):
	print("Preprocessing " + name + "...")
	features, labels = shuffle(features, labels)
	features = to_greyscale(features)
	features = scale(features)
	print("Finished preprocessng " + name + "...")
	return (features, labels)

def to_greyscale(X_data):
	result = [None for i in range(len(X_data))]
	for index in range(len(X_data)):
		if index % 10000 == 0:
			print("Converting image #"+str(index), "to greyscale.")
		result[index] = rgb2gray(X_data[index])
	return np.array(result, dtype='float32')

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return [[[val] for val in row] for row in gray]

def scale(X_data):
  return np.divide(np.subtract(X_data, 128), 128)

def load_greyscale_train_data():
	train_file = "train_greyscale_preprocessed.p"

	if os.path.isfile(fl.data_file_path(train_file)) == False:
		orig_data = load_project_data().train
		print("Unable to find pre-preprocessed greyscale data.")
		X_train, y_train = (orig_data.features, orig_data.labels)
		X_train = to_greyscale(X_train)
		print("Finished converting training images to grey scale.")
		fl.save_pickle_file(train_file, (X_train, y_train))
	else:
		print("Loading pre-preprocessed greyscale data...")
		X_train, y_train = fl.open_pickle_file(train_file)

	return ProjectDataSet({'features': X_train, 'labels': y_train })

def load_preprocessed_data():
	train_file = "train_preprocessed.p"
	valid_file = "valid_preprocessed.p"
	test_file = "test_preprocessed.p"

	if os.path.isfile(fl.data_file_path(train_file)) == False:
		orig_data = load_project_data()
		print("Unable to find pre-preprocessed data.")
		X_train, y_train = preprocess(orig_data.train.features, orig_data.train.labels, name="train data")
		X_valid, y_valid = preprocess(orig_data.valid.features, orig_data.valid.labels, name="train data")
		X_test, y_test = preprocess(orig_data.test.features, orig_data.test.labels, name="test data")

		fl.save_pickle_file(train_file, (X_train, y_train))
		fl.save_pickle_file(valid_file, (X_valid, y_valid))
		fl.save_pickle_file(test_file, (X_test, y_test)) 
	else:
		print("Loading pre-preprocessed data...")
		X_train, y_train = fl.open_pickle_file(train_file)
		X_train, y_train = shuffle(X_train, y_train)
		X_valid, y_valid = fl.open_pickle_file(valid_file)
		X_test, y_test = fl.open_pickle_file(test_file)

	return ProjectData(
		{'features': X_train, 'labels': y_train }, 
		{'features': X_valid, 'labels': y_valid }, 
		{'features': X_test, 'labels': y_test })    

def load_preprocessed_jiggered_data():
	train_file = "train_jiggered_preprocessed.p"
	valid_file = "valid_preprocessed.p"
	test_file = "test_preprocessed.p"

	if os.path.isfile(fl.data_file_path(train_file)) == False:
		orig_data = load_project_data()
		jiggered_data = load_jiggered_data()
		print("Unable to find pre-preprocessed data.")
		X_train, y_train = preprocess(jiggered_data.features, jiggered_data.labels, name="train data")
		X_valid, y_valid = preprocess(orig_data.valid.features, orig_data.valid.labels, name="valid data")
		X_test, y_test = preprocess(orig_data.test.features, orig_data.test.labels, name="test data")

		fl.save_pickle_file(train_file, (X_train, y_train))
		fl.save_pickle_file(valid_file, (X_valid, y_valid))
		fl.save_pickle_file(test_file, (X_test, y_test)) 
	else:
		print("Loading pre-preprocessed data...")
		X_train, y_train = fl.open_pickle_file(train_file)
		X_train, y_train = shuffle(X_train, y_train)
		X_valid, y_valid = fl.open_pickle_file(valid_file)
		X_test, y_test = fl.open_pickle_file(test_file)

	return ProjectData(
		{'features': X_train, 'labels': y_train }, 
		{'features': X_valid, 'labels': y_valid },
		{'features': X_test, 'labels': y_test })
	
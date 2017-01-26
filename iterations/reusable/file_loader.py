import pickle
import os
import os.path
import zipfile
from urllib.request import urlretrieve
import pickle

class ProjectData:
	def __init__(self, train, valid, test):
		self.train = ProjectDataSet(train)
		self.valid = ProjectDataSet(valid)
		self.test = ProjectDataSet(test)

class ProjectDataSet:
	def __init__(self, data):
		self.features = data['features']
		self.labels = data['labels']

def data_file_path(file_name):
	BASE_DIR = absolute_base_dir('data')
	abs_file_name = BASE_DIR + "/" + file_name
	return abs_file_name

def absolute_base_dir(name):
	base_dir_name = "CarND-Traffic-Sign-Classifier-Project"
	base_dir_list = os.path.dirname(os.path.realpath(__file__)).split("/")
	i = base_dir_list.index(base_dir_name)
	return "/".join(base_dir_list[0:i+1]) + "/" + name

def download_file(url, file):
	"""
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
	absolute_file = data_file_path(file)
	if os.path.isfile(absolute_file) == False:
		print("Unable to find " + file + ". Downloading now...")
		urlretrieve(url, absolute_file)
		print('Download Finished!')
	else:
		print(file + " already downloaded.")

def unzip_data(file_name):
	absolute_file = data_file_path(file_name)
	with zipfile.ZipFile(absolute_file, "r") as zip_ref:
		print("Extracting zipfile " + absolute_file + "...")
		zip_ref.extractall(absolute_base_dir('data'))

def open_pickle_file(file_name):
	print("Unpickling file " + file_name + ".")
	full_file_name = data_file_path(file_name)
	with open(full_file_name, mode='rb') as f:
		return pickle.load(f)

def save_pickle_file(file, data):
	abs_file = data_file_path(file)
	pickle.dump(data, open(abs_file, "wb" ) )




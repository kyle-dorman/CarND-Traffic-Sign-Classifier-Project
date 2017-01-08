# Load pickled data
import pickle
import os.path
import zipfile

class ProjectData:
	def __init__(self, train, test):
		self.train = ProjectDataSet(train)
		self.test = ProjectDataSet(test)

class ProjectDataSet:
	def __init__(self, data):
		self.features = data['features']
		self.labels = data['labels']

def load_data():
	print("Loading project data.")
	training_file, testing_file = _file_paths()

	if os.path.isfile(training_file) == False:
		print("Unable to find unzip data files.")
		_unzip_data()
	else:
		print("Data already unzipped.")

	print("Returning project data. ProjectData(train, test).")
	return _return_data()

def _file_paths():
	BASE_DIR = _absolute_base_dir()
	training_file = BASE_DIR + "/test.p"
	testing_file = BASE_DIR + "/test.p"
	return (training_file, testing_file)

def _absolute_base_dir():
	base_dir_name = "CarND-Traffic-Sign-Classifier-Project"
	base_dir_list = os.path.dirname(os.path.realpath(__file__)).split("/")
	i = base_dir_list.index(base_dir_name)
	return "/".join(base_dir_list[0:i+1])

def _unzip_data():
	traffic_zip = _absolute_base_dir() + "/traffic-signs-data.zip"
	with zipfile.ZipFile(traffic_zip,"r") as zip_ref:
		print("Extracting zipfile " + traffic_zip)
		zip_ref.extractall(_absolute_base_dir())

def _return_data():
	training_file, testing_file = _file_paths()

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)

	return ProjectData(train, test)

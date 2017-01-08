# Load pickled data
import pickle
import os.path
import zipfile
from urllib.request import urlretrieve

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
		_unzip_traffic_data()
	else:
		print("Data already unzipped.")

	return _open_pickle(training_file, testing_file)

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

def _unzip_traffic_data():
	local_traffic_file = "traffic-signs-data.zip"
	absolute_traffic_zip_file = _absolute_base_dir() + "/" + local_traffic_file
	if os.path.isfile(absolute_traffic_zip_file) == False:
		url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
		_download(url, absolute_traffic_zip_file)

	_unzip(absolute_traffic_zip_file, _absolute_base_dir())

def _unzip(zip_file, folder):
	with zipfile.ZipFile(zip_file, "r") as zip_ref:
		print("Extracting zipfile " + zip_file + "...")
		zip_ref.extractall(folder)

def _download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished!')

def _open_pickle(train_file, test_file):
	print("Returning ProjectData(train, test).")

	with open(train_file, mode='rb') as f:
		train = pickle.load(f)
	with open(test_file, mode='rb') as f:
		test = pickle.load(f)

	return ProjectData(train, test)

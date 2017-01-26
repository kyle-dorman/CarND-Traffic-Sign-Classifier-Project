# Load original pickled project data
import reusable.file_loader as fl
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from reusable.file_loader import ProjectData

def load_project_data():
	print("Loading project data.")
	train_file_name = "train.p"
	test_file_name = "test.p"
	url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
	zip_file_name = "traffic-signs-data.zip"

	fl.download_file(url, zip_file_name)
	if os.path.isfile(fl.data_file_path(train_file_name)) == False:
		print("Unable to find unzip data files. Unzipping now.")
		fl.unzip_data(zip_file_name)
	else:
		print("Data already unzipped.")

	train = fl.open_pickle_file(train_file_name)
	X_train, y_train = shuffle(train['features'], train['labels'])
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train)
	test = fl.open_pickle_file(test_file_name)

	print("Returning ProjectData(train, test).")
	return ProjectData({'features': X_train, 'labels': y_train }, 
		{'features': X_valid, 'labels': y_valid },
		test)

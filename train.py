import numpy as np

from utils import encoding
from dataset import load_clinical_data, aif360_process

data_train, data_test = load_clinical_data(
    "./data/data_aif360_withMMSE.csv",
    "./data/data_aif360_adi.csv",
    "MMSE",
    "copy",
)

data_train_sd, data_train_transf, data_test_sd = aif360_process(data_train, data_test, "MMSE")

X_train_data = data_train_sd.features
y_train_data = data_train_sd.labels.ravel()

X_test_data = data_test_sd.features
y_test_data = data_test_sd.labels.ravel()

print(X_train_data.shape)
print("After upsampling:", np.count_nonzero(y_train_data==0), np.count_nonzero(y_train_data==1), y_train_data.shape)
print(X_train_data[0][10000:])
print(data_train_sd.feature_names[10000:])

X_train_data_bylo, X_test_data_bylo = encoding("100000", X_train_data, y_train_data, X_test_data, y_test_data, 64)
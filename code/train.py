import numpy as np
import argparse
import os

from utils import encoding, test_classifier_bootstrap
from model import fit_model
from dataset import load_clinical_data, aif360_process

from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--data_root")
parser.add_argument("--data", choices=["MMSE", "ATT", "MEM"])
parser.add_argument("--data_condition", choices=["copy", "command", "combine"])
parser.add_argument("--model", choices=["LR", "SVM", "XGB"])
parser.add_argument("--fair", action="store_true")

args = parser.parse_args()
model = args.model
fair = args.fair
data_path = args.data_root

if args.data == "MMSE":
    data_file = "data_aif360_withMMSE.csv"
else:
    data_file = "data_aif360_withATTMEM.csv"

data_train, data_test = load_clinical_data(
    args,
    data_file
)

data_train_sd, data_train_transf, data_test_sd = aif360_process(data_train, data_test, args.data)

X_train_data = data_train_sd.features
y_train_data = data_train_sd.labels.ravel()

X_test_data = data_test_sd.features
y_test_data = data_test_sd.labels.ravel()

print(X_train_data.shape)
print("After upsampling:", np.count_nonzero(y_train_data==0), np.count_nonzero(y_train_data==1), y_train_data.shape)
print(X_train_data[0][10000:])
print(data_train_sd.feature_names[10000:])

# Encode clinical data with RFVAE
X_train_data_encoded, X_test_data_encoded = encoding("./encoder", X_train_data, y_train_data, X_test_data, y_test_data, 64)
# print(X_train_data_encoded.shape)

# Train machine learing models
scaler = MinMaxScaler()
scaler.fit(np.append(X_train_data_encoded,X_test_data_encoded,axis=0))
X_train_data_encoded_scaled = scaler.transform(X_train_data_encoded)
X_test_data_encoded_scaled = scaler.transform(X_test_data_encoded)

# Re-weighting
if fair:
    sample_weight = data_train_sd.instance_weights
else:
    sample_weight = data_train_transf.instance_weights

best_clf = fit_model("SVM", X_train_data_encoded_scaled, y_train_data, sample_weight = sample_weight)

results = test_classifier_bootstrap(X_test_data_encoded_scaled, y_test_data, best_clf)
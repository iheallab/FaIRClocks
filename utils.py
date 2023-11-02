import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm

import torch


from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, average_precision_score,precision_score

from sklearn.neural_network import MLPClassifier
    
batch_size = 16
n_epoch = 5

def set_seeds(my_seed=42):
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from dataset import createDatasets
from model import loadEncoder, NeuralNetwork, NeuralNetworkDemographics

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)

def encoding(model, X_train_data, y_train_data, X_test_data, y_test_data, batch_size):
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    rfvae_path = model
    Encoder = loadEncoder(rfvae_path, 10).cuda()

    Data_Transforms = transforms.Compose([
                                    transforms.Resize((64,64)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])

    train_dataset = Clocks_Dataset(X_train_data, y_train_data, transform=Data_Transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    test_dataset = Clocks_Dataset(X_test_data, y_test_data, transform=Data_Transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    feature_vector = np.empty((batch_size, 10))
    for x,y in tqdm(train_loader):
        x = x.cuda()
        with torch.no_grad():
            f = Encoder(x)
            mu = f[:, :10]
            logvar = f[:, 10:]
            z = reparametrize(mu, logvar).squeeze()
        feature_vector = np.append(feature_vector, z.cpu().detach().numpy(), axis=0)

    X_train_data_bylo = np.append(feature_vector[batch_size:], X_train_data[:, 10000:], axis=1)
    print("Train: ", X_train_data_bylo.shape)

    feature_vector = np.empty((batch_size, 10))
    for x,y in tqdm(test_loader):
        x = x.cuda()
        with torch.no_grad():
            f = Encoder(x)
            mu = f[:, :10]
            logvar = f[:, 10:]
            z = reparametrize(mu, logvar).squeeze()
        feature_vector = np.append(feature_vector, z.cpu().detach().numpy(), axis=0)

    # X_test_data_bylo = feature_vector[batch_size:]
    X_test_data_bylo = np.append(feature_vector[batch_size:], X_test_data[:, 10000:], axis=1)
    print("Test: ", X_test_data_bylo.shape)
    return X_train_data_bylo, X_test_data_bylo

# def test_classifier(model_type, X_test_data_bylo_scaled, y_test_data, best_clf):
#     print("test shape: ", X_test_data_bylo_scaled.shape)
#     test_acc = best_clf.score(X_test_data_bylo_scaled, y_test_data)
#     output = best_clf.predict(X_test_data_bylo_scaled)
#     if model_type == 'Xgb':
#         prob = best_clf.predict_proba(X_test_data_bylo_scaled)[:,1]
#     else:
#         prob = best_clf.decision_function(X_test_data_bylo_scaled)
#     cm = confusion_matrix(y_test_data, output)
#     print(cm)
#     try:
#         tn, fp, fn, tp = confusion_matrix(y_test_data, output).ravel()
#         specificity = tn/(tn+fp)
#         sensitivity = tp/(tp+fn)
#         f1 = f1_score(y_test_data, output)
#         ras = roc_auc_score(y_test_data, prob)
#         aps = average_precision_score(y_test_data, prob)
#         print("Accuracy = {:.4}\nROC AUC score = {:.4}\nf1 score = {:.4}\nsensitivity = {:.4}\nspecificity = {:.4}\naverage precision score = {:.4}".format(test_acc, ras, f1, sensitivity, specificity, aps))
#         return {'Acc': test_acc, 'ROC_AUC': ras, 'F1': f1, 'sensitivity': sensitivity, 'specificity': specificity, 'APS':aps}
#     except:
#         f1 = f1_score(y_test_data, output)
#         aps = average_precision_score(y_test_data, prob)
#         print("Accuracy = {:.4}\nf1 score = {:.4}\naverage precision score = {:.4}".format(test_acc, f1, aps))
#         return {'Acc': test_acc, 'F1': f1, 'APS':aps}

def test_classifier_bootstrap(X_test_data_bylo_scaled, y_test_data, best_clf):
    np.random.seed(42)
    def calculate_ci(m):
        m = np.array(m)
        m = m[~np.isnan(m)]
        m.sort()
        upper = np.percentile(m, 97.5)
        lower = np.percentile(m, 2.5)
        return [lower, upper]
    accuracy = []
    roc_auc = []
    f1 = []
    sensitivity = []
    specificity = []
    avg_precision = []
    precision = []
    
    test_arr = np.concatenate((X_test_data_bylo_scaled, np.broadcast_to(np.array(np.reshape(y_test_data, (-1, 1))), X_test_data_bylo_scaled.shape[:-1] + (1,))), axis = -1) #change the #broadcasting dimnensions based on the shape of X_test
    n = X_test_data_bylo_scaled.shape[-1] #number of features in the X.
    
    for i in range(100):
        idx = np.random.randint(test_arr.shape[0], size=test_arr.shape[0])#select a set of random indices with replacement. 
        bootstrap = test_arr[idx]#bootstrapping test dataframe. Randomly sampling with replacement

        #Splitting each test bootstrap into data and labels
        x_test_ = bootstrap[:,0:n] #splitting into data. 
        y_test_ = bootstrap[:,n] #splitting into labels

        test_acc = best_clf.score(x_test_, y_test_)
        output = best_clf.predict(x_test_)
        prob = best_clf.decision_function(x_test_)
        cm = confusion_matrix(y_test_, output)
        try:
            tn, fp, fn, tp = confusion_matrix(y_test_, output).ravel()
        except:
            pass
        try:
            roc_auc.append(roc_auc_score(y_test_, prob))
        except:
            pass
        accuracy.append(test_acc)
        sensitivity.append(float(tp)/(tp+fn))
        specificity.append(float(tn)/(tn+fp))
        f1.append(f1_score(y_test_, output))
        avg_precision.append(average_precision_score(y_test_, prob))
        precision.append(precision_score(y_test_, output))
        
    accuracy = np.array(accuracy)
    roc_auc = np.array(roc_auc)
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)
    
    accuracy = accuracy[~np.isnan(accuracy)]
    roc_auc = roc_auc[~np.isnan(roc_auc)]
    sensitivity = sensitivity[~np.isnan(sensitivity)]
    specificity = specificity[~np.isnan(specificity)]
    
    conf_interval_acc = calculate_ci(accuracy)
    conf_interval_specificity = calculate_ci(specificity)
    conf_interval_sensitivity = calculate_ci(sensitivity)
    conf_interval_f1 = calculate_ci(f1)
    conf_interval_ras = calculate_ci(roc_auc)
    conf_interval_aps = calculate_ci(avg_precision)
    conf_interval_pre = calculate_ci(precision)
    
    output = best_clf.predict(X_test_data_bylo_scaled)
    cm = confusion_matrix(y_test_data, output)
    print(cm)

    print(f'''Accuracy (95% C.I) = {np.median(accuracy):.2f} ({conf_interval_acc[0]:.2f}-{conf_interval_acc[1]:.2f})
    ROC AUC score (95% C.I) = {np.median(roc_auc):.2f} ({conf_interval_ras[0]:.2f}-{conf_interval_ras[1]:.2f})
    f1 score (95% C.I) = {np.median(f1):.2f} ({conf_interval_f1[0]:.2f}-{conf_interval_f1[1]:.2f})
    sensitivity (95% C.I) = {np.median(sensitivity):.2f} ({conf_interval_sensitivity[0]:.2f}-{conf_interval_sensitivity[1]:.2f})
    specificity (95% C.I) = {np.median(specificity):.2f} ({conf_interval_specificity[0]:.2f}-{conf_interval_specificity[1]:.2f})
    average precision score (95% C.I) = {np.median(avg_precision):.2f} ({conf_interval_aps[0]:.2f}-{conf_interval_aps[1]:.2f})
    precision score (95% C.I) = {np.median(precision):.2f} ({conf_interval_pre[0]:.2f}-{conf_interval_pre[1]:.2f})''')
    return {'Acc': np.median(accuracy),
            'ROC_AUC': np.median(roc_auc),
            'F1': np.median(f1),
            'sensitivity': np.median(sensitivity),
            'specificity': np.median(specificity),
            'APS':np.median(avg_precision),
            'ci': {'sensitivity': [conf_interval_sensitivity[0], conf_interval_sensitivity[1]],
                   'specificity': [conf_interval_specificity[0], conf_interval_specificity[1]]
                  },
            'AUROC': roc_auc,
            'Sen': sensitivity,
            'Spe': specificity,
           }


class Clocks_Dataset(Dataset):  #Clock dataset 
    """Clocks Dataset"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        image = self.images[ids]
        image = np.asarray(image[:10000]).reshape(100,100)  #convert images to a numpy array
        image = np.stack((image,)*3, axis=2)  #create 3-channel numpy array from single-channel numpy array
        image = Image.fromarray(image.astype('uint8'), 'RGB')  #convert numpy array into a PIL image. 
        #sample = torch.from_numpy(image)
        label = self.labels[ids]
        label = np.asarray(label)
        label = torch.from_numpy(label)
        if self.transform:
            sample = self.transform(image)
        return (sample, label)

class dementia_data(Dataset):
    def __init__(self, X, y, demo, train=True):
        self.X = X
        self.y = y
        self.demo = demo
        self.train = train
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx]
        label = self.y[idx]
        return sample, label

def combined_shap_edu(shap_values, proj = 10):
    new_shap_values = []
    tmp = []
    for sample in shap_values[0]:
        image = 0
        race = 0
        sex = 0
        ethnicity = 0
        for i in range(proj):
            image += sample[i]
        for i in range(proj+5,proj+8):
            race += sample[i]
        for i in range(proj+3,proj+5):
            sex += sample[i]
        for i in range(proj+8,proj+11):
            ethnicity += sample[i]
        values = np.array([image, sample[proj], sample[proj+1], sample[proj+2], sex,  race, ethnicity])
        tmp.append(values)
    new_shap_values.append(np.array(tmp))

    tmp = []
    for sample in shap_values[1]:
        image = 0
        race = 0
        sex = 0
        ethnicity = 0
        for i in range(proj):
            image += sample[i]
        for i in range(proj+5,proj+8):
            race += sample[i]
        for i in range(proj+3,proj+5):
            sex += sample[i]
        for i in range(proj+8,proj+11):
            ethnicity += sample[i]
        values = np.array([image, sample[proj], sample[proj+1], sample[proj+2], sex,  race, ethnicity])
        tmp.append(values)
    new_shap_values.append(np.array(tmp))
    return new_shap_values

class Clocks_Dataset_encoder(Dataset):  #Clock dataset 
    """Clocks Dataset"""
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        image = self.images[ids]
        image = np.asarray(image[:10000]).reshape(100,100)  #convert images to a numpy array
        image = np.stack((image,)*3, axis=2)  #create 3-channel numpy array from single-channel numpy array
        image = Image.fromarray(image.astype('uint8'), 'RGB')  #convert numpy array into a PIL image. 
        if self.transform:
            sample = self.transform(image)
        return sample
    
def statistic_data(df, target):
    print('数量', df['studyid'].nunique())
    print('年龄', df['age'].mean(), df['age'].std())
    print(df['Edu_Years'].mean(), df['Edu_Years'].std())
    print((df['sex'] == 'F').mean() * 100)
    print((df['Race'] == 'WHITE').mean() * 100)
    print(df[target].mean())
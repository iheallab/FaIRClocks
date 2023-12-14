import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.explainers import MetricTextExplainer

import aif360

from collections import defaultdict

target_map = {
    "MMSE": "mmse_categ",
}

def statistic_data(df, target):
    print('Data statistic')
    print(' Number of patients: ', df['studyid'].nunique())
    print(' Age: ', df['age'].mean(), df['age'].std())
    print(' Education years: ', df['Edu_Years'].mean(), df['Edu_Years'].std())
    print(' Sex distribution (F): ', (df['sex'] == 'F').mean() * 100)
    print(' Race distribution (White): ', (df['Race'] == 'WHITE').mean() * 100)
    print(f' {target} distribution: ', df[target].value_counts())

def preprocess_data(df, target):
    df['Ethnicity'] = df['Ethnicity'].map({"NOT HISPANIC": 1, "HISPANIC": 0})
    df['Race'] = df['Race'].map({"WHITE": 1, "BLACK": 0})
    df['Edu_Years'] = df['Edu_Years'].apply(lambda x: 0 if x <= 8 else 1)
    df[target] = df[target].apply(lambda x: 1 if x != 0 else 0)
    return pd.get_dummies(df, columns=['sex', 'Ethnicity', 'Race'])

def generate_data(files, clocks, data_df, target):
    data = []
    for i, file in enumerate(files):
        demo = data_df.loc[data_df['studyid'] == file]
        if not demo.empty:
            col_names = list(range(10000)) + demo.columns.to_list()
            combined_data = np.append(clocks[i], demo.to_numpy().squeeze(), axis=0)
            data.append(combined_data)
    return pd.DataFrame(np.array(data), columns=col_names)

def load_clinical_data(args, data_file):
    path_target = os.path.join(args.data_root, data_file)
    path_adi = os.path.join(args.data_root, "data_aif360_adi.csv")
    target = args.data
    data_condition = args.data_condition

    target = target_map[target]

    df_data = pd.read_csv(path_target).merge(pd.read_csv(path_adi), how='inner', on='studyid').dropna()
    cols_to_drop = ['Unnamed: 0', 'name', 'age_categ', 'edu_categorical', 'Service', 'Charlson_Comorbidity_Index', 'CPT_1', 'CPT_1_Description', 'Payer', 'Payer_categ', 'EmployeeStatus', '30_DAY_MORTALITY', 'ASA_Anest_Record']
    df_data_thin = df_data.drop(columns=cols_to_drop).drop_duplicates(subset='studyid', keep='first')

    df_data_demo = df_data_thin.iloc[:, 10000:]
    
    if target == "mmse_categ":
        df_data_demo.pop('mmse_total')
    
    statistic_data(df_data_demo, target)
    df_data_onehot = preprocess_data(df_data_demo, target)
    df_data_onehot.replace({True: 1, False: 0}, inplace=True)

    train_data, test_data = train_test_split(df_data_onehot, test_size=0.30, stratify=df_data_thin.mmse_categ, random_state=29)

    task = data_condition

    files = np.load(os.path.join(args.data_root, f"files_PRECEDE_preop_{task}_v2.npy"))
    clocks = np.load(os.path.join(args.data_root, f"x_PRECEDE_preop_{task}_v2.npy"))

    df_train = generate_data(files, clocks, train_data, target)
    df_test = generate_data(files, clocks, test_data, target)

    for df in [df_train, df_test]:
        df_tmp = df.pop(target)
        df.pop('studyid')
        df[target] = df_tmp

    return df_train, df_test

def convert_to_standard_dataset(df, target_label_name, scores_name=""):

    # List of names corresponding to protected attribute columns in the dataset.
    # Note that the terminology "protected attribute" used in AI Fairness 360 to
    # divide the dataset into multiple groups for measuring and mitigating 
    # group-level bias.
    protected_attributes=['Edu_Years']
    
    # columns from the dataset that we want to select for this Bias study
    selected_features = list(df.columns[:-1])
    
    # This privileged class is selected based on MDSS subgroup evaluation.
    # in previous steps. In our case non-homeowner (homeowner=0) are considered to 
    # be privileged and homeowners (homeowner=1) are considered as unprivileged.
    privileged_classes = [[1]]

    # Label values which are considered favorable are listed. All others are 
    # unfavorable. Label values are mapped to 1 (favorable) and 0 (unfavorable) 
    # if they are not already binary and numerical.
    favorable_target_label = [1]

    # List of column names in the DataFrame which are to be expanded into one-hot vectors.
    categorical_features = []

    # create the `StandardDataset` object
    standard_dataset = aif360.datasets.StandardDataset(df=df, label_name=target_label_name,
                                    favorable_classes=favorable_target_label,
                                    scores_name=scores_name,
                                    protected_attribute_names=protected_attributes,
                                    privileged_classes=privileged_classes,
                                    categorical_features=categorical_features,
                                    features_to_keep=selected_features)
    if scores_name=="":
        standard_dataset.scores = standard_dataset.labels.copy()
        
    return standard_dataset

def aif360_process(data_train, data_test, target):
    target = target_map[target]
    data_train_sd = convert_to_standard_dataset(data_train, target)
    data_test_sd = convert_to_standard_dataset(data_test, target)

    privileged_groups = [{'Edu_Years': 1}]
    unprivileged_groups = [{'Edu_Years': 0}]

    metric_orig = aif360.metrics.BinaryLabelDatasetMetric(data_train_sd,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups,
                                                )
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig.mean_difference())
    explainer_metric_orig = MetricTextExplainer(metric_orig)
    print(explainer_metric_orig.disparate_impact())

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)

    data_train_transf = RW.fit_transform(data_train_sd)

    metric_transf = BinaryLabelDatasetMetric(data_train_transf,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf.mean_difference())
    explainer_metric_transf = MetricTextExplainer(metric_transf)
    print(explainer_metric_transf.disparate_impact())

    return data_train_sd, data_train_transf, data_test_sd
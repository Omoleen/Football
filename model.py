import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import f_classif
from sklearn import metrics

def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data
    y_pred = model.predict(x_test)

    # predict probabilistic
    # y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # research on f0.5
    f1 = metrics.fbeta_score(y_test, y_pred, beta=0.5)

    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    clas = metrics.classification_report(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)


    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)  # when both classes are important

    # Precision recall AUC curve
    prauc = metrics.average_precision_score(y_test, y_pred_proba)  # when the positive class is the most important
    # optimal_idx = np.argmax(tpr-fpr)
    # optimal_threshold = _[optimal_idx]
    # print(f'Optimal threshold value is : {optimal_threshold}')
    roc_score = 0
    threshold_value = 0.2
    step_factor = 0.025
    thrsh_score = 0
    while threshold_value <= .8:
        temp_thresh = threshold_value
        predicted = (y_pred_proba >= temp_thresh).astype('int')
        temp_roc = metrics.average_precision_score(y_test, predicted)
        # print(f'Threshold {temp_thresh} -- {temp_roc}')
        if roc_score < temp_roc:
            roc_score = temp_roc
            thrsh_score = threshold_value
        threshold_value = threshold_value + step_factor
    optimal_threshold = thrsh_score

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'clas': clas, 'mcc': mcc, 'optimal_threshold': optimal_threshold}

def read_scraped_data():
    df = pd.read_csv('train1_without_xg_10.csv')
    # df2 = pd.read_csv('validate1_with_xg_10.csv')
    # df = pd.concat([df1, df2]).drop_duplicates(keep='first').reset_index(drop=True)
    # df.drop(['index'], inplace=True)
    print(len(df))
    df['team_a_avg_goals_sh_scored'] = df.team_a_avg_goals_sh_scored.abs()
    df['team_a_avg_goals_sh_conc'] = df.team_a_avg_goals_sh_conc.abs()
    df['team_b_avg_goals_sh_conc'] = df.team_b_avg_goals_sh_conc.abs()
    # df = df[df['team_b_corners_result'] < 20.0].copy()
    # df = df[df['team_a_corners_result'] < 20.0].copy()
    df = df[df['team_a_pos'] <= 1.0].copy()
    df = df[df['team_b_pos'] <= 1.0].copy()
    df = df[~df['team_a_name'].str.contains(r'Res\.')].copy()
    # print(df[df['team_b_ft_result'] == 11.0].reset_index().drop(['index'], axis=1).to_markdown())
    # print(df[df['team_a_corners_result'] > 15.0].reset_index().drop(['index'], axis=1).to_markdown())

    df.drop(['team_a_corners_result', 'team_b_corners_result',
             'team_a_corners_result_fh', 'team_b_corners_result_fh',
             'team_a_corners_result_sh', 'team_b_corners_result_sh'], axis=1, inplace=True)
    # print(df.describe().to_markdown())
    df.dropna(inplace=True)
    print(len(df))
    # print(df.tail().to_markdown())
    df.to_csv('pre_cleaned_without1.csv', index=False)


def dataset_split():
    df = pd.read_csv('pre_cleaned_without1.csv')
    df['draw'] = np.where(((df['team_a_ft_result'] > 0) & (df['team_b_ft_result'] > 0)) | (df['team_a_ft_result'] == df['team_b_ft_result']), 0, 1)
    # df.drop(['team_a_o7_game_corners', 'team_b_o7_game_corners', 'team_a_o8_game_corners', 'team_b_o8_game_corners', 'team_a_o9_game_corners', 'team_b_o9_game_corners', 'team_a_o10_game_corners', 'team_b_o10_game_corners', 'team_a_avg_game_corners_fh', 'team_b_avg_game_corners_fh', 'team_a_o2_corners_fh', 'team_b_o2_corners_fh', 'team_a_o3_corners_fh', 'team_b_o3_corners_fh', 'team_a_o4_corners_fh', 'team_b_o4_corners_fh', 'team_a_avg_game_corners_sh', 'team_b_avg_game_corners_sh', 'team_a_o3_corners_sh', 'team_b_o3_corners_sh', 'team_a_o4_corners_sh', 'team_b_o4_corners_sh', 'team_a_o5_corners_sh', 'team_b_o5_corners_sh', 'team_a_avg_corners_for', 'team_b_avg_corners_for', 'team_a_avg_corners_ag', 'team_b_avg_corners_ag', 'team_a_o3_team_corners', 'team_b_o3_team_corners', 'team_a_o4_team_corners', 'team_b_o4_team_corners', 'team_a_o5_team_corners', 'team_b_o5_team_corners', 'team_a_o0_team_corners_fh', 'team_b_o0_team_corners_fh', 'team_a_o1_team_corners_fh', 'team_b_o1_team_corners_fh', 'team_a_o2_team_corners_fh', 'team_b_o2_team_corners_fh', 'team_a_o3_team_corners_fh', 'team_b_o3_team_corners_fh', 'team_a_o0_team_corners_sh', 'team_b_o0_team_corners_sh', 'team_a_o1_team_corners_sh', 'team_b_o1_team_corners_sh', 'team_a_o2_team_corners_sh', 'team_b_o2_team_corners_sh', 'team_a_o3_team_corners_sh', 'team_b_o3_team_corners_sh'], inplace=True, axis=1)
    train_dataset, validate_dataset = np.split(df, [int(.8*len(df))])
    print(len(train_dataset), len(validate_dataset))
    train_dataset.to_csv('train_dataset_without.csv', index=False)
    validate_dataset.to_csv('validate_dataset_without.csv', index=False)


read_scraped_data()
dataset_split()
df = pd.read_csv('train_dataset_without.csv')
validate = pd.read_csv('validate_dataset_without.csv')
df.drop(['team_a_name', 'team_b_name', 'team_a_ft_result', 'team_b_ft_result', 'team_a_ht_result', 'team_b_ht_result'], axis=1, inplace=True)
validate.drop(['team_a_name', 'team_b_name', 'team_a_ft_result', 'team_b_ft_result', 'team_a_ht_result', 'team_b_ht_result'], axis=1, inplace=True)
num_cols = ['team_a_pos', 'team_b_pos', 'team_a_won_perc', 'team_b_won_perc', 'team_a_lost_perc', 'team_b_lost_perc',
            'team_a_draw_perc', 'team_b_draw_perc', 'team_a_avg_game_goals', 'team_b_avg_game_goals', 'team_a_avg_goals_scored',
            'team_b_avg_goals_scored', 'team_a_avg_goals_conc', 'team_b_avg_goals_conc', 'team_a_clean_sheet', 'team_b_clean_sheet',
            'team_a_failed_to_score', 'team_b_failed_to_score', 'team_a_o1_team', 'team_b_o1_team', 'team_a_o1_team_ag', 'team_b_o1_team_ag',
            'team_a_btts', 'team_b_btts', 'team_a_btts_02', 'team_b_btts_02', 'team_a_o0', 'team_b_o0', 'team_a_o1', 'team_b_o1', 'team_a_02',
            'team_b_02', 'team_a_o3', 'team_b_o3', 'team_a_o4', 'team_b_o4', 'team_a_avg_game_corners', 'team_b_avg_game_corners', 'team_a_o7_game_corners', 'team_b_o7_game_corners',
            'team_a_o8_game_corners', 'team_b_o8_game_corners', 'team_a_o9_game_corners', 'team_b_o9_game_corners', 'team_a_o10_game_corners',
            'team_b_o10_game_corners', 'team_a_avg_game_corners_fh', 'team_b_avg_game_corners_fh', 'team_a_o2_corners_fh', 'team_b_o2_corners_fh',
            'team_a_o3_corners_fh', 'team_b_o3_corners_fh', 'team_a_o4_corners_fh', 'team_b_o4_corners_fh', 'team_a_avg_game_corners_sh',
            'team_b_avg_game_corners_sh', 'team_a_o3_corners_sh', 'team_b_o3_corners_sh', 'team_a_o4_corners_sh', 'team_b_o4_corners_sh',
            'team_a_o5_corners_sh', 'team_b_o5_corners_sh', 'team_a_avg_corners_for', 'team_b_avg_corners_for', 'team_a_avg_corners_ag',
            'team_b_avg_corners_ag', 'team_a_o3_team_corners', 'team_b_o3_team_corners', 'team_a_o4_team_corners', 'team_b_o4_team_corners',
            'team_a_o5_team_corners', 'team_b_o5_team_corners', 'team_a_o0_team_corners_fh', 'team_b_o0_team_corners_fh', 'team_a_o1_team_corners_fh',
            'team_b_o1_team_corners_fh', 'team_a_o2_team_corners_fh', 'team_b_o2_team_corners_fh', 'team_a_o3_team_corners_fh', 'team_b_o3_team_corners_fh',
            'team_a_o0_team_corners_sh', 'team_b_o0_team_corners_sh', 'team_a_o1_team_corners_sh', 'team_b_o1_team_corners_sh', 'team_a_o2_team_corners_sh',
            'team_b_o2_team_corners_sh', 'team_a_o3_team_corners_sh', 'team_b_o3_team_corners_sh',  'team_a_avg_goals_fh', 'team_b_avg_goals_fh',
            'team_a_avg_goals_fh_scored', 'team_b_avg_goals_fh_scored', 'team_a_avg_goals_fh_conc', 'team_b_avg_goals_fh_conc',
            'team_a_cs_fh', 'team_b_cs_fh', 'team_a_fts_fh', 'team_b_fts_fh', 'team_a_btts_fh', 'team_b_btts_fh', 'team_a_o0_fh',
            'team_b_o0_fh', 'team_a_o1_fh', 'team_b_o1_fh', 'team_a_avg_goals_sh', 'team_b_avg_goals_sh', 'team_a_avg_goals_sh_scored',
            'team_b_avg_goals_sh_scored', 'team_a_avg_goals_sh_conc', 'team_b_avg_goals_sh_conc', 'team_a_cs_sh', 'team_b_cs_sh',
            'team_a_fts_sh', 'team_b_fts_sh', 'team_a_btts_sh', 'team_b_btts_sh', 'team_a_o0_sh', 'team_b_o0_sh', 'team_a_o1_sh',
            'team_b_o1_sh']
 # , 'team_a_exp_goals', 'team_b_exp_goals', 'team_a_exp_goals_ag', 'team_b_exp_goals_ag', 'team_a_30_ht', 'team_b_30_ht', 'team_a_75_ft', 'team_b_75_ft']

# print(df.head().to_markdown())
# print(df.columns.tolist())
#
df_ready = df.copy()
scaler = RobustScaler()
df_ready[num_cols] = scaler.fit_transform(df[num_cols])
print(df_ready.groupby("draw").size())  # print number of draws
# using Synthetic Minority Oversampling Technique to upsample
X_train = df_ready.drop(["draw"], axis=1)
y_train = df_ready["draw"]
# print(X_train_smote.shape, Y_train_smote.shape)
# sm = SMOTE(random_state=42)
# X_train, y_train = sm.fit_resample(X_train_smote, Y_train_smote)

# validate dataset
X_test = validate.drop(["draw"], axis=1)
y_test = validate["draw"]

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)



# Building Decision Tree model
dtc = LGBMClassifier(random_state=0)
dtc.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

# Building Random Forest model
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB

# Building Naive Bayes model
nb = CatBoostClassifier(random_state=0)
nb.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier

# Building KNN model
# XGBOOST
# fit model no training data
knn = XGBClassifier(random_state=0)
knn.fit(X_train, y_train)

# Evaluate Model
knn_eval = evaluate_model(knn, X_test, y_test)

# Print result
print('XGB Classifier')
print('Accuracy:', knn_eval['acc'])
print('Precision:', knn_eval['prec'])
print('Recall:', knn_eval['rec'])
print('F1 Score:', knn_eval['f1'])
print('Cohens Kappa Score:', knn_eval['kappa'])
print('Area Under Curve:', knn_eval['auc'])
print("Mattew's Correlation Coefficient:", knn_eval['mcc'])
print('Optimal Threshold:', knn_eval['optimal_threshold'])
print('Confusion Matrix:\n', knn_eval['cm'])
print('Classification Report:\n', knn_eval['clas'])


# Evaluate Model
nb_eval = evaluate_model(nb, X_test, y_test)

# Print result
print('CatBoost Classifier')
print('Accuracy:', nb_eval['acc'])
print('Precision:', nb_eval['prec'])
print('Recall:', nb_eval['rec'])
print('F1 Score:', nb_eval['f1'])
print('Cohens Kappa Score:', nb_eval['kappa'])
print('Area Under Curve:', nb_eval['auc'])
print("Mattew's Correlation Coefficient", nb_eval['mcc'])
print('Optimal Threshold:', nb_eval['optimal_threshold'])
print('Confusion Matrix:\n', nb_eval['cm'])
print('Classification Report:\n', nb_eval['clas'])


# Evaluate Model
dtc_eval = evaluate_model(dtc, X_test, y_test)

# Print result
print('LGBM Classifier')
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Cohens Kappa Score:', dtc_eval['kappa'])
print('Area Under Curve:', dtc_eval['auc'])
print("Mattew's Correlation Coefficient", dtc_eval['mcc'])
print('Optimal Threshold:', dtc_eval['optimal_threshold'])
print('Confusion Matrix:\n', dtc_eval['cm'])
print('Classification Report:\n', dtc_eval['clas'])


# Evaluate Model
rf_eval = evaluate_model(rf, X_test, y_test)

# Print result
print('Random forest')
print('Accuracy:', rf_eval['acc'])
print('Precision:', rf_eval['prec'])
print('Recall:', rf_eval['rec'])
print('F1 Score:', rf_eval['f1'])
print('Cohens Kappa Score:', rf_eval['kappa'])
print('Area Under Curve:', rf_eval['auc'])
print("Mattew's Correlation Coefficient", rf_eval['mcc'])
print('Optimal Threshold:', rf_eval['optimal_threshold'])
print('Confusion Matrix:\n', rf_eval['cm'])
print('Classification Report:\n', rf_eval['clas'])

# # best mcc score
# def best_mcc_score(actual,prediction):
#     return metrics.fbeta_score(actual, prediction, beta=0.5)
#
# grid_scorer = metrics.make_scorer(best_mcc_score, greater_is_better=True)
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
# from numpy import mean
# # Building Imba SVM
# rf = SVC(gamma='scale', class_weight='balanced', probability=True, random_state=0)
# # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
# # scores = cross_val_score(rf, X_train, y_train, scoring=grid_scorer, cv=cv, n_jobs=-1)
# # print(f'mean F0.5: {mean(scores)}')
# rf.fit(X_train, y_train)
#
# # Evaluate Model
# rf_eval = evaluate_model(rf, X_test, y_test)
#
# # Print result
# print('Imba SVM')
# print('Accuracy:', rf_eval['acc'])
# print('Precision:', rf_eval['prec'])
# print('Recall:', rf_eval['rec'])
# print('F1 Score:', rf_eval['f1'])
# print('Cohens Kappa Score:', rf_eval['kappa'])
# print('Area Under Curve:', rf_eval['auc'])
# print("Mattew's Correlation Coefficient", rf_eval['mcc'])
# print('Optimal Threshold:', rf_eval['optimal_threshold'])
# print('Confusion Matrix:\n', rf_eval['cm'])
# print('Classification Report:\n', rf_eval['clas'])

import tensorflow as tf
tf.random.set_seed(42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train, y_train, epochs=100)

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

plt.plot(
    np.arange(1, 101),
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 101),
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, 101),
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 101),
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()

# Evaluate Model
rf_eval = evaluate_model(model, X_test, y_test)

# Print result
print('Random forest')
print('Accuracy:', rf_eval['acc'])
print('Precision:', rf_eval['prec'])
print('Recall:', rf_eval['rec'])
print('F1 Score:', rf_eval['f1'])
print('Cohens Kappa Score:', rf_eval['kappa'])
print('Area Under Curve:', rf_eval['auc'])
print("Mattew's Correlation Coefficient", rf_eval['mcc'])
print('Optimal Threshold:', rf_eval['optimal_threshold'])
print('Confusion Matrix:\n', rf_eval['cm'])
print('Classification Report:\n', rf_eval['clas'])
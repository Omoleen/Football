import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from skopt import BayesSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector, RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression, SGDClassifier
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import f_classif
from sklearn import metrics


def evaluate_model(model, x_test, y_test, boundary=0.5):
    from sklearn import metrics

    # Predict Test Data
    # y_pred = model.predict(x_test)
    y_pred = (model.predict_proba(x_test)[::,1] >= boundary).astype(int)
    # y_pred_proba = model.predict_proba(x_test)[::,1]
    # for i in y_pred_proba:
    #     if i<0.6:
    #         y_pred.append(0)
    #     else:
    #         y_pred.append(1)


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
            'fpr': fpr, 'tpr': tpr, 'auc': prauc, 'cm': cm, 'clas': clas, 'mcc': mcc, 'optimal_threshold': optimal_threshold}


def best_mcc_score(actual,prediction):
    return metrics.fbeta_score(actual, prediction, beta=0.5)
    # Precision recall AUC curve
    # return metrics.average_precision_score(y_test, prediction[:,1])  # when the positive class is the most important

grid_scorer = metrics.make_scorer(best_mcc_score, greater_is_better=True)


def select_features(X_train, y_train, X_test):
    # configure to select a subset of features

    grid_scorer = metrics.make_scorer(best_mcc_score, greater_is_better=True)
    fs = SelectFromModel(LGBMClassifier(random_state=0))
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

df = pd.read_csv('train1_without_xg_10.csv')
df.drop_duplicates(keep='first', inplace=True)
# print(df[df.isna().any(axis=1)])
df['team_a_avg_goals_sh_scored'] = df.team_a_avg_goals_sh_scored.abs()
df['team_a_avg_goals_sh_conc'] = df.team_a_avg_goals_sh_conc.abs()
df['team_b_avg_goals_sh_conc'] = df.team_b_avg_goals_sh_conc.abs()
df['team_a_corners_result_sh'] = df.team_a_corners_result_sh.abs()
df['team_b_corners_result_sh'] = df.team_b_corners_result_sh.abs()
df = df[df['team_a_pos'] <= 1.0].copy()
df = df[df['team_b_pos'] <= 1.0].copy()
df = df[~df['team_a_name'].str.contains(r'Res\.')].copy()
df = df[~df['team_b_name'].str.contains(r'Res\.')].copy()
df = df[~df['team_b_name'].str.contains(r'Reserve')].copy()
df = df[~df['team_a_name'].str.contains(r'Reserve')].copy()
# df = df[df['odds_team_b'] >= df['odds_team_a']].copy()
print(len(df))

df1 = df[df.isna().any(axis=1)].copy()
df1['odds_team_b'] = df1.odds_team_a
df1['odds_draw'] = df1.team_b_corners_result_sh
df1['odds_team_a'] = df1.team_a_corners_result_sh
df1['team_b_corners_result_sh'] = df1.team_b_corners_result_fh
df1['team_a_corners_result_sh'] = df1.team_a_corners_result_fh
df1['team_b_corners_result_fh'] = df1.team_b_corners_result
df1['team_a_corners_result_fh'] = df1.team_a_corners_result
df1['team_b_corners_result'] = df1.team_b_corners_result_fh + df1.team_b_corners_result_sh
df1['team_a_corners_result'] = df1.team_a_corners_result_fh + df1.team_a_corners_result_sh
print(df1.to_markdown())
df = pd.concat((df, df1), ignore_index=True).copy()
df.dropna(inplace=True)
# df['date'] = pd.to_datetime(df['date'])
# df = df.sort_values(by='date', ascending=True).copy()

print(df.describe().to_markdown())
print(len(df))
# print(df.tail().to_markdown())
# df['draw'] = np.where(((df['team_a_ft_result'] > 0) & (df['team_b_ft_result'] > 0)) | (df['team_a_ft_result'] == df['team_b_ft_result']), 0, 1)
df['draw'] = np.where((df['team_a_ht_result'] + df['team_b_ht_result'] > 0), 1, 0)
#
# print(df.columns.tolist())
df.drop(['team_a_corners_result', 'team_b_corners_result',
         'team_a_corners_result_fh', 'team_b_corners_result_fh',
         'team_a_corners_result_sh', 'team_b_corners_result_sh', 'team_a_ft_result',
         'team_b_ft_result', 'team_a_ht_result', 'team_b_ht_result', 'league', 'date', 'team_a_name', 'team_b_name',],
        axis=1, inplace=True)
# df.drop(['team_a_avg_game_corners', 'team_b_avg_game_corners',
#           'team_a_o7_game_corners', 'team_b_o7_game_corners', 'team_a_o8_game_corners', 'team_b_o8_game_corners',
#           'team_a_o9_game_corners', 'team_b_o9_game_corners', 'team_a_o10_game_corners', 'team_b_o10_game_corners',
#           'team_a_avg_game_corners_fh', 'team_b_avg_game_corners_fh', 'team_a_o2_corners_fh', 'team_b_o2_corners_fh',
#           'team_a_o3_corners_fh', 'team_b_o3_corners_fh', 'team_a_o4_corners_fh', 'team_b_o4_corners_fh',
#           'team_a_avg_game_corners_sh', 'team_b_avg_game_corners_sh', 'team_a_o3_corners_sh', 'team_b_o3_corners_sh',
#           'team_a_o4_corners_sh', 'team_b_o4_corners_sh', 'team_a_o5_corners_sh', 'team_b_o5_corners_sh',
#           'team_a_avg_corners_for', 'team_b_avg_corners_for', 'team_a_avg_corners_ag', 'team_b_avg_corners_ag',
#           'team_a_o3_team_corners', 'team_b_o3_team_corners', 'team_a_o4_team_corners', 'team_b_o4_team_corners',
#           'team_a_o5_team_corners', 'team_b_o5_team_corners', 'team_a_o0_team_corners_fh', 'team_b_o0_team_corners_fh',
#           'team_a_o1_team_corners_fh', 'team_b_o1_team_corners_fh', 'team_a_o2_team_corners_fh',
#           'team_b_o2_team_corners_fh', 'team_a_o3_team_corners_fh', 'team_b_o3_team_corners_fh',
#           'team_a_o0_team_corners_sh', 'team_b_o0_team_corners_sh', 'team_a_o1_team_corners_sh',
#           'team_b_o1_team_corners_sh', 'team_a_o2_team_corners_sh', 'team_b_o2_team_corners_sh',],
#         axis=1, inplace=True)
# df.drop(['team_a_pos', 'team_b_pos', 'team_a_won_perc', 'team_b_won_perc',
#          'team_a_lost_perc', 'team_b_lost_perc', 'team_a_draw_perc', 'team_b_draw_perc', 'team_a_avg_game_goals',
#          'team_b_avg_game_goals', 'team_a_avg_goals_scored', 'team_b_avg_goals_scored', 'team_a_avg_goals_conc',
#          'team_b_avg_goals_conc', 'team_a_clean_sheet', 'team_b_clean_sheet', 'team_a_failed_to_score',
#          'team_b_failed_to_score', 'team_a_o1_team', 'team_b_o1_team', 'team_a_o1_team_ag', 'team_b_o1_team_ag',
#          'team_a_btts', 'team_b_btts', 'team_a_btts_02', 'team_b_btts_02', 'team_a_o0', 'team_b_o0', 'team_a_o1',
#          'team_b_o1', 'team_a_02', 'team_b_02', 'team_a_o3', 'team_b_o3', 'team_a_o4', 'team_b_o4',
#          'team_a_avg_goals_fh', 'team_b_avg_goals_fh', 'team_a_avg_goals_fh_scored', 'team_b_avg_goals_fh_scored',
#          'team_a_avg_goals_fh_conc', 'team_b_avg_goals_fh_conc', 'team_a_cs_fh', 'team_b_cs_fh', 'team_a_fts_fh',
#          'team_b_fts_fh', 'team_a_btts_fh', 'team_b_btts_fh', 'team_a_o0_fh', 'team_b_o0_fh', 'team_a_o1_fh',
#          'team_b_o1_fh', 'team_a_avg_goals_sh', 'team_b_avg_goals_sh', 'team_a_avg_goals_sh_scored',
#          'team_b_avg_goals_sh_scored', 'team_a_avg_goals_sh_conc', 'team_b_avg_goals_sh_conc', 'team_a_cs_sh',
#          'team_b_cs_sh', 'team_a_fts_sh', 'team_b_fts_sh', 'team_a_btts_sh', 'team_b_btts_sh', 'team_a_o0_sh',
#          'team_b_o0_sh', 'team_a_o1_sh', 'team_b_o1_sh',
#          'odds_team_a', 'odds_draw', 'odds_team_b',])


X = df.drop(["draw"], axis=1)
Y = df["draw"]
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    shuffle=True,
                                                    test_size=0.2,
                                                    random_state=1)


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
print(y_train.value_counts())  # print number of draws
print(f'Test data: {y_test.value_counts()}')
# using Synthetic Minority Oversampling Technique to upsample
print(X_train.shape, y_train.shape)
sm = SMOTEENN(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)


# SELECTING BEST FEATURES
# X_train, X_test, fs = select_features(X_train, y_train, X_test)

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
knn_eval = evaluate_model(knn, X_test, y_test,)

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
nb_eval = evaluate_model(nb, X_test, y_test,)

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
dtc_eval = evaluate_model(dtc, X_test, y_test,)

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
rf_eval = evaluate_model(rf, X_test, y_test,)

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


# importance = rf.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.title('Random Forest')
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()
#
#
# importance = knn.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.title('XGBoost')
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()


# fit model no training data
lr = LogisticRegression(random_state=0, max_iter=10000)
lr.fit(X_train, y_train)

# Evaluate Model
knn_eval = evaluate_model(lr, X_test, y_test,)

# Print result
print('Logistic Regression')
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



# fit model no training data
# knn = GradientBoostingClassifier(random_state=0)
# knn.fit(X_train, y_train)
#
# # Evaluate Model
# knn_eval = evaluate_model(knn, X_test, y_test)
#
# # Print result
# print('Gradient Boosting Classifier')
# print('Accuracy:', knn_eval['acc'])
# print('Precision:', knn_eval['prec'])
# print('Recall:', knn_eval['rec'])
# print('F1 Score:', knn_eval['f1'])
# print('Cohens Kappa Score:', knn_eval['kappa'])
# print('Area Under Curve:', knn_eval['auc'])
# print("Mattew's Correlation Coefficient:", knn_eval['mcc'])
# print('Optimal Threshold:', knn_eval['optimal_threshold'])
# print('Confusion Matrix:\n', knn_eval['cm'])
# print('Classification Report:\n', knn_eval['clas'])




params = {
    'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
    'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
    'min_child_weight' : [ 1, 3, 5, 7 ],
    'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
}

rs_model=RandomizedSearchCV(XGBClassifier(random_state=0),param_distributions=params,n_iter=5, scoring='accuracy',n_jobs=-1,cv=5,verbose=3)

#model fitting
rs_model.fit(X_train,y_train)

#parameters selected
print(rs_model.best_estimator_)
# Select best model with best fit
best_grid = rs_model.best_estimator_

# # Evaluate Model
# best_grid_eval = evaluate_model(best_grid, X_test, y_test)

# Evaluate Model
rf_eval = evaluate_model(best_grid, X_test, y_test)

# Print result
print('XGBoost Randomized search')
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



# ('XGBoost', knn), ('LGBM', dtc),('Catboost', nb),('Logistic', lr), ('Random', rf)

knn = VotingClassifier(estimators=[('Catboost', nb), ('Random', rf)], voting='soft', weights=[1,1])
knn.fit(X_train, y_train)

# Evaluate Model
knn_eval = evaluate_model(knn, X_test, y_test, )

# Print result
print('Voting Classifier(soft)')
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


# CatBoost Classifier is th best for over 0 first half
# Threshold: 0.6
# Accuracy: 91%

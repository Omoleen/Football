import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    clas = metrics.classification_report(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
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
        temp_roc = metrics.matthews_corrcoef(y_test, predicted)
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

df = pd.read_csv('ml_fooball_data.csv')
# print(df.columns)
df.drop_duplicates(keep='first', inplace=True)
df.dropna(inplace=True)
# print(df.describe())
# print(df.isnull().sum())
# sns.heatmap(df.corr(), annot=True)
# plt.show()
# for column in df.columns:
df['team_a_corners_result_sh'] = df.team_a_corners_result_sh.abs()
df['draw'] = np.where(df['team_a_ft_result'] == df['team_b_ft_result'], 1, 0)
df.drop(['team_a_name', 'team_b_name', 'team_a_ft_result', 'team_b_ft_result', 'team_a_ht_result',
         'team_b_ht_result', 'team_a_corners_result', 'team_b_corners_result',
         'team_a_corners_result_fh', 'team_b_corners_result_fh', 'team_a_corners_result_sh', 'team_b_corners_result_sh'], axis=1, inplace=True)
print(df.describe())
# print(df.isnull().sum())
num_cols = df.columns
df_ready = df.copy()
scaler = RobustScaler()
df_ready[num_cols] = scaler.fit_transform(df[num_cols])
print(df_ready.groupby("draw").size())  # print number of draws
# df_ready['draw'].value_counts()

# using Synthetic Minority Oversampling Technique to upsample
X_train_smote = df_ready.drop(["draw"], axis=1)
Y_train_smote = df_ready["draw"]
print(X_train_smote.shape, Y_train_smote.shape)
sm = SMOTETomek(random_state=42)
X_train_res, Y_train_res = sm.fit_resample(X_train_smote, Y_train_smote)
print(X_train_res.shape, Y_train_res.shape)
X_train, X_test, y_train, y_test = train_test_split(X_train_res, Y_train_res,
                                                    shuffle=True,
                                                    test_size=0.2,
                                                    random_state=1)

# X_train, X_test, y_train, y_test = train_test_split(X_train_smote, Y_train_smote,
#                                                     shuffle=True,
#                                                     test_size=0.2,
#                                                     random_state=1)


# Show the Training and Testing Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)
#
#
from sklearn import tree

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



# plotting graph to compare algorithms
# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
dtc_score = [dtc_eval['acc'], dtc_eval['prec'], dtc_eval['rec'], dtc_eval['f1'], dtc_eval['kappa']]
rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1'], rf_eval['kappa']]
nb_score = [nb_eval['acc'], nb_eval['prec'], nb_eval['rec'], nb_eval['f1'], nb_eval['kappa']]
knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1'], knn_eval['kappa']]

## Set position of bar on X axis
r1 = np.arange(len(dtc_score))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

## Make the plot
ax1.bar(r1, dtc_score, width=barWidth, edgecolor='white', label='LGBM Classifier')
ax1.bar(r2, rf_score, width=barWidth, edgecolor='white', label='Random Forest')
ax1.bar(r3, nb_score, width=barWidth, edgecolor='white', label='CatBoost Classifier')
ax1.bar(r4, knn_score, width=barWidth, edgecolor='white', label='XGB Classifier')

## Configure x and y axis
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(dtc_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(dtc_eval['fpr'], dtc_eval['tpr'], label='LGBM Classifier, auc = {:0.5f}'.format(dtc_eval['auc']))
ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(rf_eval['auc']))
ax2.plot(nb_eval['fpr'], nb_eval['tpr'], label='CatBoost Classifier, auc = {:0.5f}'.format(nb_eval['auc']))
ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='XGB Classifier, auc = {:0.5f}'.format(knn_eval['auc']))

## Configure x and y axis
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

# plt.show()


# model optimization using cross validation for RANDOM FOREST
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer

# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [50, 80, 100],
    'max_features': [None],
    'min_samples_leaf': [50, 60, 70],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 300, 500, 750, 1000]
}

# best mcc score
def best_mcc_score(actual,prediction):
    return matthews_corrcoef(actual, prediction)

grid_scorer = make_scorer(best_mcc_score, greater_is_better=True)

# Create a base model
rf_grids = RandomForestClassifier(random_state=0)

# Initiate the grid search model
grid_search = GridSearchCV(estimator=rf_grids, param_grid=param_grid, scoring=grid_scorer,
                           cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

print('RANDOM FOREST GRID SEARCH:')
print(grid_search.best_params_)


# Select best model with best fit
best_grid = grid_search.best_estimator_

# Evaluate Model
best_grid_eval = evaluate_model(best_grid, X_test, y_test)

# Print result
print('Accuracy:', best_grid_eval['acc'])
print('Precision:', best_grid_eval['prec'])
print('Recall:', best_grid_eval['rec'])
print('F1 Score:', best_grid_eval['f1'])
print('Cohens Kappa Score:', best_grid_eval['kappa'])
print('Area Under Curve:', best_grid_eval['auc'])
print("Mattew's Correlation Coefficient:", best_grid_eval['mcc'])
print('Optimal Threshold:', best_grid_eval['optimal_threshold'])
print('Confusion Matrix:\n', best_grid_eval['cm'])
print('Classification Report:\n', best_grid_eval['clas'])


# plotting graph to compare Random fores and random forest after cross validation
# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1'], rf_eval['kappa']]
best_grid_score = [best_grid_eval['acc'], best_grid_eval['prec'], best_grid_eval['rec'], best_grid_eval['f1'], best_grid_eval['kappa']]

## Set position of bar on X axis
r1 = np.arange(len(rf_score))
r2 = [x + barWidth for x in r1]

## Make the plot
ax1.bar(r1, rf_score, width=barWidth, edgecolor='white', label='Random Forest (Base Line)')
ax1.bar(r2, best_grid_score, width=barWidth, edgecolor='white', label='Random Forest (Optimized)')

## Add xticks on the middle of the group bars
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 0.5) for r in range(len(dtc_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
# ax1.set_ylim(0, 1)

## Create legend & Show graphic
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(rf_eval['auc']))
ax2.plot(best_grid_eval['fpr'], best_grid_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(best_grid_eval['auc']))

ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')
ax2.legend(loc=4)

plt.show()

print('Change of {:0.2f}% on accuracy.'.format(100 * ((best_grid_eval['acc'] - rf_eval['acc']) / rf_eval['acc'])))
print('Change of {:0.2f}% on precision.'.format(100 * ((best_grid_eval['prec'] - rf_eval['prec']) / rf_eval['prec'])))
print('Change of {:0.2f}% on recall.'.format(100 * ((best_grid_eval['rec'] - rf_eval['rec']) / rf_eval['rec'])))
print('Change of {:0.2f}% on F1 score.'.format(100 * ((best_grid_eval['f1'] - rf_eval['f1']) / rf_eval['f1'])))
print('Change of {:0.2f}% on Kappa score.'.format(100 * ((best_grid_eval['kappa'] - rf_eval['kappa']) / rf_eval['kappa'])))
print('Change of {:0.2f}% on AUC.'.format(100 * ((best_grid_eval['auc'] - rf_eval['auc']) / rf_eval['auc'])))
print("Change of {:0.2f}% on Mattew's Correlation Coefficient.".format(100 * ((best_grid_eval['mcc'] - rf_eval['mcc']) / rf_eval['mcc'])))


from joblib import dump, load

# Saving model
dump(best_grid, 'random_forest_gridsearch.joblib')


# model optimization using cross validation for XGBOOST
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer

# Create the parameter grid based on the results of random search
estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'subsample': [.7],
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

# best mcc score
def best_mcc_score(actual,prediction):
    return matthews_corrcoef(actual, prediction)

grid_scorer = make_scorer(best_mcc_score, greater_is_better=True)

# Create a base model
rf_grids = XGBClassifier(random_state=0)

# Initiate the grid search model
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = grid_scorer,
    n_jobs = 10,
    cv = 10,
    verbose=2
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

print('XGBOOST grid search')
print(grid_search.best_params_)


# Select best model with best fit
best_grid = grid_search.best_estimator_

# Evaluate Model
best_grid_eval = evaluate_model(best_grid, X_test, y_test)

# Print result
print('Accuracy:', best_grid_eval['acc'])
print('Precision:', best_grid_eval['prec'])
print('Recall:', best_grid_eval['rec'])
print('F1 Score:', best_grid_eval['f1'])
print('Cohens Kappa Score:', best_grid_eval['kappa'])
print('Area Under Curve:', best_grid_eval['auc'])
print("Mattew's Correlation Coefficient:", best_grid_eval['mcc'])
print('Optimal Threshold:', best_grid_eval['optimal_threshold'])
print('Confusion Matrix:\n', best_grid_eval['cm'])
print('Classification Report:\n', best_grid_eval['clas'])


# plotting graph to compare Random forest and random forest after cross validation
# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1'], rf_eval['kappa']]
best_grid_score = [best_grid_eval['acc'], best_grid_eval['prec'], best_grid_eval['rec'], best_grid_eval['f1'], best_grid_eval['kappa']]

## Set position of bar on X axis
r1 = np.arange(len(rf_score))
r2 = [x + barWidth for x in r1]

## Make the plot
ax1.bar(r1, rf_score, width=barWidth, edgecolor='white', label='XGBOOST (Base Line)')
ax1.bar(r2, best_grid_score, width=barWidth, edgecolor='white', label='XGBOOST (Optimized)')

## Add xticks on the middle of the group bars
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 0.5) for r in range(len(dtc_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
# ax1.set_ylim(0, 1)

## Create legend & Show graphic
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='XGBOOST, auc = {:0.5f}'.format(rf_eval['auc']))
ax2.plot(best_grid_eval['fpr'], best_grid_eval['tpr'], label='XGBOOST, auc = {:0.5f}'.format(best_grid_eval['auc']))

ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')
ax2.legend(loc=4)

plt.show()

print('Change of {:0.2f}% on accuracy.'.format(100 * ((best_grid_eval['acc'] - rf_eval['acc']) / rf_eval['acc'])))
print('Change of {:0.2f}% on precision.'.format(100 * ((best_grid_eval['prec'] - rf_eval['prec']) / rf_eval['prec'])))
print('Change of {:0.2f}% on recall.'.format(100 * ((best_grid_eval['rec'] - rf_eval['rec']) / rf_eval['rec'])))
print('Change of {:0.2f}% on F1 score.'.format(100 * ((best_grid_eval['f1'] - rf_eval['f1']) / rf_eval['f1'])))
print('Change of {:0.2f}% on Kappa score.'.format(100 * ((best_grid_eval['kappa'] - rf_eval['kappa']) / rf_eval['kappa'])))
print('Change of {:0.2f}% on AUC.'.format(100 * ((best_grid_eval['auc'] - rf_eval['auc']) / rf_eval['auc'])))
print("Change of {:0.2f}% on Mattew's Correlation Coefficient.".format(100 * ((best_grid_eval['mcc'] - rf_eval['mcc']) / rf_eval['mcc'])))


from joblib import dump, load

# Saving model
dump(best_grid, 'XGBOOST_gridsearch.joblib')

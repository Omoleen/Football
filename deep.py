import tensorflow as tf
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
tf.random.set_seed(42)

def evaluate_model(model, x_test, y_test, boundary=0.5):
    from sklearn import metrics

    # Predict Test Data
    # y_pred = model.predict(x_test)
    y_pred = (model.predict(x_test)[::,0] >= boundary).astype(int)
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



df = pd.read_csv('train1_without_xg_10.csv')

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
# df = df[df['odds_team_b'] <= 4.0].copy()
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
#          'team_a_o7_game_corners', 'team_b_o7_game_corners', 'team_a_o8_game_corners', 'team_b_o8_game_corners',
#          'team_a_o9_game_corners', 'team_b_o9_game_corners', 'team_a_o10_game_corners', 'team_b_o10_game_corners',
#          'team_a_avg_game_corners_fh', 'team_b_avg_game_corners_fh', 'team_a_o2_corners_fh', 'team_b_o2_corners_fh',
#          'team_a_o3_corners_fh', 'team_b_o3_corners_fh', 'team_a_o4_corners_fh', 'team_b_o4_corners_fh',
#          'team_a_avg_game_corners_sh', 'team_b_avg_game_corners_sh', 'team_a_o3_corners_sh', 'team_b_o3_corners_sh',
#          'team_a_o4_corners_sh', 'team_b_o4_corners_sh', 'team_a_o5_corners_sh', 'team_b_o5_corners_sh',
#          'team_a_avg_corners_for', 'team_b_avg_corners_for', 'team_a_avg_corners_ag', 'team_b_avg_corners_ag',
#          'team_a_o3_team_corners', 'team_b_o3_team_corners', 'team_a_o4_team_corners', 'team_b_o4_team_corners',
#          'team_a_o5_team_corners', 'team_b_o5_team_corners', 'team_a_o0_team_corners_fh', 'team_b_o0_team_corners_fh',
#          'team_a_o1_team_corners_fh', 'team_b_o1_team_corners_fh', 'team_a_o2_team_corners_fh',
#          'team_b_o2_team_corners_fh', 'team_a_o3_team_corners_fh', 'team_b_o3_team_corners_fh',
#          'team_a_o0_team_corners_sh', 'team_b_o0_team_corners_sh', 'team_a_o1_team_corners_sh',
#          'team_b_o1_team_corners_sh', 'team_a_o2_team_corners_sh', 'team_b_o2_team_corners_sh',],
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

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(256, activation='relu', name='layer1'),
        tf.keras.layers.Dense(128, activation='relu', name='layer2'),
        tf.keras.layers.Dense(256, activation='relu', name='layer3'),
        tf.keras.layers.Dense(128, activation='relu', name='layer4'),
        tf.keras.layers.Dense(50, activation='relu', name='layer5'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='layer6'),
    ]
)

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ],
)

history = model.fit(X_train, y_train, epochs=100)

# Evaluate Model
knn_eval = evaluate_model(model, X_test, y_test)

# Print result
print('Tensorflow')
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
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 101),
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()

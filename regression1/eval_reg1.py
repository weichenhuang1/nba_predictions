import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import math

model = keras.models.load_model("models/multreg1")

raw_stats = pd.read_csv("model_1_stats/2023_model_1_stats.csv")
raw_games = pd.read_csv("game_scores/2022_2023_nba_games.csv")

stats = raw_stats[['team', 'off_rtg', 'def_rtg', 'efg', 'reb_diff']]
game_outcomes = raw_games[['team1', 'team2', 'team1_score', 'team2_score']]
game_outcomes['point_diff'] = game_outcomes['team1_score'] - game_outcomes['team2_score']

for index, row in game_outcomes.iterrows():
    if index % 2 == 1:
        temp = game_outcomes.at[index, 'team1']
        game_outcomes.at[index, 'team1'] = game_outcomes.at[index, 'team2']
        game_outcomes.at[index, 'team2'] = temp
        temp2 = game_outcomes.at[index, 'team1_score']
        game_outcomes.at[index, 'team1_score'] = game_outcomes.at[index, 'team2_score']
        game_outcomes.at[index, 'team2_score'] = temp2
        game_outcomes.at[index, 'point_diff'] = -1 * game_outcomes.at[index, 'point_diff']

train_and_eval = pd.DataFrame(columns = ["off_rtg", 'def_rtg', 'efg', 'reb_diff', 'off_rtg2', 'def_rtg2', 'efg2', 'reb_diff2', 'point_diff'])
for index, row in game_outcomes.iterrows():
    for x, row2 in stats.iterrows():
        if game_outcomes.at[index, 'team1'] in stats.loc[x, 'team']:
            train_and_eval.at[index, "off_rtg"] = stats.loc[x, 'off_rtg']
            train_and_eval.at[index, "def_rtg"] = stats.loc[x, 'def_rtg']
            train_and_eval.at[index, "efg"] = stats.loc[x, 'efg']
            train_and_eval.at[index, "reb_diff"] = stats.loc[x, 'reb_diff']
        if game_outcomes.at[index, 'team2'] in stats.loc[x, 'team']:
            train_and_eval.at[index, "off_rtg2"] = stats.loc[x, 'off_rtg']
            train_and_eval.at[index, "def_rtg2"] = stats.loc[x, 'def_rtg']
            train_and_eval.at[index, "efg2"] = stats.loc[x, 'efg']
            train_and_eval.at[index, "reb_diff2"] = stats.loc[x, 'reb_diff']
train_and_eval['point_diff'] = game_outcomes['point_diff']

eval = train_and_eval.iloc[-1187:].copy()

eval_y = eval.pop('point_diff')

eval  = np.asarray(eval).astype('float32')

eval_y  = np.asarray(eval_y).astype('float32')

correct_counter = 0
total_counter = 0

predictions = model.predict(eval).tolist()

verify_df = pd.DataFrame({'Predicted Value': predictions, 'Actual Value': eval_y})

for index, row in verify_df.iterrows():
    if verify_df.at[index, 'Predicted Value'][0] > 0 and verify_df.at[index, 'Actual Value'] > 0:
        correct_counter += 1
        total_counter += 1
    elif verify_df.at[index, 'Predicted Value'][0] < 0 and verify_df.at[index, 'Actual Value'] < 0:
        correct_counter += 1
        total_counter += 1
    else:
        total_counter += 1
accuracy = float(correct_counter)/float(total_counter)
print(accuracy)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

#get raw stats from old dfs
raw_stats = pd.read_csv("model_1_stats/2023_model_1_stats.csv")
raw_games = pd.read_csv("game_scores/2022_2023_nba_games.csv")

stats = raw_stats[['team', 'off_rtg', 'def_rtg', 'efg', 'reb_diff']]
game_outcomes = raw_games[['team1', 'team2', 'output']]

new_game_outcomes = game_outcomes 

#create new df, one entry for each game
for index, row in new_game_outcomes.iterrows():
    if index % 2 == 1:
        temp = new_game_outcomes.at[index, 'team1']
        new_game_outcomes.at[index, 'team1'] = new_game_outcomes.at[index, 'team2']
        new_game_outcomes.at[index, 'team2'] = temp
        new_game_outcomes.at[index, 'output'] = 1


#format dataset
train_and_eval = pd.DataFrame(columns = ["off_rtg", 'def_rtg', 'efg', 'reb_diff', 'off_rtg2', 'def_rtg2', 'efg2', 'reb_diff2', 'output'])
for index, row in new_game_outcomes.iterrows():
    for x, row2 in stats.iterrows():
        if new_game_outcomes.at[index, 'team1'] in stats.loc[x, 'team']:
            train_and_eval.at[index, "off_rtg"] = stats.loc[x, 'off_rtg']
            train_and_eval.at[index, "def_rtg"] = stats.loc[x, 'def_rtg']
            train_and_eval.at[index, "efg"] = stats.loc[x, 'efg']
            train_and_eval.at[index, "reb_diff"] = stats.loc[x, 'reb_diff']
        if new_game_outcomes.at[index, 'team2'] in stats.loc[x, 'team']:
            train_and_eval.at[index, "off_rtg2"] = stats.loc[x, 'off_rtg']
            train_and_eval.at[index, "def_rtg2"] = stats.loc[x, 'def_rtg']
            train_and_eval.at[index, "efg2"] = stats.loc[x, 'efg']
            train_and_eval.at[index, "reb_diff2"] = stats.loc[x, 'reb_diff']
train_and_eval['output'] = new_game_outcomes['output']

train = train_and_eval.iloc[:950].copy()
eval = train_and_eval.iloc[-237:].copy()

train_y = train.pop('output')
eval_y = eval.pop('output')

train = np.asarray(train).astype('float32')
eval  = np.asarray(eval).astype('float32')

train_y = np.asarray(train_y).astype('float32')
eval_y  = np.asarray(eval_y).astype('float32')

winning_team = ["team1, team2"]


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (8,)), 
    keras.layers.Dense(256, activation = 'relu', name = "HL1"), 
    keras.layers.Dense(128, activation = 'relu', name = "HL2"),
    keras.layers.Dense(128, activation = 'relu', name = "HL3"),
    keras.layers.Dense(256, activation = 'relu', name = "HL4"),
    keras.layers.Dense(2, activation = 'softmax') 
])

model.compile(optimizer = 'adam',     
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']) 
model.fit(train, train_y, epochs = 200)

test_loss, test_acc = model.evaluate(eval, eval_y, verbose = 1)
print('Test accuracy:', test_acc)

predictions = model.predict(eval)
print(predictions)

model.save("models/classification1")

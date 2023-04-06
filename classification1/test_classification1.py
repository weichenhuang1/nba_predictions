import pandas as pd
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("models/classification1")

# Define the team names
team1 = input("What is your first team? (3 letter code) ")
team2 = input("What is your second team? (3 letter code) ")

# Load the statistics for both teams
raw_stats = pd.read_csv("model_1_stats/2023_model_1_stats.csv")
stats = raw_stats[['team', 'off_rtg', 'def_rtg', 'efg', 'reb_diff']]

test_row = pd.DataFrame(columns = ["off_rtg", 'def_rtg', 'efg', 'reb_diff', 'off_rtg2', 'def_rtg2', 'efg2', 'reb_diff2'])
for i in range(1):
    for x, row2 in stats.iterrows():
        if team1 in stats.loc[x, 'team']:
            test_row.at[i, "off_rtg"] = stats.loc[x, 'off_rtg']
            test_row.at[i, "def_rtg"] = stats.loc[x, 'def_rtg']
            test_row.at[i, "efg"] = stats.loc[x, 'efg']
            test_row.at[i, "reb_diff"] = stats.loc[x, 'reb_diff']
        if team2 in stats.loc[x, 'team']:
            test_row.at[i, "off_rtg2"] = stats.loc[x, 'off_rtg']
            test_row.at[i, "def_rtg2"] = stats.loc[x, 'def_rtg']
            test_row.at[i, "efg2"] = stats.loc[x, 'efg']
            test_row.at[i, "reb_diff2"] = stats.loc[x, 'reb_diff']

test_row = np.asarray(test_row).astype('float32')

# Predict the outcome and probability for team 1
predictions = model.predict(test_row)
team1_probability = predictions[0][0]
print(f"The probability of {team1} winning is {team1_probability:.2f}")
print(f"The probability of {team2} winning is {1- team1_probability:.2f}")
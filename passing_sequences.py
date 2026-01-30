import json
import joblib
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import RegressorChain
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from functions_store import jsonRequest, importJson
from ngboost import NGBRegressor
from ngboost.distns import Normal

choiceModel = joblib.load('data/choice_predictor.pkl')

# Show ALL columns (no matter how many)
pd.set_option('display.max_columns', None)

# Prevent wrapping to the next line (make the "screen" infinitely wide)
pd.set_option('display.width', 1000)

def getPasses(leagueId, seasonId):
    with requests.Session() as session:
        usedProcess = {}
        matchIds = []
        validTypes = ['cross', 'pass', 'goal', 'ball-movement', 'miss', 'ball-touch', 'post', 'throw-in']
        switch = True
        count = 0
        while switch:
            matches = jsonRequest(
                f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/last/{count}",
                3 * 24, defaultSession=session)
            for match in matches['events']:
                if match['status']['code'] == 100:
                    matchIds.append(match['id'])
            if matches['hasNextPage']:
                count += 1
            else:
                switch = False
        count = 0
        for matchId in matchIds:
            incidents = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/incidents", cache_ttl=0, defaultSession=session)['incidents']
            for incident in incidents:
                if incident['incidentType'] == 'goal' and 'footballPassingNetworkAction' in incident:
                    if 'incidentClass' in incident:
                        if incident['incidentClass'] != 'ownGoal':
                            sameIncident = False
                            for action in incident['footballPassingNetworkAction']:
                                if action['eventType'] in validTypes:
                                    count += 1
                                    usedProcess[str(count)] = action
                                    if sameIncident and usedProcess[str(count-1)]['eventType'] in ['ball-movement', 'cross', 'pass', 'ball-touch', 'throw-in']:
                                        if usedProcess[str(count-1)]['eventType'] == 'ball-movement':
                                            usedProcess[str(count-1)]['movementEnd'] = action['playerCoordinates']
                                        elif usedProcess[str(count-1)]['eventType'] in ['cross', 'pass', 'ball-touch', 'throw-in'] and 'passEndCoordinates' not in usedProcess[str(count-1)]:
                                            usedProcess[str(count-1)]['passEndCoordinates'] = action['playerCoordinates']
                                    sameIncident = True
        return usedProcess

def predictNextPass(df, xModel, yModel, temperature=0.75):
    xDist = xModel.pred_dist(df[['xStart', 'yStart']])
    mu_x = xDist.loc[0]
    sigma_x = xDist.scale[0]
    simulated_x = np.random.normal(mu_x, sigma_x * temperature)
    while simulated_x < 0 or simulated_x > 100:
        simulated_x = np.random.normal(mu_x, sigma_x * temperature)
    df['xEnd'] = simulated_x
    yDist = yModel.pred_dist(df[['xStart', 'yStart', 'xEnd']])
    mu_y = yDist.loc[0]
    sigma_y = yDist.scale[0]
    simulated_y = np.random.normal(mu_y, sigma_y * temperature)
    while simulated_y < 0 or simulated_y > 100:
        simulated_y = np.random.normal(mu_y, sigma_y * temperature)
    return simulated_x, simulated_y

filepath = "data/ligueSequences2526.json"
"""
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(getPasses(34, 77356), f, indent=4)
"""
df = pd.concat(
    (pd.DataFrame(importJson("data/premSequences2526.json")).T,
     pd.DataFrame(importJson("data/premSequences2425.json")).T,
     pd.DataFrame(importJson("data/FASequences2526.json")).T,
     pd.DataFrame(importJson("data/FASequences2425.json")).T,
     pd.DataFrame(importJson("data/laligaSequences2526.json")).T,
     pd.DataFrame(importJson("data/laligaSequences2425.json")).T,
     pd.DataFrame(importJson("data/serieASequences2526.json")).T,
     pd.DataFrame(importJson("data/serieASequences2425.json")).T,
     pd.DataFrame(importJson("data/bundeSequences2526.json")).T,
     pd.DataFrame(importJson("data/bundeSequences2425.json")).T,
     pd.DataFrame(importJson("data/ligueSequences2526.json")).T,
     pd.DataFrame(importJson("data/ligueSequences2425.json")).T,
     pd.DataFrame(importJson("data/UCLSequences2526.json")).T),
    ignore_index=True
)
#df = pd.DataFrame(importJson(filepath)).T

playerNextMoveData = df

targetValues = ['goal', 'miss', 'post']
otherValues = ['cross', 'pass', 'ball-touch', 'throw-in']

playerNextMoveData['eventType'] = playerNextMoveData['eventType'].replace(otherValues, 0)
playerNextMoveData['eventType'] = playerNextMoveData['eventType'].replace(targetValues, 1)
playerNextMoveData['eventType'] = playerNextMoveData['eventType'].replace('ball-movement', 2)

passPredictionData = playerNextMoveData[playerNextMoveData['eventType'] == 0]
shotPredictionData = playerNextMoveData[playerNextMoveData['eventType'] == 1]
movementPredictionData = playerNextMoveData[playerNextMoveData['eventType'] == 2]

playerNextMoveData = playerNextMoveData[['eventType', 'playerCoordinates', 'isHome']]
playerNextMoveData['x'] = playerNextMoveData['playerCoordinates'].str['x']
playerNextMoveData['x'] = np.where(playerNextMoveData['isHome'] == False, 100 - playerNextMoveData['x'], playerNextMoveData['x'])
playerNextMoveData['y'] = playerNextMoveData['playerCoordinates'].str['y']
playerNextMoveData = playerNextMoveData.drop(columns=['playerCoordinates'])
print(playerNextMoveData)

plt.figure(figsize=(12, 8)) # Make the pitch relatively large

# 'hue' is the parameter that determines colour based on a column
# We cast to .astype(str) to ensure it uses discrete colours, not a gradient
sns.scatterplot(
    data=playerNextMoveData,
    x='x',
    y='y',
    hue=playerNextMoveData['eventType'].astype(str),
    palette='viridis',  # 'viridis', 'deep', or 'Set1' are good choices
    s=50  # size of dots
)
plt.show()

X_move = playerNextMoveData.drop(columns=['eventType', 'isHome'])
y_move = playerNextMoveData['eventType']

moveModel = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')

moveModel.fit(X_move, y_move)

passPredictionData = passPredictionData[['playerCoordinates', 'passEndCoordinates', 'isHome']]
passPredictionData['xStart'] = passPredictionData['playerCoordinates'].str['x']
passPredictionData['yStart'] = passPredictionData['playerCoordinates'].str['y']
passPredictionData['xEnd'] = passPredictionData['passEndCoordinates'].str['x']
passPredictionData['yEnd'] = passPredictionData['passEndCoordinates'].str['y']
passPredictionData['xStart'] = np.where(passPredictionData['isHome'] == False, 100 - passPredictionData['xStart'], passPredictionData['xStart'])
passPredictionData['xEnd'] = np.where(passPredictionData['isHome'] == False, 100 - passPredictionData['xEnd'], passPredictionData['xEnd'])
passPredictionData = passPredictionData.drop(columns=['playerCoordinates', 'passEndCoordinates'])
print(passPredictionData[passPredictionData[['xStart', 'yStart', 'xEnd', 'yEnd']].isna().any(axis=1)])
passPredictionData = passPredictionData.dropna(subset=['xStart', 'yStart', 'xEnd', 'yEnd'])

X_pass = passPredictionData.drop(columns=['xEnd', 'yEnd', 'isHome'])
y_pass = passPredictionData.drop(columns=['xStart', 'yStart', 'isHome'])
X_pass_x = passPredictionData.drop(columns=['xEnd', 'yEnd', 'isHome'])
y_pass_x = passPredictionData.drop(columns=['xStart', 'yStart', 'yEnd', 'isHome'])
X_pass_y = passPredictionData.drop(columns=['yEnd', 'isHome'])
y_pass_y = passPredictionData.drop(columns=['xStart', 'yStart', 'xEnd', 'isHome'])

passModel = RegressorChain(XGBRegressor(objective='reg:squarederror'), order=[0, 1])
model_x = NGBRegressor(Dist=Normal, verbose=False)
model_x.fit(X_pass_x, y_pass_x)

model_y = NGBRegressor(Dist=Normal, verbose=False)
model_y.fit(X_pass_y, y_pass_y)

passModel.fit(X_pass, y_pass)

shotPredictionData = shotPredictionData[['playerCoordinates', 'isHome', 'goalShotCoordinates']]
shotPredictionData['x'] = shotPredictionData['playerCoordinates'].str['x']
shotPredictionData['y'] = shotPredictionData['playerCoordinates'].str['y']
shotPredictionData['x'] = np.where(shotPredictionData['isHome'] == False, 100 - shotPredictionData['x'], shotPredictionData['x'])
shotPredictionData['goal'] = shotPredictionData['goalShotCoordinates'].str['y']


print(movementPredictionData.columns)
movementPredictionData = movementPredictionData[['playerCoordinates', 'movementEnd', 'isHome']]
movementPredictionData['xStart'] = movementPredictionData['playerCoordinates'].str['x']
movementPredictionData['yStart'] = movementPredictionData['playerCoordinates'].str['y']
movementPredictionData['xEnd'] = movementPredictionData['movementEnd'].str['x']
movementPredictionData['yEnd'] = movementPredictionData['movementEnd'].str['y']
movementPredictionData['xStart'] = np.where(movementPredictionData['isHome'] == False, 100 - movementPredictionData['xStart'], movementPredictionData['xStart'])
movementPredictionData['xEnd'] = np.where(movementPredictionData['isHome'] == False, 100 - movementPredictionData['xEnd'], movementPredictionData['xEnd'])
movementPredictionData = movementPredictionData.drop(columns=['playerCoordinates', 'movementEnd'])

X_dribble = movementPredictionData.drop(columns=['xEnd', 'yEnd', 'isHome'])
y_dribble = movementPredictionData.drop(columns=['xStart', 'yStart', 'isHome'])

dribbleModel = RegressorChain(XGBRegressor(objective='reg:squarederror'), order=[0, 1])

dribbleModel.fit(X_dribble, y_dribble)


joblib.dump(moveModel, 'data/choice_predictor.pkl')

joblib.dump(passModel, 'data/pass_predictor.pkl')

joblib.dump(dribbleModel, 'data/movement_predictor.pkl')

joblib.dump(model_x, 'data/x_predictor.pkl')
joblib.dump(model_y, 'data/y_predictor.pkl')


def iterativePredict():
    inputX = input("Select your X coordinate: ")
    inputY = input("Select your Y coordinate: ")

    def nextChoice(x, y):
        y_pred = moveModel.predict_proba(pd.DataFrame({'1': {'x': float(x), 'y': float(y)}}).T)

        raw_probs = y_pred[0]

        probabilities = raw_probs / raw_probs.sum()


        options = ['Pass', 'Shot', 'Dribble']
        print(f"Probs: {options[0]}:{probabilities[0]:.2f} {options[1]}:{probabilities[1]:.2f} {options[2]}:{probabilities[2]:.2f}")  # readable printing


        choice = np.random.choice(options, p=probabilities)
        return choice

    def passDestination(x, y):
        y_pred = passModel.predict(pd.DataFrame({'1': {'xStart': float(x), 'yStart': float(y)}}).T)
        predX, predY = predictNextPass(pd.DataFrame({'1': {'xStart': float(x), 'yStart': float(y)}}).T, model_x, model_y)
        return predX, predY

    def dribbleDestination(x, y):
        y_pred = dribbleModel.predict(pd.DataFrame({'1': {'xStart': float(x), 'yStart': float(y)}}).T)
        return y_pred[0]

    sequence = [((int(inputX),int(inputY)), 'yellow')]
    switch = True
    while switch:
        choice = nextChoice(inputX, inputY)
        if choice == 'Shot':
            sequence.append(((100,50), 'red'))
            print("GOLAZOOOO!!!")
            switch = False
        elif choice == 'Pass':
            inputX, inputY = passDestination(inputX, inputY)
            sequence.append(((int(inputX),int(inputY)), 'green'))
            print(f"{choice} to ({inputX}, {inputY})")
        elif choice == 'Dribble':
            inputX, inputY = dribbleDestination(inputX, inputY)
            sequence.append(((int(inputX),int(inputY)), 'blue'))
            print(f"{choice} to ({inputX}, {inputY})")
    return sequence


def plot_possession_chain(data):
    # Separate the coordinates and the "Move Types" (colors) for easier handling
    coords = [item[0] for item in data]  # [(10,30), (33,14), ...]
    move_colors = [item[1] for item in data]  # ['yellow', 'blue', 'green'...]

    plt.figure(figsize=(12, 8))

    # --- 1. Draw the Connecting Lines ---
    # We iterate from 0 to the second-to-last point
    for i in range(len(coords) - 1):
        start_point = coords[i]
        end_point = coords[i + 1]

        # The color of the line is determined by the destination node's tuple
        # e.g., line to ((33, 14), 'blue') should be blue
        line_color = move_colors[i + 1]

        plt.plot([start_point[0], end_point[0]],
                 [start_point[1], end_point[1]],
                 color=line_color,
                 linewidth=3,
                 zorder=1)  # zorder=1 ensures lines stay BEHIND the dots

    # --- 2. Define Dot Colors ---
    # Logic: First is Yellow, Last is Red, everything else is Black
    num_points = len(coords)
    dot_colors = ['yellow'] + ['black'] * (num_points - 2) + ['red']

    # --- 3. Draw the Dots ---
    # Zip separates x and y for the scatter plot
    x_vals = [c[0] for c in coords]
    y_vals = [c[1] for c in coords]

    plt.scatter(x_vals, y_vals, c=dot_colors, s=150, edgecolors='white', zorder=2)

    # --- 4. Make it look like a pitch ---
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Possession Sequence Visualisation")
    # Invert Y axis if your pitch data uses 0 as top-left (standard in some formats)
    # plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.show()


# Run the function
plot_possession_chain(iterativePredict())

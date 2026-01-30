from safe_requests import safe_request
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
import shap
import datetime
from scipy.optimize import linear_sum_assignment
import math
import itertools

key = {
    "FB": 0,
    "CB": 1,
    "DM": 2,
    "WM": 3,
    "CM": 4,
    "W": 5,
    "AM": 6,
    "ST": 7
}

statCategories = ['accurateCross', 'accurateLongBalls', 'accurateOppositionHalfPasses', 'accurateOwnHalfPasses',
                  'accuratePass', 'aerialLost', 'aerialWon', 'ballRecovery', 'bigChanceCreated', 'bigChanceMissed',
                  'blockedScoringAttempt', 'challengeLost', 'clearanceOffLine', 'dispossessed', 'duelLost',
                  'duelWon',
                  'errorLeadToAGoal', 'errorLeadToAShot', 'expectedAssists', 'expectedGoals',
                  'expectedGoalsOnTarget',
                  'fouls', 'goalAssist', 'goals', 'hitWoodwork', 'interceptionWon', 'keyPass', 'lastManTackle',
                  'minutesPlayed', 'onTargetScoringAttempt', 'outfielderBlock', 'ownGoals', 'penaltyConceded',
                  'penaltyMiss', 'penaltyWon', 'position', 'possessionLostCtrl', 'rating', 'ratingVersions',
                  'shotOffTarget', 'totalClearance', 'totalContest', 'totalCross', 'totalLongBalls', 'totalOffside',
                  'totalOppositionHalfPasses', 'totalOwnHalfPasses', 'totalPass', 'totalShots', 'totalTackle',
                  'touches', 'unsuccessfulTouch', 'wasFouled', 'wonContest', 'wonTackle']


def jsonRequest(url, cache_ttl=None, defaultSession=None):
    if cache_ttl is None:
        resp = safe_request(url, defaultSession=defaultSession)  # uses default CACHE_TTL
    else:
        cache_ttl = cache_ttl * 3600
        resp = safe_request(url, cache_ttl=cache_ttl, defaultSession=defaultSession)
    return resp.json()

def importAndTransformDict(file_path):

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    for x in data.copy().keys():
        if 'minutesPlayed' not in data[x] or data[x]['position'] == 'GK':
            data.pop(x)

    for n in data.keys():
        for m in statCategories:
            if m not in data[n]:
                data[n][m] = 0

    df = pd.DataFrame(data).T
    return df

def MLModel():
    combinedDF = pd.concat(
        (importAndTransformDict("data/playerStats2223.json"), importAndTransformDict("data/playerStats2324.json"), importAndTransformDict("data/playerStats2425.json")), ignore_index=True)

    X = combinedDF.drop(columns=['position', 'ratingVersions'])
    X[list(X.columns)] = combinedDF[list(X.columns)].apply(pd.to_numeric, errors='coerce')

    y_position = combinedDF['position']

    y = y_position.map(key)
    y = y.apply(pd.to_numeric, errors='coerce')

    model = XGBClassifier()
    model.fit(X, y)

    return model, X

def matchProbs(matchId, model, X):
    lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups")

    homeLineups = lineups['home']['players']
    awayLineups = lineups['away']['players']

    homePlayerStats = []

    count = 0
    for player in homeLineups:
        if player['position'] != 'G' and count < 10:
            count += 1
            homePlayerStats.append(player['statistics'])

    awayPlayerStats = []

    count = 0
    for player in awayLineups:
        if player['position'] != 'G' and count < 10:
            count += 1
            awayPlayerStats.append(player['statistics'])

    for n in range(len(homePlayerStats)):
        for m in statCategories:
            if m not in homePlayerStats[n]:
                homePlayerStats[n][m] = 0

    for n in range(len(awayPlayerStats)):
        for m in statCategories:
            if m not in awayPlayerStats[n]:
                awayPlayerStats[n][m] = 0

    columns = list(X.columns)

    homePlayerStats = pd.DataFrame(homePlayerStats)
    homePlayerStats = homePlayerStats[columns]
    homePlayerStats = homePlayerStats.apply(pd.to_numeric, errors='coerce')

    awayPlayerStats = pd.DataFrame(awayPlayerStats)
    awayPlayerStats = awayPlayerStats[columns]
    awayPlayerStats = awayPlayerStats.apply(pd.to_numeric, errors='coerce')

    homeProb = model.predict_proba(homePlayerStats)
    awayProb = model.predict_proba(awayPlayerStats)
    return homeProb, awayProb

def formation_likelihood(prob_matrix, formation, positions, return_mapping=False):
    """
    Improved two-stage formation likelihood:
    1. Locks in players whose top predicted positions match positions in the formation,
       prioritizing higher probabilities (not list order).
    2. Uses Hungarian algorithm to optimally assign remaining players to remaining positions.
    3. Returns total absolute probability and optional mapping.
    """

    n_players = prob_matrix.shape[0]
    slots = list(formation)
    position_indices = [positions.index(pos) for pos in slots]

    # --- Stage 1: Identify overlaps (but sort them by confidence first) ---
    player_top_pos = np.argmax(prob_matrix, axis=1)
    top_probs = np.max(prob_matrix, axis=1)

    # Sort players by how confident the model is about their top position
    player_order = sorted(range(n_players), key=lambda i: top_probs[i], reverse=True)

    locked_players = []
    locked_positions = []
    remaining_players = list(range(n_players))
    remaining_slots = list(range(len(slots)))

    for player_idx in player_order:
        top_pos = positions[player_top_pos[player_idx]]
        if top_pos in slots:
            # Find an available formation slot for that position
            for slot_idx in remaining_slots:
                if slots[slot_idx] == top_pos:
                    locked_players.append(player_idx)
                    locked_positions.append(slot_idx)
                    remaining_players.remove(player_idx)
                    remaining_slots.remove(slot_idx)
                    break  # move to next player once locked

    # --- Stage 2: Optimal assignment for remaining players/positions ---
    assigned_pairs = []
    if remaining_players and remaining_slots:
        submatrix = prob_matrix[np.ix_(remaining_players, [position_indices[i] for i in remaining_slots])]
        cost_matrix = 1 - submatrix  # higher probability = lower cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_pairs = list(zip(
            [remaining_players[r] for r in row_ind],
            [remaining_slots[c] for c in col_ind]
        ))

    # Combine overlaps and assignments
    all_pairs = list(zip(locked_players, locked_positions)) + assigned_pairs

    # --- Stage 3: Compute total probability ---
    total_prob = 1.0
    for player_idx, slot_idx in all_pairs:
        pos_index = positions.index(slots[slot_idx])
        prob = prob_matrix[player_idx, pos_index]
        total_prob *= prob

    # --- Stage 4: Optional mapping output ---
    if return_mapping:
        mapping = [
            (f"Player {p+1}", slots[s], float(prob_matrix[p, positions.index(slots[s])]))
            for p, s in all_pairs
        ]
        return total_prob, mapping

    return total_prob


def round_sig(x, sig=3):
    """
    Round to a given number of significant figures and
    display in fixed-point decimal notation (no scientific notation).
    """
    if x == 0:
        return "0"
    # round numerically to significant figures
    rounded = round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))
    # format as fixed-point (avoiding scientific notation)
    formatted = f"{rounded:.{sig+6}f}".rstrip("0").rstrip(".")
    # ensure there's always a 0 before the decimal if number < 1
    if formatted.startswith("."):
        formatted = "0" + formatted
    return formatted


def scale_ratings_to_10(ratings):
    """
    Scales a list or array of player ratings to the range [1, 10].
    """
    ratings = np.array(ratings, dtype=float)
    min_r, max_r = np.min(ratings), np.max(ratings)

    if min_r == max_r:
        return np.full_like(ratings, 5)

    scaled = 1 + 9 * (ratings - min_r) / (max_r - min_r)
    return scaled

def importJson(file_path):

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data

def getFullTeamName(name):
    data = jsonRequest(f"http://www.sofascore.com/api/v1/search/teams?q={name}&page=0")['results']
    count = 0
    for i in data:
        if count < 10 and 'gender' in i['entity']:
            count += 1
            print(f"{count}. {i['entity']['name']} ({i['entity']['gender']})")
    choice = input("Which number would you like to choose? ")
    return data[int(choice) - 1]['entity']['name']

def zeroDivide(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator
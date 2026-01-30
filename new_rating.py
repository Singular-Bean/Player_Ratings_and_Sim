from matplotlib.style.core import available
from collections import Counter
from functions_store import *
import datetime
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
import requests
import socket
import statsmodels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def log_ipv6(hostname):
    try:
        # socket.AF_INET6 forces it to look for IPv6 addresses only
        # The result is a list of tuples, so we grab the first one
        result = socket.getaddrinfo(hostname, None, socket.AF_INET6)

        # The address is the 0th element of the 4th item in the tuple
        ipv6_address = result[0][4][0]
        print(f"IPv6 for {hostname}: {ipv6_address}")
        return ipv6_address

    except socket.gaierror:
        print(f"Could not resolve IPv6 for {hostname} (it might only have IPv4)")
    except IndexError:
        print(f"No IPv6 addresses found for {hostname}")


# --- Usage Examples ---

# 1. Get IPv6 for a website
log_ipv6("sofascore.com")

# 2. Get IPv6 for YOUR own computer
my_hostname = socket.gethostname()
log_ipv6(my_hostname)

pd.set_option('future.no_silent_downcasting', True)

defaultColumns = ['totalPass', 'accuratePass', 'totalLongBalls', 'accurateOwnHalfPasses',
                  'totalOwnHalfPasses', 'accurateOppositionHalfPasses',
                  'totalOppositionHalfPasses', 'totalCross', 'aerialLost', 'aerialWon',
                  'duelLost', 'duelWon', 'wonContest', 'totalClearance',
                  'interceptionWon', 'ballRecovery', 'wasFouled', 'fouls',
                  'minutesPlayed', 'touches', 'possessionLostCtrl',
                  'expectedGoals', 'expectedGoalsOnTarget', 'expectedAssists', 'keyPass',
                  'totalShots', 'isSub', 'yellowCard', 'redCard',
                  'penaltyScored', 'overperform', 'lostContest', 'contestSuccess',
                  'position', 'accurateLongBalls', 'challengeLost', 'bigChanceCreated',
                  'totalTackle', 'wonTackle', 'outfielderBlock', 'ownGoals',
                  'dispossessed', 'unsuccessfulTouch', 'shotOffTarget',
                  'blockedScoringAttempt', 'accurateCross', 'bigChanceMissed',
                  'onTargetScoringAttempt', 'totalOffside', 'goals',
                  'clearanceOffLine', 'hitWoodwork', 'penaltyWon', 'penaltyConceded',
                  'lastManTackle', 'errorLeadToAShot', 'penaltyMiss']



positionsKey = {
    'D': 0,
    'M': 1,
    'F': 2
}

def deleteSeason(leagueId, seasonId):
    folderPath = "cache/"
    matchIds = []
    count = 1
    while safe_request(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/{count}").status_code == 200:
        matches = jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/{count}")['events']
        for match in matches:
            if str(match['id']) not in matchIds:
                matchIds.append(str(match['id']))
        count += 1
    for fileName in os.listdir(folderPath):
        if any(targetString in fileName for targetString in matchIds):
            filePath = os.path.join(folderPath, fileName)
            try:
                os.remove(filePath)
                print(f"Deleted: {fileName}")
            except Exception as e:
                print(f"Error deleting {fileName}: {e}")


def isOverperform(oddsData, home=True):
    data = oddsData
    for odds in data:
        if odds['marketName'] == "Full time":
            matchOdds = odds['choices']
            break
    total = 0
    for odd in matchOdds:
        odd['percentageValue'] = 1 / (
                1 + int(odd['fractionalValue'].split("/")[0]) / int(odd['fractionalValue'].split("/")[1]))
        total += odd['percentageValue']
    for odd in matchOdds:
        odd['percentageValue'] /= total
    homeOdds = matchOdds[0]['percentageValue']
    drawOdds = matchOdds[1]['percentageValue']
    awayOdds = matchOdds[2]['percentageValue']
    if home:
        if homeOdds > awayOdds:
            favourite = True
        else:
            favourite = False
        if (matchOdds[0]['winning'] and favourite == False) or (matchOdds[1]['winning'] and favourite == False):
            return 1, favourite
        elif matchOdds[0]['winning'] and favourite:
            return 1, favourite
        else:
            return 0, favourite
    else:
        if homeOdds < awayOdds:
            favourite = True
        else:
            favourite = False
        if (matchOdds[2]['winning'] and favourite == False) or (matchOdds[1]['winning'] and favourite == False):
            return 1, favourite
        elif matchOdds[2]['winning'] and favourite:
            return 1, favourite
        else:
            return 0, favourite

def individualOverperform(player, favourite, incidents, home=True):
    playerId = player['player']['id']
    fullTimeScore = [incidents[0]['homeScore'], incidents[0]['awayScore']]
    subbedScore = None
    looking = 0
    for incident in incidents:
        if incident['incidentType'] == 'substitution' and looking == 0:
            if incident['playerOut']['id'] == playerId:
                looking = 1
        elif incident['incidentType'] == 'goal' and looking == 1:
            subbedScore = [incident['homeScore'], incident['awayScore']]
            break
    if subbedScore == None and looking == 0:
        subbedScore = fullTimeScore
    elif subbedScore == None and looking == 1:
        subbedScore = [0, 0]
    if home:
        if favourite:
            if subbedScore[0] > subbedScore[1]:
                player['statistics']['overperform'] = 1
            else:
                player['statistics']['overperform'] = 0
        else:
            if subbedScore[0] >= subbedScore[1]:
                player['statistics']['overperform'] = 1
            else:
                player['statistics']['overperform'] = 0
    else:
        if favourite:
            if subbedScore[1] > subbedScore[0]:
                player['statistics']['overperform'] = 1
            else:
                player['statistics']['overperform'] = 0
        else:
            if subbedScore[1] >= subbedScore[0]:
                player['statistics']['overperform'] = 1
            else:
                player['statistics']['overperform'] = 0
    return player

def isSubYellowRedPenalty(playerId, incidents):
    isSub = 0
    isYellow = 0
    isRed = 0
    penaltyScored = 0
    for incident in incidents:
        if incident['incidentType'] == 'substitution':
            if 'playerIn' in incident:
                if incident['playerIn']['id'] == playerId:
                    isSub = 1
        elif incident['incidentType'] == 'card' and 'player' in incident:
            if incident['incidentClass'] == 'yellow':
                if incident['player']['id'] == playerId:
                    isYellow = 1
            elif (incident['incidentClass'] == 'yellowRed' or incident[
                'incidentClass'] == 'red') and 'player' in incident:
                if incident['player']['id'] == playerId:
                    isRed = 1
        if incident['incidentType'] == 'goal':
            if incident['incidentClass'] == 'penalty':
                if incident['player']['id'] == playerId:
                    penaltyScored += 1
    return isSub, isYellow, isRed, penaltyScored


def convertStats(stats):
    forbiddenColumns = ['rating', 'ratingVersions', 'statisticsType', 'goalAssist']
    columns = ['totalPass', 'totalLongBalls', 'totalOwnHalfPasses', 'totalOppositionHalfPasses', 'totalCross',
               'totalShots', 'totalTackle', 'totalContest']
    columnSuccess = ['accuratePass', 'accurateLongBalls', 'accurateOwnHalfPasses', 'accurateOppositionHalfPasses',
                     'accurateCross', 'onTargetScoringAttempt', 'wonTackle', 'wonContest']
    columnFail = ['inaccuratePass', 'inaccurateLongBalls', 'inaccurateOwnHalfPasses', 'inaccurateOppositionHalfPasses',
                  'inaccurateCross', 'offTargetScoringAttempt', 'lostTackle', 'lostContest']
    successRates = ['passSuccess', 'longBallSuccess', 'ownHalfPassSuccess', 'oppositionHalfPassSuccess', 'crossSuccess',
                    'onTargetSuccess', 'tackleSuccess', 'contestSuccess']
    keys = list(stats.keys())
    for key in keys:
        if key in forbiddenColumns:
            stats.pop(key)

    for x in range(len(columns)):
        if columns[x] in stats:
            if columnSuccess[x] in stats:
                stats[columnFail[x]] = stats[columns[x]] - stats[columnSuccess[x]]
                stats[successRates[x]] = zeroDivide(stats[columnSuccess[x]], stats[columns[x]])
            else:
                stats[columnFail[x]] = stats[columns[x]]
                stats[successRates[x]] = 0
            stats.pop(columns[x])
    if 'errorLeadToAGoal' in stats:
        if 'errorLeadToAShot' in stats:
            stats['errorLeadToAShot'] += stats['errorLeadToAGoal']
        else:
            stats['errorLeadToAShot'] = stats['errorLeadToAGoal']
        stats.pop('errorLeadToAGoal')
    return stats

def convertStatsPer90(stats):
    forbiddenColumns = ['rating', 'ratingVersions', 'statisticsType', 'goalAssist']
    columns = ['totalPass', 'totalLongBalls', 'totalOwnHalfPasses', 'totalOppositionHalfPasses', 'totalCross',
               'totalShots', 'totalTackle', 'totalContest']
    columnSuccess = ['accuratePass', 'accurateLongBalls', 'accurateOwnHalfPasses', 'accurateOppositionHalfPasses',
                     'accurateCross', 'onTargetScoringAttempt', 'wonTackle', 'wonContest']
    columnFail = ['inaccuratePass', 'inaccurateLongBalls', 'inaccurateOwnHalfPasses', 'inaccurateOppositionHalfPasses',
                  'inaccurateCross', 'offTargetScoringAttempt', 'lostTackle', 'lostContest']
    successRates = ['passSuccess', 'longBallSuccess', 'ownHalfPassSuccess', 'oppositionHalfPassSuccess', 'crossSuccess',
                    'onTargetSuccess', 'tackleSuccess', 'contestSuccess']
    dummies = ['isSub', 'yellowCard', 'redCard', 'overperform']
    keys = list(stats.keys())
    for key in keys:
        if key in forbiddenColumns:
            stats.pop(key)

    for x in range(len(columns)):
        if columns[x] in stats:
            if columnSuccess[x] in stats:
                stats[columnFail[x]] = stats[columns[x]] - stats[columnSuccess[x]]
                stats[successRates[x]] = zeroDivide(stats[columnSuccess[x]], stats[columns[x]])
            else:
                stats[columnFail[x]] = stats[columns[x]]
                stats[successRates[x]] = 0
            stats.pop(columns[x])
    if 'errorLeadToAGoal' in stats:
        if 'errorLeadToAShot' in stats:
            stats['errorLeadToAShot'] += stats['errorLeadToAGoal']
        else:
            stats['errorLeadToAShot'] = stats['errorLeadToAGoal']
        stats.pop('errorLeadToAGoal')
    if 'minutesPlayed' in stats:
        for key in stats.keys():
            if key not in dummies and key not in successRates and key != 'position' and key != 'minutesPlayed':
                stats[key] = (stats[key] / stats['minutesPlayed'])*90
        stats['minutesPlayed'] = 90
    return stats

def getSeasonStatList(leagueId, seasonId):
    with requests.Session() as session:
        stats = {}
        switch = True
        count = 0
        matchIds = []
        while switch:
            getGames = jsonRequest(
                f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/last/{count}")
            games = getGames['events']
            for game in games:
                if game['status']['code'] == 100 and safe_request(f"http://www.sofascore.com/api/v1/event/{game['id']}/lineups", cache_ttl=0, defaultSession=session).status_code == 200 and safe_request(f"http://www.sofascore.com/api/v1/event/{game['id']}/odds/1/all", cache_ttl=0, defaultSession=session).status_code == 200:
                    matchIds.append(game['id'])
            count += 1
            switch = getGames['hasNextPage']
        for matchId in matchIds:
            lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", cache_ttl=0, defaultSession=session)
            incidents = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/incidents", cache_ttl=0, defaultSession=session)['incidents']
            odds = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/odds/1/all", cache_ttl=0, defaultSession=session)['markets']
            print(matchId)
            homeOverperform, homeFav = isOverperform(odds)
            awayOverperform, awayFav = isOverperform(odds, home=False)
            for x in lineups['home']['players']:
                if 'statistics' in x:
                    x = lineups['home']['players'].index(x)
                    player = lineups['home']['players'][x]
                    playerId = player['player']['id']
                    player['statistics']['isSub'], player['statistics']['yellowCard'], player['statistics']['redCard'], \
                    player['statistics']['penaltyScored'] = isSubYellowRedPenalty(playerId, incidents)
                    player['statistics']['overperform'] = homeOverperform
                    player = individualOverperform(player, homeFav, incidents)
                    player['statistics'] = convertStats(player['statistics'])
                    playerMatchId = f"{playerId}-{matchId}"
                    position = player['position']
                    if position != 'G' and 'minutesPlayed' in player['statistics']:
                        if player['statistics']['minutesPlayed'] >= 20:
                            player['statistics']['position'] = position
                            stats[playerMatchId] = player['statistics']
            for x in lineups['away']['players']:
                if 'statistics' in x:
                    x = lineups['away']['players'].index(x)
                    player = lineups['away']['players'][x]
                    playerId = player['player']['id']
                    player['statistics']['isSub'], player['statistics']['yellowCard'], player['statistics']['redCard'], \
                    player['statistics']['penaltyScored'] = isSubYellowRedPenalty(playerId, incidents)
                    player['statistics']['overperform'] = awayOverperform
                    player = individualOverperform(player, awayFav, incidents, home=False)
                    player['statistics'] = convertStats(player['statistics'])
                    playerMatchId = f"{playerId}-{matchId}"
                    position = player['position']
                    if position != 'G' and 'minutesPlayed' in player['statistics']:
                        if player['statistics']['minutesPlayed'] >= 20:
                            player['statistics']['position'] = position
                            stats[playerMatchId] = player['statistics']
        return stats


filepath = 'data/playerStatsPrem2526.json'


if filepath.endswith('Prem2223.json'):
    leagueId = 17
    seasonId = 41886
elif filepath.endswith('Prem2324.json'):
    leagueId = 17
    seasonId = 52186
elif filepath.endswith('Prem2425.json'):
    leagueId = 17
    seasonId = 61627
elif filepath.endswith('Prem2526.json'):
    leagueId = 17
    seasonId = 76986
elif filepath.endswith('Laliga2223.json'):
    leagueId = 8
    seasonId = 42409
elif filepath.endswith('Laliga2324.json'):
    leagueId = 8
    seasonId = 52376
elif filepath.endswith('Laliga2425.json'):
    leagueId = 8
    seasonId = 61643
elif filepath.endswith('SerieA2223.json'):
    leagueId = 23
    seasonId = 42415
elif filepath.endswith('SerieA2324.json'):
    leagueId = 23
    seasonId = 52760
elif filepath.endswith('SerieA2425.json'):
    leagueId = 23
    seasonId = 63515
elif filepath.endswith('Bunde2223.json'):
    leagueId = 35
    seasonId = 42268
elif filepath.endswith('Bunde2324.json'):
    leagueId = 35
    seasonId = 52608
elif filepath.endswith('Bunde2425.json'):
    leagueId = 35
    seasonId = 63516
elif filepath.endswith('Ligue2223.json'):
    leagueId = 34
    seasonId = 42273
elif filepath.endswith('Ligue2324.json'):
    leagueId = 34
    seasonId = 52571
elif filepath.endswith('Ligue2425.json'):
    leagueId = 34
    seasonId = 61736
else:
    exit()

#with open(filepath, "w", encoding="utf-8") as f:
#    json.dump(getSeasonStatList(leagueId, seasonId), f, indent=4)

# with open(filepath, "w", encoding="utf-8") as f:
#   json.dump(getSeasonStatList2526(leagueId, seasonId), f, indent=4)

# with open(filepath, "w", encoding="utf-8") as f:
#    json.dump(getSeasonStatListPer90(leagueId, seasonId), f, indent=4)


combinedDF = pd.concat(
    (pd.DataFrame(importJson("data/playerStatsPrem2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsPrem2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsPrem2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsLaliga2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsLaliga2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsLaliga2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsSerieA2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsSerieA2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsSerieA2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsBunde2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsBunde2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsBunde2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsLigue2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsLigue2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsLigue2425.json")).T.fillna(0).infer_objects(copy=False)),
    ignore_index=True
)
"""
combinedDF = pd.concat(
    (pd.DataFrame(importJson("data/playerStatsTestPrem2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestPrem2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestPrem2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestLaliga2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestLaliga2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestLaliga2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestSerieA2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestSerieA2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestSerieA2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestBunde2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestBunde2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestBunde2425.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestLigue2223.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestLigue2324.json")).T.fillna(0).infer_objects(copy=False),
     pd.DataFrame(importJson("data/playerStatsTestLigue2425.json")).T.fillna(0).infer_objects(copy=False)),
    ignore_index=True
)

# 1. Generate the confusion matrix
# 'favourite' acts as our "prediction" and 'overperform' as the "actual"
cm = confusion_matrix(combinedDF['overperform'], combinedDF['favourite'])

# 2. Plot the matrix for better visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])

# Use a clean plot style
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax)

plt.title('Confusion Matrix: Favourite vs Overperformance')
plt.xlabel('Is Favourite')
plt.ylabel('Did Overperform')
plt.show()

# 3. Optional: Print the raw numbers
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Not Fav & Didn't Overperform): {tn}")
print(f"False Positives (Fav & Didn't Overperform): {fp}")
print(f"False Negatives (Not Fav & Did Overperform): {fn}")
print(f"True Positives (Fav & Did Overperform): {tp}")
"""
includedColumns = list(combinedDF.columns)
#includedColumns.remove('favourite')

includeGoals = input("Remove goals? (y/n): ") == 'n'


if includeGoals == False:
    includedColumns.remove('goals')
    includedColumns.remove('penaltyScored')
    defaultColumns.remove('goals')
    defaultColumns.remove('penaltyScored')

def getTournamentStatList(leagueId, seasonId, round):
    usableRounds = []
    matchIds = []
    if safe_request(
            f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/rounds").status_code == 200:
        roundInfo = jsonRequest(
            f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/rounds")
    else:
        roundInfo = None
    currentRound = roundInfo['currentRound']
    maxPosition = roundInfo['rounds'].index(currentRound)
    if maxPosition < 3:
        return None
    validPositions = range(0, maxPosition)
    position = roundInfo['rounds'].index(round)
    if position == 0:
        testingRounds = [0, 1, 2]
    elif position == maxPosition:
        testingRounds = [maxPosition - 2, maxPosition - 1, maxPosition]
    else:
        testingRounds = [position - 1, position, position + 1]
    for x in testingRounds:
        if 'prefix' in roundInfo['rounds'][x]:
            usableRounds.append(
                (roundInfo['rounds'][x]['prefix'], roundInfo['rounds'][x]['slug'], roundInfo['rounds'][x]['round'], 2))
        elif 'slug' in roundInfo['rounds'][x]:
            usableRounds.append((roundInfo['rounds'][x]['slug'], roundInfo['rounds'][x]['round'], 1))
        else:
            usableRounds.append((roundInfo['rounds'][x]['round'], 0))
    for round in usableRounds:
        if round[-1] == 0:
            roundEvents = jsonRequest(
                f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/{round[0]}")[
                'events']
            for event in roundEvents:
                matchIds.append(event['id'])
        elif round[-1] == 1:
            roundEvents = jsonRequest(
                f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/{round[1]}/slug/{round[0]}")[
                'events']
            for event in roundEvents:
                matchIds.append(event['id'])
        elif round[-1] == 2:
            roundEvents = jsonRequest(
                f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/{round[2]}/slug/{round[1]}/prefix/{round[0]}")[
                'events']
            for event in roundEvents:
                matchIds.append(event['id'])
    safeMatchIds = []
    for matchId in matchIds:
        if jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}")['event']['status']['code'] == 100 and safe_request(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0).status_code == 200:
            safeMatchIds.append(matchId)
    stats = {}
    for matchId in safeMatchIds:
        lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0)
        incidents = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/incidents", 0)['incidents']
        for homePlayer in lineups['home']['players']:
            playerMatchId = f"{homePlayer['player']['id']}-{matchId}"
            position = homePlayer['player']['position']
            if position != 'G' and 'statistics' in homePlayer:
                homePlayer['statistics']['isSub'], homePlayer['statistics']['yellowCard'], homePlayer['statistics'][
                    'redCard'], homePlayer['statistics']['penaltyScored'] = isSubYellowRedPenalty(homePlayer['player']['id'],
                                                                                           incidents)
                homePlayer['statistics']['position'] = position
                homePlayer['statistics'] = convertStats(homePlayer['statistics'])
                stats[playerMatchId] = homePlayer['statistics']
        for awayPlayer in lineups['away']['players']:
            playerMatchId = f"{awayPlayer['player']['id']}-{matchId}"
            position = awayPlayer['player']['position']
            if position != 'G' and 'statistics' in awayPlayer:
                awayPlayer['statistics']['isSub'], awayPlayer['statistics']['yellowCard'], awayPlayer['statistics'][
                    'redCard'], awayPlayer['statistics']['penaltyScored'] = isSubYellowRedPenalty(awayPlayer['player']['id'],
                                                                                           incidents)
                awayPlayer['statistics']['position'] = position
                awayPlayer['statistics'] = convertStats(awayPlayer['statistics'])
                stats[playerMatchId] = awayPlayer['statistics']
    statList = pd.DataFrame(stats).T.columns.values.tolist()
    newColumns = statList.copy()
    for x in statList:
        if x not in includedColumns:
            newColumns.remove(x)
    print(f"Columns not included:")
    for x in includedColumns:
        if x not in statList:
            print(x)
    return newColumns


def trainModel(dfOG, matchColumns=defaultColumns):
    df = dfOG.copy()
    testingDf = pd.DataFrame(importJson("data/playerStatsPrem2526.json")).T.fillna(0).infer_objects(copy=False)

    if 'position' in matchColumns:
        df['position'] = df['position'].map(positionsKey)
        testingDf['position'] = testingDf['position'].map(positionsKey)

    X = df[matchColumns]
    y = df['overperform']

    X_test = testingDf[matchColumns]
    y_test = testingDf['overperform']

    model = XGBClassifier(eval_metric='logloss', random_state=42)

    model.fit(X, y)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    ratings = [x[1] * 10 for x in y_prob]

    sns.kdeplot(ratings, fill=True)
    plt.title("KDE of values")
    plt.show()

    # 4. Simple Feature Importance
    # Since it's just one model now, we can access this attribute directly
    importances = model.feature_importances_

    # Optional: Print them nicely to verify
    # for col, score in zip(matchColumns, importances):
    #     print(f"{col}: {score*100:.2f}%")

    return model, X, X_test


def getRating(model, playerRow):
    rowDict = playerRow.to_dict(orient='records')[0]
    if 'penaltyScored' in rowDict:
        playerRowScoredPen = rowDict.copy()
        playerRowMissedPen = rowDict.copy()
        playerRowNotTaken = rowDict.copy()
        scoredMissedList = [rowDict['penaltyScored'], rowDict['penaltyMiss']]
        pensTaken = scoredMissedList[0] + scoredMissedList[1]
        if pensTaken > 0:
            playerRowNotTaken['goals'] -= playerRowNotTaken['penaltyScored']
            playerRowNotTaken['penaltyScored'] = 0
            playerRowNotTaken['penaltyMiss'] = 0
            playerRowScoredPen['penaltyScored'] += playerRowScoredPen['penaltyMiss']
            playerRowScoredPen['goals'] += playerRowScoredPen['penaltyMiss']
            playerRowScoredPen['penaltyMiss'] = 0
            playerRowMissedPen['penaltyMiss'] += playerRowMissedPen['penaltyScored']
            playerRowMissedPen['goals'] -= playerRowMissedPen['penaltyScored']
            playerRowMissedPen['penaltyScored'] = 0

            playerRowScoredPenDf = pd.DataFrame(playerRowScoredPen, index=[0])
            playerRowMissedPenDf = pd.DataFrame(playerRowMissedPen, index=[0])
            playerRowNotTakenDf = pd.DataFrame(playerRowNotTaken, index=[0])

            scoredProbs = model.predict_proba(playerRowScoredPenDf)[:, 1]

            missedProbs = model.predict_proba(playerRowMissedPenDf)[:, 1]

            notTakenProbs = model.predict_proba(playerRowNotTakenDf)[:, 1]

            probs = model.predict_proba(playerRow)[:, 1]

            xG = 0.79**pensTaken
            xRating = (xG*scoredProbs)+((1-xG)*missedProbs)
            addedValue = probs - xRating
            rating = notTakenProbs + addedValue
            return rating[0]*10

    estimate = model.predict_proba(playerRow)[:, 1]
    return estimate[0]*10


def getMatchAndExplainPlayer(defaultMatchId=None, model=None, categories=None, X=None, X_test=None):
    if defaultMatchId is None:
        TeamA = input("\nName the home team: ")
        homeTeam = getFullTeamName(TeamA)
        TeamB = input("\nName the away team: ")
        awayTeam = getFullTeamName(TeamB)

        searches = jsonRequest(f"http://www.sofascore.com/api/v1/search/events?q={homeTeam}%20{awayTeam}&page=0")[
            'results']

        matchList = []

        for results in searches:
            result = results['entity']
            epoch = result['startTimestamp']
            dt_object = datetime.datetime.fromtimestamp(epoch)
            human_readable_time = dt_object.strftime('%d-%m-%Y')
            if result['homeTeam']['name'] == homeTeam and result['awayTeam']['name'] == awayTeam and epoch > 1656630000:
                matchList.append((result['name'], result['id'], human_readable_time, epoch))

        matchList = sorted(matchList, key=lambda x: x[3], reverse=True)

        for x in range(len(matchList)):
            print(f"{x + 1}. {matchList[x][2]}")
        choice = int(input("\nWhich number fixture would you like to select? ")) - 1

        matchId = matchList[choice][1]
        matchInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}")['event']
        if matchInfo['status']['code'] != 100:
            print('Match not complete.')
            exit()
    else:
        matchId = defaultMatchId
        matchInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}")['event']
        if matchInfo['status']['code'] != 100:
            print('Match not complete.')
            exit()
    if model == None:
        matchInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}")['event']
        roundInfo = matchInfo['roundInfo']
        roundInfo.pop('cupRoundType', None)
        leagueId = matchInfo['tournament']['uniqueTournament']['id']
        seasonId = matchInfo['season']['id']

        categories = getTournamentStatList(leagueId, seasonId, roundInfo)
        if 'penaltyScored' in categories and 'penaltyMiss' not in categories:
            categories.append('penaltyMiss')
        elif 'penaltyMiss' in categories and 'penaltyScored' not in categories:
            categories.append('penaltyScored')

        model, X, X_test = trainModel(combinedDF, categories)

    lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0)
    incidents = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/incidents", 0)['incidents']

    homeAway = input("\nWould you like to pick from the home team? (y/n) ")

    if homeAway.lower() == 'y':
        players = lineups['home']['players']
        availablePlayers = []
        for player in players:
            if player['player']['position'] != 'G' and 'statistics' in player:
                if 'minutesPlayed' in player['statistics']:
                    if player['statistics']['minutesPlayed'] >= 20:
                        availablePlayers.append(player)
        for x in range(len(availablePlayers)):
            print(f"{x + 1}. {availablePlayers[x]['player']['name']}")
        answer = int(input("\nWhich player would you like to pick? ")) - 1
        chosenPlayer = availablePlayers[answer]
    else:
        players = lineups['away']['players']
        availablePlayers = []
        for player in players:
            if player['player']['position'] != 'G' and 'statistics' in player:
                if 'minutesPlayed' in player['statistics']:
                    if player['statistics']['minutesPlayed'] >= 20:
                        availablePlayers.append(player)
        for x in range(len(availablePlayers)):
            print(f"{x + 1}. {availablePlayers[x]['player']['name']}")
        answer = int(input("\nWhich player would you like to pick? ")) - 1
        chosenPlayer = availablePlayers[answer]


    chosenPlayer['statistics']['isSub'], chosenPlayer['statistics']['yellowCard'], chosenPlayer['statistics'][
        'redCard'], chosenPlayer['statistics']['penaltyScored'] = isSubYellowRedPenalty(chosenPlayer['player']['id'],
                                                                                 incidents)
    chosenPlayer['statistics']['position'] = chosenPlayer['position']
    chosenPlayer['statistics'] = convertStats(chosenPlayer['statistics'])

    df_player = pd.DataFrame([chosenPlayer['statistics']])

    df_player = df_player.reindex(columns=categories, fill_value=0)

    df_player['position'] = df_player['position'].map(positionsKey)

    df_player = df_player.apply(pd.to_numeric, errors='coerce').fillna(0)
    #print(df_player.T)

    rating = getRating(model, df_player)
    print(f"\nPlayer Rating: {rating:.2f}")

    print(f"Generating Waterfall Plot for {chosenPlayer['player']['name']}...")

    # --- SHAP for Single Model ---

    # 1. Create the Explainer
    # We pass the single 'model' directly.
    # model_output='probability' ensures the units are 0-1 (0% to 100%), not log-odds.
    explainer = shap.TreeExplainer(
        model,
        data=X,
        model_output='probability'
    )

    # 2. Calculate SHAP values
    # This generates a proper Explanation object automatically
    shap_values = explainer(df_player)

    # 3. Plot
    # shap_values[0] extracts the explanation for the specific player/row provided
    shap.plots.waterfall(shap_values[0], max_display=15)
    plt.show()

    return model, df_player, X, X_test


def getAndRateWholeMatch():
    TeamA = input("\nName the home team: ")
    homeTeam = getFullTeamName(TeamA)
    TeamB = input("\nName the away team: ")
    awayTeam = getFullTeamName(TeamB)

    searches = jsonRequest(f"http://www.sofascore.com/api/v1/search/events?q={homeTeam}%20{awayTeam}&page=0")['results']

    matchList = []

    for results in searches:
        result = results['entity']
        epoch = result['startTimestamp']
        dt_object = datetime.datetime.fromtimestamp(epoch)
        human_readable_time = dt_object.strftime('%d-%m-%Y')
        if result['homeTeam']['name'] == homeTeam and result['awayTeam']['name'] == awayTeam and epoch > 1656630000:
            matchList.append((result['name'], result['id'], human_readable_time, epoch))

    matchList = sorted(matchList, key=lambda x: x[3], reverse=True)

    for x in range(len(matchList)):
        print(f"{x + 1}. {matchList[x][2]}")
    choice = int(input("\nWhich number fixture would you like to select? ")) - 1

    matchId = matchList[choice][1]

    matchInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}")['event']
    if matchInfo['status']['code'] != 100:
        print('Match not complete.')
        exit()
    roundInfo = matchInfo['roundInfo']
    roundInfo.pop('cupRoundType', None)
    leagueId = matchInfo['tournament']['uniqueTournament']['id']
    seasonId = matchInfo['season']['id']

    categories = getTournamentStatList(leagueId, seasonId, roundInfo)
    if 'penaltyScored' in categories and 'penaltyMiss' not in categories:
        categories.append('penaltyMiss')
    elif 'penaltyMiss' in categories and 'penaltyScored' not in categories:
        categories.append('penaltyScored')
    model, X, X_test = trainModel(combinedDF, categories)

    lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0)
    incidents = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/incidents", 0)['incidents']

    total = {}
    names = []
    for homePlayer in lineups['home']['players']:
        position = homePlayer['position']
        if position != 'G' and 'statistics' in homePlayer:
            if 'minutesPlayed' in homePlayer['statistics']:
                if homePlayer['statistics']['minutesPlayed'] >= 20:
                    names.append(homePlayer['player']['name'])
                    homePlayer['statistics']['isSub'], homePlayer['statistics']['yellowCard'], homePlayer['statistics'][
                        'redCard'], homePlayer['statistics']['penaltyScored'] = isSubYellowRedPenalty(homePlayer['player']['id'],
                                                                                               incidents)
                    homePlayer['statistics']['position'] = position
                    homePlayer['statistics'] = convertStats(homePlayer['statistics'])
                    total[homePlayer['player']['name']] = homePlayer['statistics']
    homePlayerCount = len(names)
    for awayPlayer in lineups['away']['players']:
        position = awayPlayer['position']
        if position != 'G' and 'statistics' in awayPlayer:
            if 'minutesPlayed' in awayPlayer['statistics']:
                if awayPlayer['statistics']['minutesPlayed'] >= 20:
                    names.append(awayPlayer['player']['name'])
                    awayPlayer['statistics']['isSub'], awayPlayer['statistics']['yellowCard'], awayPlayer['statistics'][
                        'redCard'], awayPlayer['statistics']['penaltyScored'] = isSubYellowRedPenalty(awayPlayer['player']['id'],
                                                                                               incidents)
                    awayPlayer['statistics']['position'] = position
                    awayPlayer['statistics'] = convertStats(awayPlayer['statistics'])
                    total[awayPlayer['player']['name']] = awayPlayer['statistics']
    matchDf = pd.DataFrame(total)
    matchDf = matchDf.T
    matchDf = matchDf.reindex(columns=categories, fill_value=0)
    matchDf['position'] = matchDf['position'].map(positionsKey)
    matchDf = matchDf.apply(pd.to_numeric, errors='coerce').fillna(0)

    # print(matchDf) # Optional: comment this out to keep console clean

    print(f"\n{'Player Name':<25} | {'Rating':<6}")
    print("------------Home Team:------------")

    for x in range(len(names)):
        if x == homePlayerCount:
            print("------------Away Team:------------")
        # We must select the row as a DataFrame (iloc[[x]]) not a Series (iloc[x])
        # so dimensions stay correct for the model
        player_row = matchDf.iloc[[x]]

        rating = getRating(model, player_row)

        print(f"{names[x]:<25} | {rating:.2f}")
    return matchId, model, categories, X, X_test


def getAndAverageWholeMatch(matchId, model, categories=defaultColumns):
    # if 'goalAssist' in categories:
    #    categories.remove('goalAssist')
    matchInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}", 24 * 7)['event']
    if matchInfo['status']['code'] != 100:
        print('Match not complete.')
        exit()
    roundInfo = matchInfo['roundInfo']
    roundInfo.pop('cupRoundType', None)
    leagueId = matchInfo['tournament']['uniqueTournament']['id']
    seasonId = matchInfo['season']['id']

    lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0)
    incidents = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/incidents", 0)['incidents']

    homeTotal = {}
    awayTotal = {}
    homeNames = []
    awayNames = []
    count = 0
    for homePlayer in lineups['home']['players']:
        if count < 11:
            position = homePlayer['position']
            if position != 'G' and len(homePlayer['statistics'].keys()) > 6:
                homeNames.append(homePlayer['player']['name'])
                homePlayer['statistics']['isSub'], homePlayer['statistics']['yellowCard'], homePlayer['statistics'][
                    'redCard'], homePlayer['statistics']['penaltyScored'] = isSubYellowRedPenalty(homePlayer['player']['id'],
                                                                                           incidents)
                homePlayer['statistics']['position'] = position
                homeTotal[homePlayer['player']['name']] = homePlayer['statistics']
            count += 1
    count = 0
    for awayPlayer in lineups['away']['players']:
        if count < 11:
            position = awayPlayer['position']
            if position != 'G' and len(awayPlayer['statistics'].keys()) > 6:
                awayNames.append(awayPlayer['player']['name'])
                awayPlayer['statistics']['isSub'], awayPlayer['statistics']['yellowCard'], awayPlayer['statistics'][
                    'redCard'], awayPlayer['statistics']['penaltyScored'] = isSubYellowRedPenalty(awayPlayer['player']['id'],
                                                                                           incidents)
                awayPlayer['statistics']['position'] = position
                awayTotal[awayPlayer['player']['name']] = awayPlayer['statistics']
        count += 1
    homeMatchDf = pd.DataFrame(homeTotal)
    homeMatchDf = homeMatchDf.T
    homeMatchDf = homeMatchDf.reindex(columns=categories, fill_value=0)
    homeMatchDf['position'] = homeMatchDf['position'].map(positionsKey)
    homeMatchDf = homeMatchDf.apply(pd.to_numeric, errors='coerce').fillna(0)

    homePreds = np.array(model.predict_proba(homeMatchDf))
    homeRatings = []
    for x in range(len(homeNames)):
        homeRatings.append(homePreds[x][1] * 10)

    awayMatchDf = pd.DataFrame(awayTotal)
    awayMatchDf = awayMatchDf.T
    awayMatchDf = awayMatchDf.reindex(columns=categories, fill_value=0)
    awayMatchDf['position'] = awayMatchDf['position'].map(positionsKey)
    awayMatchDf = awayMatchDf.apply(pd.to_numeric, errors='coerce').fillna(0)

    awayPreds = np.array(model.predict_proba(awayMatchDf))
    awayRatings = []
    for x in range(len(awayNames)):
        awayRatings.append(awayPreds[x][1] * 10)
    return np.mean(homeRatings), np.mean(awayRatings)


def leagueHasPassmap(leagueId):
    seasonId = jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/seasons")['seasons'][0]['id']
    game = jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/last/0")[
        'events'][0]
    if 'hasEventPlayerStatistics' not in game:
        game['hasEventPlayerStatistics'] = False
    return game['hasEventPlayerStatistics'], seasonId


def checkStats(leagueId, seasonId, statCategories=defaultColumns):
    categoriesCopy = statCategories.copy()
    newStats = []
    matchIds = []
    matches = \
    jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/last/0")[
        'events']
    for match in matches:
        matchInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{match['id']}", 24 * 7)['event']
        if matchInfo['status']['code'] != 100:
            matchIds.append(match['id'])
    for matchId in matchIds:
        lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0)
        if 'home' in lineups:
            homePlayers = lineups['home']['players']
            awayPlayers = lineups['away']['players']
            for homePlayer in homePlayers:
                if homePlayer['position'] != 'G':
                    for key in homePlayer['statistics'].keys():
                        if key in categoriesCopy:
                            categoriesCopy.remove(key)
                        if key not in statCategories and key not in newStats:
                            newStats.append(key)
            for awayPlayer in awayPlayers:
                if awayPlayer['position'] != 'G':
                    for key in awayPlayer['statistics'].keys():
                        if key in categoriesCopy:
                            categoriesCopy.remove(key)
                        if key not in statCategories and key not in newStats:
                            newStats.append(key)
    return newStats, categoriesCopy

def getLeagueAverageRatings():
    leagueShortName = input("Enter the league name ")
    data = jsonRequest(f"http://www.sofascore.com/api/v1/search/unique-tournaments?q={leagueShortName}&page=0")['results']
    count = 0
    for i in data:
        if count < 10:
            count += 1
            print(f"{count}. {i['entity']['name']} ({i['entity']['category']['name']})")
    choice = input("Which number would you like to choose? ")
    leagueId = data[int(choice) - 1]['entity']['id']
    seasons = jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/seasons")['seasons']
    count = 0
    for season in seasons:
        if count < 10:
            count += 1
            print(f"{count}. {season['name']}")
    choice = input("Which number would you like to choose? ")
    seasonId = seasons[int(choice) - 1]['id']
    switch = True
    count = 0
    matchIds = []
    while switch:
        getGames = jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/last/{count}")
        games = getGames['events']
        for game in games:
            if game['status']['code'] == 100 and safe_request(f"http://www.sofascore.com/api/v1/event/{game['id']}/lineups", cache_ttl=0).status_code == 200 and game['id'] not in matchIds:
                matchIds.append(game['id'])
        count += 1
        switch = getGames['hasNextPage']
    roundInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchIds[0]}")['event']['roundInfo']
    roundInfo.pop('cupRoundType', None)
    categories = getTournamentStatList(leagueId, seasonId, roundInfo)
    model, X, X_test = trainModel(combinedDF, categories)
    total = {}
    names = []

    for matchId in matchIds:
        lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0)
        incidents = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/incidents", 0)['incidents']

        for player in lineups['home']['players']:
            key = f"{matchId}-->{player['player']['name']}"
            position = player['position']
            if position != 'G' and 'statistics' in player:
                if len(player['statistics'].keys()) > 5:
                    player['statistics']['isSub'], player['statistics']['yellowCard'], player['statistics'][
                        'redCard'], player['statistics']['penaltyScored'] = isSubYellowRedPenalty(
                        player['player']['id'],
                        incidents)
                    if player['statistics']['minutesPlayed'] >= 20:
                        player['statistics']['position'] = position
                        player['statistics'] = convertStats(player['statistics'])
                        names.append(player['player']['name'])
                        total[key] = player['statistics']
        for player in lineups['away']['players']:
            key = f"{matchId}-->{player['player']['name']}"
            position = player['position']
            if position != 'G' and 'statistics' in player:
                if len(player['statistics'].keys()) > 5:
                    player['statistics']['isSub'], player['statistics']['yellowCard'], player['statistics'][
                        'redCard'], player['statistics']['penaltyScored'] = isSubYellowRedPenalty(
                        player['player']['id'],
                        incidents)
                    if player['statistics']['minutesPlayed'] >= 20:
                        player['statistics']['position'] = position
                        player['statistics'] = convertStats(player['statistics'])
                        names.append(player['player']['name'])
                        total[key] = player['statistics']

    seasonDf = pd.DataFrame(total)
    seasonDf = seasonDf.T

    seasonDf = seasonDf.reindex(columns=categories, fill_value=0)

    seasonDf['position'] = seasonDf['position'].map(positionsKey)
    seasonDf = seasonDf.apply(pd.to_numeric, errors='coerce').fillna(0)
    ratingsDict = {}
    uniqueNames = list(set(names))
    for name in uniqueNames:
        ratingsDict[name] = []

    ratings = np.array(model.predict_proba(seasonDf))
    indexList = seasonDf.index.tolist()
    matchIdAndNames = []
    for index in indexList:
        matchIdAndNames.append(index.split("-->"))
    for y in range(len(matchIdAndNames)):
        matchIdAndNames[y].append(ratings[y][1] * 10)
    #print(sorted(matchIdAndNames, key=lambda x: x[2], reverse=True))
    for x in range(len(names)):
        ratingsDict[names[x]].append(ratings[x][1])
    for name in uniqueNames:
        ratingsDict[name] = (np.mean(ratingsDict[name]), len(ratingsDict[name]), np.std(ratingsDict[name]))
    ratingsList = list(ratingsDict.items())
    ratingsList = sorted(ratingsList, key=lambda x: x[1][0], reverse=True)
    filteredList = [item for item in ratingsList if item[1][1] > 9]
    print(f"\nNum   {'Player Name':<25} | {'Rating':<6} | {'Std Dev':<7} | Matches")
    for x in range(len(filteredList)):
        print(f"{str(x+1)+'.':<5} {filteredList[x][0]:<25} | {filteredList[x][1][0]*10:.4f} | {filteredList[x][1][2]*10:<7.4f} | {filteredList[x][1][1]:<2}")
    choice = int(input("\nWhich number player would you like to see matches for? ")) - 1
    selectedName = filteredList[choice][0]
    playerId = jsonRequest(f"http://www.sofascore.com/api/v1/search/player-team-persons?q={selectedName}&page=0")['results'][0]['entity']['id']
    playerMatches = []
    for index in matchIdAndNames:
        if index[1] == selectedName:
            playerMatches.append(index)
    switch = True
    count = 0
    playedForTeamMap = {}
    while switch:
        playedMatches = jsonRequest(f"http://www.sofascore.com/api/v1/player/{playerId}/unique-tournament/{leagueId}/events/last/{count}")
        playedForTeamMap.update(playedMatches['playedForTeamMap'])
        count += 1
        switch = playedMatches['hasNextPage']
    for playerMatch in playerMatches:
        matchId = playerMatch[0]
        playedForTeam = playedForTeamMap[matchId]
        matchInfo = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}")['event']
        if matchInfo['status']['code'] == 100:
            if matchInfo['homeTeam']['id'] == playedForTeam:
                venue = 'home'
                opposition = matchInfo['awayTeam']['name']
            elif matchInfo['awayTeam']['id'] == playedForTeam:
                venue = 'away'
                opposition = matchInfo['homeTeam']['name']
            else:
                break

            minutesPlayed = 0
            for player in jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", 0)[venue]['players']:
                if player['player']['id'] == playerId and 'statistics' in player:
                    if 'minutesPlayed' in player['statistics']:
                        minutesPlayed = player['statistics']['minutesPlayed']


            time = matchInfo['startTimestamp']
            playerMatch.append(venue)
            playerMatch.append(opposition)
            playerMatch.append(time)
            playerMatch.append(minutesPlayed)
    playerMatches = sorted(playerMatches, key=lambda x: x[5], reverse=True)
    print(f"Match performances for {selectedName}:\n")
    for match in playerMatches:
        print(f"{match[2]:.2f} vs {match[4]} ({match[3]}) - {match[6]}'")
    return filteredList

#deleteSeason(17, 76986)


choice = int(input("Would you like to:\n1. Get average ratings for a whole season\n2. Get ratings for a whole match\n3. Get one individual player's match rating\n"))
if choice == 1:
    getLeagueAverageRatings()
elif choice == 2:
    matchId, model, categories, X, X_test = getAndRateWholeMatch()
    explainPlayer = input("\nWould you like to explain a player's rating? (y/n) ")
    if explainPlayer == 'y':
        getMatchAndExplainPlayer(matchId, model, categories, X, X_test)
elif choice == 3:
    getMatchAndExplainPlayer()
else:
    quit()
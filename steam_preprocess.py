from numpy import interp
import pandas as pd

# This file is used to clean up the existing dataset. 
# It creates three new files containing the maximum play time per game, the adjusted ratings per user per game 
# and the shortened original dataset

def main():

    games = pd.read_csv('steam-200k.csv', index_col=False, header=None, names=['user', 'game', 'status', 'time'])
    # Keep all rows that contain 'play'
    gamesShort = games.loc[games['status'] == 'play'].copy()
    gamesShort.to_csv(path_or_buf="steam-100k.csv", header=False, index=False, columns=['user','game','time'])
    # Sort games based on time and keep only the most played one per game
    gameMaxTimes = gamesShort.sort_values('time', ascending=False).drop_duplicates(['game']).copy()
    # Write to new csv
    gameMaxTimes.to_csv('steam-max_times.csv', header=False, index=False, columns=['game','time'])
    gamesShort.index = range(len(gamesShort.index))

    # Update time column
    for i, row in gamesShort.iterrows():
        userTime = gamesShort.iloc[i]['time']
        game = gamesShort.iloc[i]['game']
        gameMax = gameMaxTimes.loc[gameMaxTimes['game'] == game, 'time'].iloc[0]
        userRating = interp(userTime, [0, gameMax], [1, 100]).round()
        gamesShort.at[i, 'time'] = userRating

    gamesShort.to_csv('steam-ratings-max.csv', header=False, index=False, columns=['user','game','time'])

if __name__ == "__main__":
    main()
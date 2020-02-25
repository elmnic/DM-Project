from surprise import Dataset
from surprise import Reader
from surprise import similarities
from surprise import prediction_algorithms
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

import pandas as pd
from random import randint

def main():
    # Load user ratings
    ratings = pd.read_csv('steam-ratings_max.csv', index_col=False, header=None, names=['user', 'game', 'rating'])

    # Load data into similarity algorithm
    reader = Reader(rating_scale=(1,100))
    data = Dataset.load_from_df(ratings, reader=reader)
    sim_options = {'name': 'cosine',
                   'user_based': False # Item-based
                   }
    # Split data into train and test set
    trainset, testset = train_test_split(data, test_size=20) # 20% test set

    # Create model
    sim_algo = prediction_algorithms.KNNBasic(k=100, sim_options=sim_options)

    # Fit model to train set
    sim_algo.fit(trainset)
    sim_algo.test(testset)

    # # Cross-validation with 5 folds on the data
    #print("Testing cross validation")
    #cross_validate(sim_algo, data, verbose=True)

    # Get unique games and unique users
    uqGames = pd.read_csv('steam-max_times.csv', index_col=False, header=None, names=['game', 'time'])
    uqUsers = ratings.drop_duplicates(['user']).copy()
    uqUsers.index = range(len(uqUsers.index))

    # Pick a random user for recommendation
    uid = uqUsers.iloc[randint(0, len(uqUsers) - 1)]['user']
    userPredictions_list = [[]].pop()

    # Create predictions for all games that the user haven't played yet
    for _, rowGames in uqGames.iterrows():
        iid = rowGames['game']
        try: # Try to find an already existing rating for the [user,game] pair
            ratings.loc[uid, iid]
            break
        except KeyError: # Otherwise create and store prediction
            pred = sim_algo.predict(uid, iid)
            userPredictions_list.append([uid, iid, pred.est])

    # Convert to DataFrame and sort according to rating
    predictionDF = pd.DataFrame(userPredictions_list, columns=['user', 'game', 'est']).sort_values(by=['user', 'est'], ascending=False)

    print("Users top 5 rated games:")
    print(ratings.loc[ratings['user'] == uid, :].sort_values('rating', ascending=False).head())

    print("Users top 5 recommended games:")
    print(predictionDF.head())


if __name__ == "__main__":
    main()
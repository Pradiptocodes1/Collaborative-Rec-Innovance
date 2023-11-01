import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from math import sqrt

def recommend_movies(user_input, top_n=20):

    cols_u = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users=pd.read_csv('u.user', sep='|', names=cols_u, encoding='latin-1', parse_dates=True)

    ratings_u = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings=pd.read_csv('u.data', sep='\t', names=ratings_u, encoding='latin-1')

    movies_u = ['movie_id' , 'title', 'release_date', 'video_release_date', 'imdb_url']
    movies = pd.read_csv('u.item', sep='|', names=movies_u, usecols=range(5), encoding='latin-1')

    # Merge data
    movie_ratings = pd.merge(movies, ratings)
    movie_df = pd.merge(movie_ratings, users)

    # Data preprocessing
    movie_df.drop(movie_df.columns[[3, 4, 7]], axis=1, inplace=True)
    ratings.drop("timestamp", inplace=True, axis=1)
    movies.drop(movies.columns[[3, 4]], inplace=True, axis=1)
    input_movies = pd.DataFrame(userInput)
    movies['title'] = movies['title'].str.replace(r'\(\d\d\d\d\)', '')
    movies['title'] = movies['title'].apply(lambda x: x.strip())
    input_id = movies[movies['title'].isin(input_movies['title'].tolist())]
    input_movies = pd.merge(input_id, input_movies)
    input_movies = input_movies.drop('release_date',axis= 1)

    user_subset = ratings[ratings['movie_id'].isin(input_movies['movie_id'].tolist())]

    user_subset_group = user_subset.groupby(['user_id'])
    user_subset_group = sorted(user_subset_group, key=lambda x: len(x[1]), reverse=True)
    #Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    pearsonCorrelationDict = {}
    #For every user group in our subset
    for name, group in user_subset_group:
        #Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movie_id')
        input_movies = input_movies.sort_values(by='movie_id')
        #Get the N for the formula
        nRatings = len(group)
        #Get the review scores for the movies that they both have in common
        temp_df = input_movies[input_movies['movie_id'].isin(group['movie_id'].tolist())]
        #And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()
        #Let's also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        #Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
        Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)

    #If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
          pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
        else:
          pearsonCorrelationDict[name] = 0
    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))

    topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
    topUsersRating=topUsers.merge(ratings, left_on='userId', right_on='user_id', how='inner')
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
    tempTopUsersRating = topUsersRating.groupby('movie_id').sum()[['similarityIndex','weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']

    # Top N recommendations
    recommendation_df = pd.DataFrame()
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index

    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

    # Get the top N recommended movie titles
    top_movies_titles = movies.loc[movies['movie_id'].isin(recommendation_df.head(top_n)['movieId'].tolist()), 'title']
    titles = top_movies_titles.tolist()
    return titles



# Get the top 20 recommended movies based on user input
userInput = []
for i in range(5):
    title = input("Enter the title of movie " + str(i+1) + ": ")
    rating = float(input("Enter your rating for movie " + str(i+1) + ": "))
    userInput.append({'title': title, 'rating': rating})

titles = recommend_movies(userInput, top_n=20)
print("Generating 20 recommendations.....")
print()
print()
for title in titles:
    print(title)
import numpy as np
import pandas as pd
import json
import sklearn as skl

def toVec(genres):
    vec = [0] * 20
    # print(genres)
    for g in genres.split('|'):
        # print(g)
        vec[genretoidx[g]] += 1
    return vec

def fx(rating, movieId):
    # print(genredict[movieId], rating)
    return (rating * np.array(genredict[movieId])).astype(object)

def gx(genrating):
    return np.rint((10 / (1 + np.exp(-genrating.astype(float)*0.2)))).astype(object)

movies = pd.read_csv("./ml-latest-small/movies.csv")
genre = {}
for l in movies['genres'].str.split('|'):
    for g in l:
        if g not in genre:
            genre[g] = 1
        else:
            genre[g] += 1
            
genretoidx = {}
idxtogenre = []
for i, g in enumerate(genre):
    # print(g, i)
    genretoidx[g] = i
    idxtogenre.append(g)

    
genredict = {m[0] : toVec(m[1]) for m in movies[['movieId','genres']].to_numpy()}
with open('data.json', 'w') as fp:
    json.dump(genredict, fp)
    
    
ratings = pd.read_csv("./ml-latest-small/ratings.csv")
train, test = skl.model_selection.train_test_split(ratings, test_size=0.2)
train['rating'] = train.groupby('userId')['rating'].transform(lambda x: (x - x.mean()) / x.std())
train['genrating'] = np.vectorize(fx)(train['rating'], train['movieId'])
train["sigmoid"] = np.vectorize(gx)(train['genrating'])

scores = train.groupby('userId')['genrating'].sum()
users = (train.groupby(['userId']).size()).to_frame(name="rated")
users["scores"] = scores
import numpy as np
import pandas as pd
import json
import sklearn as skl
from sklearn.model_selection import GroupShuffleSplit

movies = pd.read_csv("../ml-25m/movies.csv") #relative
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
    
def toVec(genres):
    vec = [0] * 20
    # print(genres)
    for g in genres.split('|'):
        # print(g)
        vec[genretoidx[g]] += 1
    return vec
    
genredict = {m[0] : toVec(m[1]) for m in movies[['movieId','genres']].to_numpy()}
with open('data.json', 'w') as fp:
    json.dump(genredict, fp)
    
def fx(rating, movieId):
    # print(genredict[movieId], rating)
    return (rating * np.array(genredict[movieId])).astype(object)

def gx(genrating):
    return np.rint((10 / (1 + np.exp(-genrating.astype(float)*0.2)))).astype(object)

ratings = pd.read_csv("../ml-25m/ratings.csv")#relative
ratings['rating'] = ratings.groupby('userId')['rating'].transform(lambda x: (x - x.mean()) / x.std())
ratings['genrating'] = np.vectorize(fx)(ratings['rating'], ratings['movieId'])
ratings["sigmoid"] = np.vectorize(gx)(ratings['genrating'])

scores = ratings.groupby('userId')['genrating'].sum()
users = (ratings.groupby(['userId']).size()).to_frame(name="rated")
users["scores"] = scores
users["sigmoid"] = np.vectorize(gx)(users['scores'])

train_mod = ratings[['userId', 'movieId']]
# train_mod.merge(users, on="userId")
cols = ["userId"]
train_mod = train_mod.merge(users, left_on='userId', right_on='userId')
train_mod["genre"] = train_mod['movieId'].apply(lambda x: np.random.choice(np.nonzero(np.array(genredict[x]) == 1)[0])
)
train_mod["reward"] = 1
train_mod.drop(columns=["movieId", "rated", "scores"], inplace=True)

train_false = train_mod.copy()
train_false['reward'] = 0
train_false['genre'] = np.random.randint(1, 20, train_false.shape[0])

train_final = pd.concat([train_mod, train_false], ignore_index=True)
train_final = train_final.iloc[:,[2,3,0,1]]
train_final[list(genre.keys())] = pd.DataFrame(train_final.sigmoid.tolist(), index= train_final.index)
train_final.drop(columns=["userId", "sigmoid"], inplace=True)
train_final = train_final.sample(frac=1)
train_final.head()

train_final.dropna(inplace=True)

np.savetxt("./Traindataset.txt", train_final.values, fmt='%d')
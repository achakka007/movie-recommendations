from datetime import datetime

from surprise import SVD
import numpy as np
from surprise import Reader, Dataset
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity
import pandas as pd
import os
import xgboost as xgb

# Read CSV file
data = pd.read_csv(os.path.join("ml-latest-small", "ratings.csv"))

# Drop timestamp column
data = data.drop('timestamp', axis=1)

# Split train and test data
train_data = data.iloc[:int(data.shape[0]*0.80)]
test_data = data.iloc[int(data.shape[0]*0.80):]


# Specify how to read the data frame.
reader = Reader(rating_scale=(1, 5))

# Create the traindata from the data frame
train_data_mf = Dataset.load_from_df(
    train_data[['userId', 'movieId', 'rating']],
    reader
)  # Get training data from MovieLens

# Build train set
trainset = train_data_mf.build_full_trainset()
svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svd.fit(trainset)

# Get predictions for train set
train_preds = svd.test(trainset.build_testset())
train_pred_mf = np.array([pred.est for pred in train_preds])

# Create a sparse matrix
train_sparse_matrix = sparse.csr_matrix(
    (train_data.rating.values, (train_data.userId.values, train_data.movieId.values)))

train_averages = dict()
# Get global average of ratings in train set.
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average
train_averages

Output = {'global': 3.5199769425298757}


def get_average_ratings(sparse_matrix, of_users):

    ax = 1 if of_users else 0

    sum_of_ratings = sparse_matrix.sum(axis=ax).A1

    is_rated = sparse_matrix != 0

    no_of_ratings = is_rated.sum(axis=ax).A1

    u, m = sparse_matrix.shape

    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i]
                       for i in range(u if of_users else m)
                       if no_of_ratings[i] != 0}

    return average_ratings


train_averages['user'] = get_average_ratings(
    train_sparse_matrix, of_users=True)

train_averages['movie'] = get_average_ratings(
    train_sparse_matrix, of_users=False)

train_users, train_movies, train_ratings = sparse.find(train_sparse_matrix)

final_data = pd.DataFrame()
count = 0


start = datetime.now()
for (user, movie, rating) in zip(train_users, train_movies, train_ratings):
    user_sim = cosine_similarity(
        train_sparse_matrix[user], train_sparse_matrix).ravel()

    top_sim_users = user_sim.argsort()[::-1][1:]

    top_ratings = train_sparse_matrix[top_sim_users, movie].toarray().ravel()

    top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
    top_sim_users_ratings.extend(
        [train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))

    movie_sim = cosine_similarity(
        train_sparse_matrix[:, movie].T, train_sparse_matrix.T).ravel()

    top_sim_movies = movie_sim.argsort()[::-1][1:]

    top_ratings = train_sparse_matrix[user, top_sim_movies].toarray().ravel()

    top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
    top_sim_movies_ratings.extend(
        [train_averages['user'][user]]*(5-len(top_sim_movies_ratings)))

    row = list()
    row.append(user)
    row.append(movie)

    row.append(train_averages['global'])

    row.extend(top_sim_users_ratings)

    row.extend(top_sim_movies_ratings)

    row.append(train_averages['user'][user])

    row.append(train_averages['movie'][movie])

    row.append(rating)
    count = count + 1
    final_data = final_data.append([row])
    print(count)

    if (count) % 10000 == 0:
        print("Done for {} rows----- {}".format(count, datetime.now() - start))


final_data.columns = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5',
                      'smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating']

final_data['mf_svd'] = train_pred_mf

test_sparse_matrix = sparse.csr_matrix(
    (test_data.rating.values, (test_data.userId.values, test_data.movieId.values)))


test_averages = dict()

test_global_average = test_sparse_matrix.sum()/test_sparse_matrix.count_nonzero()
test_averages['global'] = test_global_average
test_averages


def get_average_ratings(sparse_matrix, of_users):

    ax = 1 if of_users else 0

    sum_of_ratings = sparse_matrix.sum(axis=ax).A1

    is_rated = sparse_matrix != 0

    no_of_ratings = is_rated.sum(axis=ax).A1

    u, m = sparse_matrix.shape

    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i]
                       for i in range(u if of_users else m)
                       if no_of_ratings[i] != 0}

    return average_ratings


test_averages['user'] = get_average_ratings(test_sparse_matrix, of_users=True)

test_averages['movie'] = get_average_ratings(
    test_sparse_matrix, of_users=False)
print('\n AVerage rating of movie 15 :', test_averages['movie'][15])

test_users, test_movies, test_ratings = sparse.find(test_sparse_matrix)

final_test_data = pd.DataFrame()
count = 0

start = datetime.now()
for (user, movie, rating) in zip(test_users, test_movies, test_ratings):
    user_sim = cosine_similarity(
        test_sparse_matrix[user], test_sparse_matrix).ravel()
    top_sim_users = user_sim.argsort()[::-1][1:]

    top_ratings = test_sparse_matrix[top_sim_users, movie].toarray().ravel()

    top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
    top_sim_users_ratings.extend(
        [test_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))

    movie_sim = cosine_similarity(
        test_sparse_matrix[:, movie].T, test_sparse_matrix.T).ravel()

    top_sim_movies = movie_sim.argsort()[::-1][1:]

    top_ratings = test_sparse_matrix[user, top_sim_movies].toarray().ravel()

    top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
    top_sim_movies_ratings.extend(
        [test_averages['user'][user]]*(5-len(top_sim_movies_ratings)))

    row = list()
    row.append(user)
    row.append(movie)

    row.append(test_averages['global'])

    row.extend(top_sim_users_ratings)

    row.extend(top_sim_movies_ratings)

    row.append(test_averages['user'][user])

    row.append(test_averages['movie'][movie])

    row.append(rating)
    count = count + 1
    final_test_data = final_test_data.append([row])
    if (count) % 1000 == 0:
        print(count)

    if (count) % 10000 == 0:
        print("Done for {} rows----- {}".format(count, datetime.now() - start))

final_test_data.columns = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5',
                           'smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating']


reader = Reader(rating_scale=(1, 5))

test_data_mf = Dataset.load_from_df(
    test_data[['userId', 'movieId', 'rating']], reader)

testset = test_data_mf.build_full_trainset()

test_preds = svd.test(testset.build_testset())

test_pred_mf = np.array([pred.est for pred in test_preds])
final_test_data['mf_svd'] = test_pred_mf


def get_error_metrics(y_true, y_pred):
    rmse = np.sqrt(
        np.mean([(y_true[i] - y_pred[i])**2 for i in range(len(y_pred))]))
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return rmse, mape


x_train = final_data.drop(['user', 'movie', 'rating'], axis=1)
y_train = final_data['rating']

x_test = final_test_data.drop(['user', 'movie', 'rating'], axis=1)
y_test = final_test_data['rating']

xgb_model = xgb.XGBRegressor(
    n_jobs=13, random_state=15, n_estimators=100)

train_results = dict()
test_results = dict()


print('Training the model..')
start = datetime.now()
xgb_model.fit(x_train, y_train, eval_metric='rmse')
print('Done. Time taken : {}\n'.format(datetime.now()-start))
print('Done \n')

print('Evaluating the model with TRAIN data...')
start = datetime.now()
y_train_pred = xgb_model.predict(x_train)

rmse_train, mape_train = get_error_metrics(y_train.values, y_train_pred)

train_results = {'rmse': rmse_train,
                 'mape': mape_train,
                 'predictions': y_train_pred}

print(train_results)

print('Evaluating Test data')
y_test_pred = xgb_model.predict(x_test)
rmse_test, mape_test = get_error_metrics(
    y_true=y_test.values, y_pred=y_test_pred)

test_results = {'rmse': rmse_test,
                'mape': mape_test,
                'predictions': y_test_pred}

print(test_results)

from surprise import SVD
import numpy as np
# import surprise
from surprise import Reader, Dataset

# It is to specify how to read the data frame.
reader = Reader(rating_scale=(1, 5))
# create the traindata from the data frame
train_data_mf = Dataset.load_from_df(
    train_data[['userId', 'movieId', 'rating']], reader)  # Get training data from MovieLens
# build the train set from traindata.
# It is of dataset format from surprise library
trainset = train_data_mf.build_full_trainset()
svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svd.fit(trainset)

# getting predictions of train set
train_preds = svd.test(trainset.build_testset())
train_pred_mf = np.array([pred.est for pred in train_preds])

# Creating a sparse matrix
train_sparse_matrix = sparse.csr_matrix(
    (train_data.rating.values, (train_data.userId.values, train_data.movieId.values)))

train_averages = dict()
# get the global average of ratings in our train set.
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average
train_averages

Output = {'global': 3.5199769425298757}  # : to =

# Next, let’s create a function which takes the sparse matrix as input and gives the average ratings of a movie given by all users, and the average rating of all movies given by a single user.
# get the user averages in dictionary (key: user_id/movie_id, value: avg rating)


def get_average_ratings(sparse_matrix, of_users):

    # average ratings of user/axes
    ax = 1 if of_users else 0  # 1 - User axes,0 - Movie axes
    # ".A1" is for converting Column_Matrix to 1-D numpy array
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of ratings ( whether a user rated that movie or not)
    is_rated = sparse_matrix != 0
    # no of ratings that each user OR movie..
    no_of_ratings = is_rated.sum(axis=ax).A1
    # max_user and max_movie ids in sparse matrix
    u, m = sparse_matrix.shape
    # create a dictionary of users and their average ratings..
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i]
                       for i in range(u if of_users else m)
                       if no_of_ratings[i] != 0}

    return average_ratings  # return that dictionary of average ratings


train_averages['user'] = get_average_ratings(
    train_sparse_matrix, of_users=True)

train_averages['movie'] = get_average_ratings(
    train_sparse_matrix, of_users=False)

# compute the similar Users of the "user"
user_sim = cosine_similarity(
    train_sparse_matrix[user], train_sparse_matrix).ravel()
# we are ignoring 'The User' from its similar users.
top_sim_users = user_sim.argsort()[::-1][1:]
# get the ratings of most similar users for this movie
top_ratings = train_sparse_matrix[top_sim_users, movie].toarray().ravel()
# we will make it's length "5" by adding movie averages to
top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
top_sim_users_ratings.extend(
    [train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))

# compute the similar movies of the "movie"
movie_sim = cosine_similarity(train_sparse_matrix[:, movie].T,
                              train_sparse_matrix.T).ravel()
top_sim_movies = movie_sim.argsort()[::-1][1:]
# we are ignoring 'The User' from its similar users.
# get the ratings of most similar movie rated by this user
top_ratings = train_sparse_matrix[user, top_sim_movies].toarray().ravel()
# we will make it's length "5" by adding user averages to
top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
top_sim_movies_ratings.extend(
    [train_averages['user'][user]]*(5-len(top_sim_movies_ratings)))

# prepare train data
x_train = final_data.drop(['user', 'movie', 'rating'], axis=1)
y_train = final_data['rating']
# initialize XGBoost model
xgb_model = xgb.XGBRegressor(
    silent=False, n_jobs=13, random_state=15, n_estimators=100)
# fit the model
xgb_model.fit(x_train, y_train, eval_metric='rmse')

# dictionaries for storing train and test results
test_results = dict()
# from the trained model, get the predictions
y_est_pred = xgb_model.predict(x_test)
# get the rmse and mape of train data
rmse = np.sqrt(np.mean([(y_test.values[i] - y_test_pred[i])**2 for i in
                        range(len(y_test_pred))]))
mape = np.mean(np.abs((y_test.values - y_test_pred)/y_true.values)) * 100
# store the results in train_results dictionary
test_results = {'rmse': rmse_test,
                'mape': mape_test, 'predictions': y_test_pred}

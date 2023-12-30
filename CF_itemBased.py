import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import mean_absolute_error, precision_score, recall_score

top_k = int(sys.argv[1])
train_prec = float(sys.argv[2])

columns = ['userId','movieId','rating']
df = pd.read_csv('ml-latest-small/ratings.csv', usecols=columns)

# Preprocess
df1 = df.groupby(['userId']).filter(lambda x: len(x)>=5)
df2 = df.groupby(['movieId']).filter(lambda x: len(x)>=5)
merged = pd.merge(df1, df2)

training_data, testing_data = train_test_split(merged, test_size=0.1, train_size=train_prec, random_state=200)
# Create a pivot table of the data to get the ratings matrix
ratings_matrix = training_data.pivot_table(values='rating',columns='userId',index='movieId').fillna(0)
# Calculate Pearson similarity
corrMatrix = (ratings_matrix.T).corr(method='pearson')
# allUsersListPerMovieId : Dataframe with a list of all users rated every movie
allUsersListPerMovieId = (training_data.groupby(['movieId'])['userId'].apply(lambda x: x.tolist()).reset_index(name='users_who_rated'))

avg_preds=[]
avg_w_preds=[]
avg_user_w_preds=[]
true_rates = []
for index, row in testing_data.iterrows():
    user_id = row[0]
    item_id = row[1]
    true_rating = row[2]
    users_rated_this_movie = allUsersListPerMovieId[allUsersListPerMovieId['movieId']==item_id]['users_who_rated']
    users_rated_this_movie = [item for sublist in users_rated_this_movie.values.tolist() for item in sublist]
    
    # rated_items_id : movies that the user from test set has rated in training set
    rated_items = training_data[training_data.loc[:,'userId'] == user_id]
    rated_items_id = rated_items.loc[:,'movieId']

    # Checks weather or not the train-test split have left any ratings of that user in the training set
    if(rated_items_id.empty):
        continue
    # c2: List with movieIds of movies from corellation matrix
    c2=[]
    c2 = list(corrMatrix.columns)
    if item_id in c2:
        c2.remove(item_id) 
        true_rates.append(true_rating)
    else:
        continue
    # item_id's similairies with all the other movies (except itself)
    item_similarities = (corrMatrix.loc[c2,item_id]).sort_values(ascending=False).to_frame()
    item_similarities.set_axis([*item_similarities.columns[:-1], 'similarity'], axis=1, inplace=True)
    
    # The moviesUserRatedInfos Dataframe contains the columns: |movieId|rating|similarity|
    moviesUserRatedInfos = pd.merge(rated_items, item_similarities, on='movieId',).drop(columns = ['userId']).sort_values('similarity',ascending=False).head(top_k)
    moviesUserRatedInfos.set_axis([*moviesUserRatedInfos.columns[:-1], 'similarity'], axis=1, inplace=True)
    # Merge the above dataframe with allUsersListPerMovieId
    moviesUserRatedInfosWithCounts = pd.merge(moviesUserRatedInfos, allUsersListPerMovieId, on='movieId')
    # Find the number of common users rated the 2 movies
    common = []
    for _, rows2 in moviesUserRatedInfosWithCounts.iterrows():
        common.append(len(set(users_rated_this_movie).intersection(rows2['users_who_rated'])))
    
    # *** Predictions ***
    # Average Prediction Function
    predict_average = moviesUserRatedInfos['rating'].mean()
    
    # Weighted Average Prediction Function with weights based on pearson similarity
    predict_w_average = np.average(moviesUserRatedInfos['rating'], weights=moviesUserRatedInfos['similarity'])
    
    # Weighted Average Prediction Function with weights based on the number of common users who rated both movies
    userWeight = np.power(1.06, common)
    predict_user_wheighted = np.average(moviesUserRatedInfos['rating'], weights=userWeight)

    avg_preds.append(predict_average)
    avg_w_preds.append(predict_w_average)
    avg_user_w_preds.append(predict_user_wheighted)

true_rates = np.array(true_rates)
avg_preds = np.array(avg_preds)
avg_w_preds = np.array(avg_w_preds)
avg_user_w_preds = np.array(avg_user_w_preds)

true_rates_binary = np.where(true_rates >=3, 1, 0)
avg_preds_binary = np.where(avg_preds >=3, 1, 0)
avg_w_preds_binary = np.where(avg_w_preds >=3, 1, 0)
avg_user_w_preds_binary = np.where(avg_user_w_preds >=3, 1, 0)

mae_avg = mean_absolute_error(true_rates, avg_preds)
precision_avg = precision_score(true_rates_binary, avg_preds_binary)
recall_avg = recall_score(true_rates_binary, avg_preds_binary)

mae_w_avg = mean_absolute_error(true_rates, avg_w_preds)
precision_w_avg = precision_score(true_rates_binary, avg_w_preds_binary)
recall_w_avg = recall_score(true_rates_binary, avg_w_preds_binary)

mae_user_w_avg = mean_absolute_error(true_rates, avg_user_w_preds)
precision_user_w_avg = precision_score(true_rates_binary, avg_user_w_preds_binary)
recall_user_w_avg = recall_score(true_rates_binary, avg_user_w_preds_binary)

print('For k =',top_k,"we have the following scores:")
print("\nAverage Function:")
print("MAE:", mae_avg)
print("Precision:", precision_avg)
print("Recall:", recall_avg)

print("\nAverage Weighted Function:")
print("MAE:", mae_w_avg)
print("Precision:", precision_w_avg)
print("Recall:", recall_w_avg)

print("\nAverage User Weighted Function:")
print("Total predictions:")
print("MAE:", mae_user_w_avg)
print("Precision:", precision_user_w_avg)
print("Recall:", recall_user_w_avg)

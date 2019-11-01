import pandas as pd
import numpy as np
#import matplotlib as plt
import pickle
#%matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer
#from app import Recommender

class build_model():
    def __init__(self):
        self.movie = ''
        self.genre=''
        self.rating = 0.0
        
    #ratings dataset
    ratings = pd.read_csv('ratings.csv')
    ratings.head()
    ratings.drop('timestamp', inplace=True, axis=1)


    #users dataset
    users = pd.read_csv('users.csv') 
    users.head()


    #movies (items) dataset
    movies = pd.read_csv('movies1.csv')
    movies.head()
    m = movies.copy()
    m1 = m.copy()
    m1.drop('movieId', inplace = True, axis=1)
    m1.head()
    
    movie_ratings = pd.merge(movies, ratings, on='movieId', how='left')
    movie_ratings.head()

    #calculate average ratings for each movie
    avg = movie_ratings.copy()
    co = avg.groupby('title')['rating'].count()
    avg1 = avg.groupby('title')['rating'].mean()
    avg1.head(10)

     #merge users and movie_ratings dataframes
    df= pd.merge(movie_ratings, users, on='userId')
    df.head(10)
    
    #dropping unusable columns
    df.drop(['age','sex','occupation','zip_code'], inplace=True, axis=1)

    # filter the movies data frame
    movies2 = movies[movies.movieId.isin(df)]
    m2 = m1.copy()
    m2.head()
     # map movie to id:
    Mapping_file = dict(zip(movies.movieId.tolist(), movies.title.tolist()))
    Mapping_file
    m1
    '''
    FUNCTION TO RETURN THE MOVIE DETAILS OF THE MOVIE SEARCHED
    '''
    def display_movie(title, m1=m1):
        display = m1[m1.title == title]
        
        return display
    
    display_movie('Balto')    
    '''
    FUNCTION TO FIND TOP 20 MOVIES
    '''
    def toptwenty(df=df, avg1=avg1):
        df3 = df.copy()  
        df3.head(10)
        list(df3)    
        avg1.head(10)
        df4 = pd.merge(df3,avg1, on='title')
        df4.head(10)
        df4 = df4.drop_duplicates()
        list(df4)
        df4.drop('userId', axis=1, inplace=True)
        df4.drop(['movieId','rating_x'], axis=1, inplace=True)
        df4.columns = ['title','genres','year','rating']
        df4 = df4.drop_duplicates()
        a =  df4.sort_values(by='rating', ascending=False, axis=0).head(20)
        return a
    
    top20 = toptwenty()
#    top20
    
    '''
    CONTENT BASED FILTERING TO RECOMMEND TOP 20 MOVIES BASED ON GENRE OF THE SEARCHED MOVIE
    '''
    #Import TfIdfVectorizer from scikit-learn
    
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
    
    #Replace NaN with an empty string
    m['genres'] = m['genres'].fillna('')
    
    # Check and clean NaN values
    print ("Number of movies Null values: ", max(movies.isnull().sum()))
    print ("Number of ratings Null values: ", max(ratings.isnull().sum()))
    movies.dropna(inplace=True)
    ratings.dropna(inplace=True)
    
    
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(m['genres'])
    
    #Output the shape of tfidf_matrix
    tfidf_matrix.shape

    # Import linear_kernel
    from sklearn.metrics.pairwise import linear_kernel
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    #Construct a reverse map of indices and movie titles
    indices = pd.Series(m.index, index=m['title']).drop_duplicates()
    
    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(title, cosine_sim=cosine_sim, top20=top20, Mappig_file=Mapping_file, indices=indices, m1=m1):
        # Get the index of the movie that matches the title
        idx = indices[title]
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:21]
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        # Return the top 10 most similar movies
        return m1.iloc[movie_indices]  
        
#    get_recommendations('Balto')
    
#    from scipy.sparse import csr_matrix
    
    
    

    
#    no = df.groupby('title')['rating'].count()
#    no
#    
#    data_matrix = df.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True) 
#    data_matrix.head()
#    data_matrix.fillna(0, inplace=True)
#    movie_to_idx = { movie: i for i, movie in enumerate(list(movies.set_index('movieId').loc[data_matrix.index].title))}
#    
#    movie_to_idx
#
#
#    from sklearn.metrics.pairwise import cosine_similarity
#        
#    cosine_sim2 = cosine_similarity(data_matrix)
#    np.fill_diagonal(cosine_sim2, 0)
#    data_matrix= pd.DataFrame(cosine_sim2)
#    data_matrix.head(20)
#    
#    df3 = df.copy()  
#    df3.head(10)
#    list(df3)    
#    avg1.head(10)
#    df4 = pd.merge(df3,avg1, on='title')
#    df4.head(10)
#    df4 = df4.drop_duplicates()
#    list(df4)
#    df4.drop('userId', axis=1, inplace=True)
#    df4.drop(['movieId','rating_x'], axis=1, inplace=True)
#    df4.columns = ['title','genres','year','rating']
#    
#        
#    
#    #titles2= movies['title']
#    #indices2 = pd.Series(movies.index, index=movies['title'])
#    indices2 = pd.Series(m.index, index=m['title']).drop_duplicates()
#    def top_recommendations(title):
#        idx = indices2[title]
#        sim_scores = list(enumerate(cosine_sim2[idx]))
#        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#        sim_scores = sim_scores[1:21]
#        movie_indices = [i[0] for i in sim_scores]
#        return m2.iloc[movie_indices]
#
#
#    #top_recommendations('John Wick')

#    def get_recommendations1(title, cosine_sim1=cosine_sim, cosine_sim2=cosine_sim2):
#        # Get the index of the movie that matches the title
#        idx = indices[title]
#    
#        # Get the pairwsie similarity scores of all movies with that movie
#        sim_scores1 = list(enumerate(cosine_sim1[idx]))
#        sim_scores2 = list(enumerate(cosine_sim2[idx]))
#    
#        # Sort the movies based on the similarity scores
#        sim_scores1 = sorted(sim_scores1, key=lambda x: x[1], reverse=True)
#        sim_scores2 = sorted(sim_scores2, key=lambda x: x[1], reverse=True)
#    
#    
#        # Get the scores of the 10 most similar movies
#        sim_scores1 = sim_scores1[1:21]
#        sim_scores2 = sim_scores2[1:21]
#        # Get the movie indices
#        movie_indices1 = [i[0] for i in sim_scores1]
#        movie_indices2 = [i[0] for i in sim_scores2]
#    
#        # Return the top 10 most similar movies
#        d= {'a' : m1.iloc[movie_indices1],
#            'b' : m1.iloc[movie_indices2]}
#        
#        return d
#    get_recommendations1('Pleasantville')

#    print(top20)
    print(m1)  
    p1 = pickle.dump(Mapping_file, open('map.pkl','wb'))
    p2 = pickle.dump(top20, open("model.pkl", 'wb'))
    
    
if __name__ == "__main__":
    build_model()
    










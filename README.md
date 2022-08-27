# my-new-project
my project name is movies recommendation system  when we search any movies my project suggest the 5 movies name 
# Movies recommendar project 
### jupitor notbook ####
import numpy as np
import pandas as pd

movies= pd.read_csv('tmdb_5000_movies.csv') ## here we uploading the data set of movies 
credits= pd.read_csv('tmdb_5000_credits.csv') # here we upload the credits data set 

movies.head(1) # showing the movies data set 

credits.head(1) ## showing the credits data set 

credits.head()['cast'] ## i want to see the cast column

## we have two data set of same movies now we joind the two data set



movies=movies.merge(credits, on='title') # in this i am merge the two data set on the basic of title 



movies.head(1) # some colums add in my movies data set 

movies['original_language'].value_counts() # in this we check the total values count of the language 

movies.info()  ##chcking all information of movies data set

credits.info() ##chcking all information of credits data set

# genres              ## we keeping thes column to make project 
# id 
# keywords 
# title
# overview
# cast
# crew 
## in this we exctracting some column which is required to aur project 

movies=movies[['movie_id', 'title','overview','genres','keywords','cast','crew']]

movies.head(1)

movies.isnull().sum() ## in the overview column we have 3 missing values 

movies.dropna(inplace=True) ## so we drop the 3 missing values 

movies.isnull().sum() ## now we dont have a missing values 

movies.duplicated().sum() ## check the duplicate values 

movies.iloc[0].genres

#{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}

#[ Action ,Adventure,Fantasy,Sci-fi] ## i want to this formate so lets do the code 


import ast

def convert (obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

convert('{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}')

### in this the error is "string indices must be integers" to remove this we
# import the "ast"
# "ast.literal_eval"  
# then we have the result like this as follow 

import ast
ast.literal_eval('{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}')

movies['genres'].apply(convert)

movies['genres']=movies['genres'].apply(convert)

movies.head(1)

movies['keywords']=movies['keywords'].apply(convert)

movies.head(1)

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else: 
             break
    return L


movies['cast'].apply(convert3)

movies['cast']=movies['cast'].apply(convert3)

movies.head(1)


def fetch_director (obj):
    L = []
    for i in ast.literal_eval(obj):
        if i ['job'] =='Director':
            L.append(i['name'])
            break
    return L

movies['crew']=movies['crew'].apply(fetch_director)

movies.head(1)

movies['overview'][0]

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])## Removing the spaces between the two name and this is the code 

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x]) ## applying on this column 

movies.head() # we get the final output 

movies['tags']= movies['overview'] +movies['genres']+movies['keywords']+movies['cast']+movies['crew']

movies.head()

new_df=movies[['movie_id','title','tags']]
new_df

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


new_df.head()

new_df['tags'][0]

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

new_df

import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y) 

new_df['tags']=new_df['tags'].apply(stem)

new_df['tags'][0]

new_df['tags'][1]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()

cv.fit_transform(new_df['tags']).toarray().shape

vectors

vectors[0]

cv.get_feature_names()

ps.stem('loving')

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

similarity.shape

 similarity[1]

def recommend(movies):
    movies_index =new_df[new_df['title']== movies].index[0]
    distances= similarity[movies_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    

recommend('Avatar')

sorted(similarity[0],reverse=True)[1:6]

list(enumerate(similarity[0]))[1:6]

sorted(list(enumerate(similarity[0])),reverse=True)[1:6]

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6] 

import pickle

new_df['title'].values

new_df.to_dict()

pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))
### jupitor Notbook end  ###

### pycham start ###
import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/w500/" + data ['poster_path']

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id= movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movies_posters

movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity= pickle.load(open('similarity.pkl','rb'))

st.title('Movies Recommendor System')

selected_movie_name = st.selectbox(
'Hi Dewa welcome to the word of web-page',
movies['title'].values)

if st.button('Recommend'):
    names,posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
    st.image(posters[4])
## pycham end ###

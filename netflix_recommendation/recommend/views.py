from django.shortcuts import render
from django.http import HttpResponse
from netflix_recommendation.settings import BASE_DIR

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Recommendation System
pd.set_option('display.max_colwidth', None)
netflix_overall = pd.read_csv('static/netflix_titles.csv')

#removing stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
netflix_overall['description'] = netflix_overall['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(netflix_overall['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(netflix_overall.index, index=netflix_overall['show_id'])
#.drop_duplicates()

def get_recommendations(id, cosine_sim=cosine_sim):
    idx = indices[id]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]

#Create your wordcloud
def word_cloud(genres):
    gen=[]

    for i in genres:
        i=list(i.split(','))
    for j in i:
        gen.append(j.replace(' ',""))

    text = list(set(gen))
    wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(str(text))
    plt.rcParams['figure.figsize'] = (13, 13)
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    wordcloud.to_file("static/word.jpeg")
    return wordcloud

# Create your views here.

def index(request):
    compressed = netflix_overall[['show_id','title']]
    word = word_cloud(list(netflix_overall['listed_in']))
    context={'data':compressed,'image':word}
    return render(request,'index.html',context)


def recommend(request):
    show_id = int(request.POST['show'])
    try:
        show_name = netflix_overall[netflix_overall['show_id']==show_id]['title'].to_string(index=False).strip()
        description = netflix_overall[netflix_overall['show_id']==show_id]['description'].to_string(index=False).strip()
        director = netflix_overall[netflix_overall['show_id']==show_id]['director'].to_string(index=False).strip()
        recommendations = get_recommendations(show_id)
        context = {'title':show_name,'description':description,'director':director,'recommended':recommendations}
    except:
        context={'message':'Movie Not Found'}
    return render(request,'recommend.html',context)


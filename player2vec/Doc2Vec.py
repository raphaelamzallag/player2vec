#!/usr/bin/env python
# coding: utf-8

# In[71]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import umap.umap_ as umap
import ast


def list_of_list(df):
    df['actions'] = df['actions'].apply(lambda x: ast.literal_eval(x))
    lst = []
    for action in df['actions']:
        lst.append(action)
    return lst
    
def doc2vec(df, size, window, min_count, epochs, alpha):
    sentences = list_of_list(df)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    doc2vec = Doc2Vec(documents, vector_size=size, window=window, min_count=min_count, epochs=epochs, alpha=alpha)
    return doc2vec

def X_ndarray(doc2vec):
    values = doc2vec.docvecs
    lst = []
    for i in range(len(doc2vec.docvecs)):
        lst.append(values[i])
    X = np.array(lst)
    return X

def dataframe(X, base):
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['player_name'] = base['player_name']
    df['team'] = base['team']
    df['position'] = base['position']
    df['foot'] = base['foot']
    df['goals'] = base['goals']
    df['agg_position'] = base['agg_position']
    return df

def tsne(doc2vec, label):
    # Dimension reduction
    tsne = TSNE(n_components=2)
    X = X_ndarray(doc2vec)
    X_tsne = tsne.fit_transform(X)
    # Dataframe 
    df = dataframe(X_tsne, label)
    # Plot
    fig = px.scatter(df, x='x', y='y', hover_name='player', color=df.label)
    return fig.show()

def umap_2d(doc2vec, base, label):
    # Dimension reduction
    umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
    X = X_ndarray(doc2vec)
    proj_2d = umap_2d.fit_transform(X)
    # Dataframe 
    df  = dataframe(proj_2d, base)
    # Plot
    fig_2d = px.scatter(df, x='x', y='y', hover_data=['player_name','position', 'foot', 'team'], color=df[label])
    fig_2d.update_traces(marker={'size': 5})
    return df, fig_2d.show()

def umap_3d(doc2vec, base, label):
    # Dimension reduction
    umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)
    X = X_ndarray(doc2vec)
    proj_3d = umap_3d.fit_transform(X)
    # Dataframe 
    df  = dataframe(proj_3d, base)
    # Plot
    fig_3d = px.scatter(df, x='x', y='y', z='z', size='foot', hover_data=['player_name','position', 'foot', 'team'], color=df[label])
    fig_3d.update_traces(marker={'size': 5})
    return fig_3d.show()


# In[72]:


df = pd.read_csv('player2vec_label_agg.csv', encoding='utf-8')
df.head()


# In[73]:


model = doc2vec(df, 32, 5, 10, 10, 0.1)


# In[74]:


df_final, show = umap_2d(model, df, 'agg_position')
df_final


# In[75]:


#df_final.to_csv('player2vec_final_df.csv')


# In[76]:


joueur = ['Messi', 'Cristiano Ronaldo', 'Thierry Henry', 'Varane', 'Daniel Alves', 'Kevin De', 'Mbapp', 'Griezmann', 'Laporte', 'Neuer']

players = []
for i in joueur:
    for n in range(len(df['player_name'])):
        if i in df['player_name'].iloc[n]:
            players.append((df['Unnamed: 0'].iloc[n], df['player_name'].iloc[n]))
players


# In[77]:


dict_similar = {}

for i in players:
    dict_similar.keys
    tmp = model.dv.most_similar(model[i[0]], topn=20)
    players_similar = []
    for n in tmp[1:]:
        players_similar.append(df_final.iloc[n[0]]['player_name'])
    dict_similar[i[1]] = players_similar

dict_similar


# In[79]:


import joblib


# In[80]:


filename = 'finalized_model.sav'
joblib.dump(model, filename)


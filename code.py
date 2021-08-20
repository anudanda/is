import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from skleran.metrics.pairwise import cosine_similarity
import pyodbc
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer
import matplotlib.pyplot as plt

odbc_str= 'DSN='+mydsn+';UID='+myuser+';PWD='+mypass
conn1 = pyodbc.connect(odbc_str, autocommit = true)

q = """ """
d = pd.read_sql(q, conn1)

sym = []
df[] = df[].isin(sym)

def abc():


f = lambda x: abc()
q = [f(x) for x in queries]

#Create lookup
q_lookup = pd.Series(q, index-queries)

df['q_modif'] = q_lookup[df['q']].values

define priors:
p = 1/df[''].nunqiue()
a = p
b = 1-p

posteriors:
a = a+df[]
b = b+ df[]-df[]

df[] = np.sqrt(a/b*(a+b+1))

# Create sparse Matrix: row = q_index, col = u_index, value=weight
X = (df['weight'].tolist(),(df['q_index'].tolist(), df['u_index'].tolist())
X_csr = csr_matrix(X, shape = (len(q_lookup), len(p_lookup)))

similarities = cosine_similarity(X_csr, dense_output = True)

def most_sim(q, topn=10):
  idx = q_lookup(query)
  x = similarities[idx, :]
  top_n_indx = np.argsort(x)[-(to_n+1): -1]
  top_queries = pd.Series(q_lookup)[top_n_indx][::-1]
  weights = x[top_n_idx][::-1]
  return(pd.DF("q":top_queries.index, "S":weights})
  
def make_graph(X_sim, nodes, min_sim):

  G = nx.Graph()
  G.add_nodes_from(nodes)
  idx = np.where(X_sim > min_sim)
  edges = [(nodes[x[0]], nodes[x[1]]) for x in zip(idx[0], idx[1]) if x[0] != x[1]]
  G.add_edges_from(edges)
  return(G)
  
def get_neighbours(G, x, n_deg=2):
  out = []
  iter_vals =[x]
  
  for i in np.arange(n_deg):
   for n in iter_vals:
     neigh = list(G.neigh(n))
     out.extend(neigh)
     iter_vals = neigh
  return (list(set(out))

def get_topics(X_sim, seed_topics, min_sim, n_deg):
  G= make_graph(X_sim, nodes, min_sim)
  df_out = pd.DF(columns=["qf","clst", "top_q"])
  i=0
  for t in seed_topics:
    if t in df_out["qf"].tolist():
      pass
    else:
      neighbors = [t]+get_neigh(G,t, n_deg)
      d = pd.DF({"qf": neigh, "cluster":i, "top_q": t})
      df_out = df_out.append(i)
      i+=1
 return df_out.groupby("qf", as_index=False).first()
 
 p = df_new.groupby("qf"["total_cnt"]).max()
 pct_search = (p/total_s_cnt).sort_values()
 
 
 df_lab = get_topics(similarities, queries_formatted, pct_search_index[:1000], min_sim=0)
 
 
      
  
  
  
  
  




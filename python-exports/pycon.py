# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # A Beginner's Introduction to Pydata: How to Build a Minimal Recommendation System

# <markdowncell>

# ## Welcome!
# 
# - About us
# - Environment + data files check: [https://us.pycon.org/2013/community/tutorials/28](https://us.pycon.org/2013/community/tutorials/28/)
# - Ipydra check

# <markdowncell>

# ## The recommendation problem
# 
# Recommenders have been around since at least 1992. Today we see different flavours of recommenders, deployed across different verticals: 
# 
# - Amazon
# - Netflix
# - Facebook
# - Last.fm.
# 
# What exactly do they do?
# 
# ### Definitions from the literature
# 
# *In a typical recommender system people provide recommendations as inputs, which
# the system then aggregates and directs to appropriate recipients.* -- Resnick
# and Varian, 1997
# 
# *Collaborative filtering simply means that people collaborate to help one
# another perform filtering by recording their reactions to documents they read.*
# -- Goldberg et al, 1992
# 
# *In its most common formulation, the recommendation problem is reduced to the
# problem of estimating ratings for the items that have not been seen by a
# user. Intuitively, this estimation is usually based on the ratings given by this
# user to other items and on some other information [...] Once we can estimate
# ratings for the yet unrated items, we can recommend to the user the item(s) with
# the highest estimated rating(s).* -- Adomavicius and Tuzhilin, 2005
# 
# *Driven by computer algorithms, recommenders help consumers
# by selecting products they will probably like and might buy
# based on their browsing, searches, purchases, and preferences.* -- Konstan and Riedl, 2012

# <markdowncell>

# ### Notation
# 
# - $U$ is the set of users in our domain. Its size is $|U|$.
# - $I$ is the set of items in our domain. Its size is $|I|$.
# - $I(u)$ is the set of items that user $u$ has rated.
# - $-I(u)$ is the complement of $I(u)$ i.e., the set of items not yet seen by user $u$.
# - $U(i)$ is the set of users that have rated item $i$.
# - $-U(i)$ is the complement of $U(i)$.

# <markdowncell>

# ### Goal of a recommendation system
# 
# $$ 
# \newcommand{\argmax}{\mathop{\rm argmax}\nolimits}
# \forall{u \in U},\; i^* = \argmax_{i \in -I(u)} [S(u,i)] 
# $$

# <markdowncell>

# ### Problem statement
# 
# The recommendation problem in its most basic form is quite simple to define:
# 
# ```
# |-------------------+-----+-----+-----+-----+-----|
# | user_id, movie_id | m_1 | m_2 | m_3 | m_4 | m_5 |
# |-------------------+-----+-----+-----+-----+-----|
# | u_1               | ?   | ?   | 4   | ?   | 1   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_2               | 3   | ?   | ?   | 2   | 2   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_3               | 3   | ?   | ?   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_4               | ?   | 1   | 2   | 1   | 1   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_5               | ?   | ?   | ?   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_6               | 2   | ?   | 2   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_7               | ?   | ?   | ?   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_8               | 3   | 1   | 5   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_9               | ?   | ?   | ?   | ?   | 2   |
# |-------------------+-----+-----+-----+-----+-----|
# ```
# 
# *Given a partially filled matrix of ratings ($|U|x|I|$), estimate the missing values.*

# <markdowncell>

# ### Content-based filtering
# 
# Generic expression (notice how this is kind of a 'row-based' approach):
# 
# $$ 
# \newcommand{\aggr}{\mathop{\rm aggr}\nolimits}
# r_{u,i} = \aggr_{i' \in I(u)} [r_{u,i'}] 
# $$

# <markdowncell>

# #### Content-based: simple ratings-based recommendations
# 
# Purely based on ratings information.
# 
# $$ r_{u,i} = \bar r_u = \frac{\sum_{i' \in I(u)} r_{u,i'}}{|I(u)|} $$

# <markdowncell>

# ### Collaborative filtering
# 
# Generic expression (notice how this is kind of a 'col-based' approach):
# 
# $$ 
# \newcommand{\aggr}{\mathop{\rm aggr}\nolimits}
# r_{u,i} = \aggr_{u' \in U(i)} [r_{u',i}] 
# $$

# <markdowncell>

# #### Collaborative filtering: simple ratings-based recommendations
# 
# Also based solely on ratings information.
# 
# $$ r_{u,i} = \bar r_i = \frac{\sum_{u' \in U(i)} r_{u',i}}{|U(i)|} $$

# <markdowncell>

# ### Hybrid solutions
# 
# The literature has lots of examples of systems that try to combine the strengths
# of the two main approaches. This can be done in a number of ways:
# 
# - Combine the predictions of a content-based system and a collaborative system.
# - Incorporate content-based techniques into a collaborative approach.
# - Incorporarte collaborative techniques into a content-based approach.
# - Unifying model.

# <markdowncell>

# ### Challenges
# 
# #### Availability of item metadata
# 
# Content-based techniques are limited by the amount of metadata that is available
# to describe an item. There are domains in which feature extraction methods are
# expensive or time consuming, e.g., processing multimedia data such as graphics,
# audio/video streams. In the context of grocery items for example, it's often the
# case that item information is only partial or completely missing. Examples
# include:
# 
# - Ingredients
# - Nutrition facts
# - Brand
# - Description
# - County of origin
# 
# #### New user problem
# 
# A user has to have rated a sufficient number of items before a recommender
# system can have a good idea of what their preferences are. In a content-based
# system, the aggregation function needs ratings to aggregate.
# 
# #### New item problem
# 
# Collaborative filters rely on an item being rated by many users to compute
# aggregates of those ratings. Think of this as the exact counterpart of the new
# user problem for content-based systems.
# 
# #### Data sparsity
# 
# When looking at the more general versions of content-based and collaborative
# systems, the success of the recommender system depends on the availability of a
# critical mass of user/item iteractions. We get a first glance at the data
# sparsity problem by quantifying the ratio of existing ratings vs $|U|x|I|$. A
# highly sparse matrix of interactions makes it difficult to compute similarities
# between users and items. As an example, for a user whose tastes are unusual
# compared to the rest of the population, there will not be any other users who
# are particularly similar, leading to poor recommendations.

# <markdowncell>

# ## About this tutorial
# We've put this together from our experience and a number of sources, please check the references at the bottom of this document.
# 
# ### What this tutorial is
# 
# The goal of this tutorial is to provide you with a hands-on overview of two of
# the main libraries from the scientific and data analysis communities. We're going to
# use:
# 
# - ipython -- [ipython.org](http://ipython.org/)
# - numpy -- [numpy.org](http://www.numpy.org)
# - pandas -- [pandas.pydata.org](http://pandas.pydata.org/)
# - (bonus) pytables -- [pytables.org](http://www.pytables.org)
# 
# ### What this tutorial is not
# 
# - An exhaustive overview of the recommendation literature
# - A set of recipes that will win you the next Netflix/Kaggle/? challenge.

# <markdowncell>

# ## Roadmap
# 
# What exactly are we going to do? Here's high-level overview:
# 
# - learn about NumPy arrays
# - learn about DataFrames
# - iterate over a few implementations of a minimal reco engine
# - challenge

# <markdowncell>

# ### Dataset
# 
# MovieLens from GroupLens Research: [grouplens.org](http://www.grouplens.org/)
# 
# The MovieLens 1M data set contains 1 million ratings collected from 6000 users
# on 4000 movies.
# 

# <markdowncell>

# ### Flow chart: the big picture

# <codecell>

from IPython.core.display import Image 
Image(filename='./pycon_reco_flow.png')

# <markdowncell>

# ## NumPy: Numerical Python
# 
# ### What is it?
# 
# *It is a Python library that provides a multidimensional array object, various
# derived objects (such as masked arrays and matrices), and an assortment of
# routines for fast operations on arrays, including mathematical, logical, shape
# manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear
# algebra, basic statistical operations, random simulation and much more.*

# <codecell>

import numpy as np

# set some print options
np.set_printoptions(precision=4)
np.set_printoptions(threshold=5)
np.set_printoptions(suppress=True)

# init random gen
np.random.seed(2)

# <markdowncell>

# ### NumPy's basic data structure: the ndarray
# 
# Think of ndarrays as the building blocks for pydata. A multidimensional array
# object that acts as a container for data to be passed between algorithms. Also,
# libraries written in a lower-level language, such as C or Fortran, can operate
# on the data stored in a NumPy array without copying any data.

# <codecell>

import numpy as np

# build an array using the array function
arr = np.array([0, 9, 5, 4, 3])
arr

# <markdowncell>

# ### Array creation examples
# 
# There are several functions that are used to create new arrays:
# 
# - `np.array`
# - `np.asarray`
# - `np.arange`
# - `np.ones`
# - `np.ones_like`
# - `np.zeros`
# - `np.zeros_like`
# - `np.empty`
# - `np.random.randn` and other funcs from the random module

# <codecell>

np.zeros(4)

# <codecell>

np.ones(4)

# <codecell>

np.empty(4)

# <codecell>

np.arange(4)

# <markdowncell>

# ### dtype and shape
# 
# NumPy's arrays are containers of homogeneous data, which means all elements are
# of the same type. The 'dtype' propery is an object that specifies the data type
# of each element. The 'shape' property is a tuple that indicates the size of each
# dimension.

# <codecell>

arr = np.random.randn(5)
arr

# <codecell>

arr.dtype

# <codecell>

arr.shape

# <codecell>

# you can be explicit about the data type that you want
np.empty(4, dtype=np.int32)

# <codecell>

np.array(['numpy','pandas','pytables'], dtype=np.string_)

# <codecell>

float_arr = np.array([4.4, 5.52425, -0.1234, 98.1], dtype=np.float64)
# truncate the decimal part
float_arr.astype(np.int32)

# <markdowncell>

# ### Indexing and slicing
# 
# #### Just what you would expect from Python

# <codecell>

arr = np.array([0, 9, 1.02, 4, 64])
arr[3]

# <codecell>

arr[1:3]

# <codecell>

# set the last two elements to 555
arr[-2:] = 555
arr

# <markdowncell>

# #### Indexing behaviour for multidimensional arrays
# 
# A good way to think about indexing in multidimensional arrays is that you are
# moving along the values of the shape property. So, a 4d array `arr_4d`, with a
# shape of `(w,x,y,z)` will result in indexed views such that:
# 
# - `arr_4d[i].shape == (x,y,z)`
# - `arr_4d[i,j].shape == (y,z)`
# - `arr_4d[i,j,k].shape == (z,)`
# 
# For the case of slices, what you are doing is selecting a range of elements
# along a particular axis:

# <codecell>

arr_2d = np.array([[5,3,4],[0,1,2],[1,1,10],[0,0,0.1]])
arr_2d

# <codecell>

# get the first row
arr_2d[0]

# <codecell>

# get the first column
arr_2d[:,0]

# <codecell>

# get the first two rows
arr_2d[:2]

# <markdowncell>

# #### Careful, it's a view!
# 
# A slice does not return a copy, which means that any modifications will be
# reflected in the source array. This is a design feature of NumPy to avoid memory
# problems.

# <codecell>

arr = np.array([0, 3, 1, 4, 64])
arr

# <codecell>

slice = arr[2:4]
slice[1] = 99
arr

# <markdowncell>

# #### (Fancy) Boolean indexing
# 
# Boolean indexing allows you to select data subsets of an array that satisfy a
# given condition.

# <codecell>

arr = np.array([10, 20])
idx = np.array([True, False])
arr[idx]

# <codecell>

arr_2d = np.random.randn(4,8)
arr_2d

# <codecell>

arr_2d < 0

# <codecell>

arr_2d[arr_2d < 0]

# <codecell>

arr_2d[(arr_2d > -0.1) & (arr_2d < 0)]

# <codecell>

arr_2d[arr_2d < 0] = 0
arr_2d

# <markdowncell>

# #### (Fancy) list-of-locations indexing
# 
# Fancy indexing is indexing with integer arrays.

# <codecell>

arr = np.arange(18).reshape(6,3)
arr

# <codecell>

# fancy selection of rows in a particular order
arr[[0,4,4]]

# <codecell>

# index into individual elements and flatten
arr[[5,3,1],[2,1,0]]

# <codecell>

# select a submatrix
arr[np.ix_([5,3,1],[2,1])]

# <markdowncell>

# --> Go to question set

# <markdowncell>

# ### Vectorization
# Vectorization is at the heart of NumPy and it enables us to express operations
# without writing any for loops.  Operations between arrays with equal shapes are
# performed element-wise.

# <codecell>

arr = np.array([0, 9, 1.02, 4, 32])
arr - arr

# <codecell>

arr * arr

# <markdowncell>

# ### Broadcasting Rules
# 
# Vectorized operations between arrays of different sizes and between arrays and
# scalars are subject to the rules of broadcasting. The idea is quite simple in
# many cases:

# <codecell>

arr = np.array([0, 9, 1.02, 4, 64])
5 * arr 

# <codecell>

10 + arr

# <codecell>

arr ** .5

# <codecell>

arr = np.random.randn(4,2)
arr

# <codecell>

mean_row = np.mean(arr, axis=0)
mean_row

# <codecell>

centered_rows = arr - mean_row
np.mean(centered_rows, axis=0)

# <codecell>

mean_col = np.mean(arr, axis=1)
mean_col

# <codecell>

centered_cols = arr - mean_col

# <codecell>

# make the 1-D array a column vector
mean_col.reshape((4,1))

# <codecell>

centered_cols = arr - mean_col.reshape((4,1))
centered_cols.mean(axis=1)

# <markdowncell>

# #### A note about NANs: 
# 
# Per the floating point standard IEEE 754, NaN is a floating point value that, by definition, is not equal to any other floating point value.

# <codecell>

np.nan != np.nan

# <codecell>

np.array([10,6,5,4,np.nan,1,np.nan]) == np.nan

# <codecell>

np.isnan(np.array([10,6,5,4,np.nan,1,np.nan]))

# <markdowncell>

# --> Go to question set

# <markdowncell>

# ## pandas: Python Data Analysis Library
# 
# ### What is it?
# 
# *Python has long been great for data munging and preparation, but less so for
# data analysis and modeling. pandas helps fill this gap, enabling you to carry
# out your entire data analysis workflow in Python without having to switch to a
# more domain specific language like R.*
# 
# The heart of pandas is the DataFrame object for data manipulation. It features:
# 
# - a powerful index object
# - data alignment
# - handling of missing data
# - aggregation with groupby
# - data manipuation via reshape, pivot, slice, merge, join

# <codecell>

import pandas as pd

pd.set_printoptions(precision=3, notebook_repr_html=True)

# <markdowncell>

# ### Series: labelled arrays
# 
# The pandas Series is the simplest datastructure to start with. It is a subclass
# of ndarray that supports more meaninful indices.

# <markdowncell>

# #### Let's look at some creation examples for Series

# <codecell>

import pandas as pd

values = np.array([2.0, 1.0, 5.0, 0.97, 3.0, 10.0, 0.0599, 8.0])
ser = pd.Series(values)
print ser

# <codecell>

values = np.array([2.0, 1.0, 5.0, 0.97, 3.0, 10.0, 0.0599, 8.0])
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
ser = pd.Series(data=values, index=labels)
print ser

# <codecell>

movie_rating = {
    'age': 1,
    'gender': 'F',
    'genres': 'Drama',
    'movie_id': 1193,
    'occupation': 10,
    'rating': 5,
    'timestamp': 978300760,
    'title': "One Flew Over the Cuckoo's Nest (1975)",
    'user_id': 1,
    'zip': '48067'
    }
ser = pd.Series(movie_rating)
print ser

# <codecell>

ser.index

# <codecell>

ser.values

# <markdowncell>

# #### Series indexing

# <codecell>

ser[0]

# <codecell>

ser['gender']

# <codecell>

ser.get_value('gender')

# <markdowncell>

# #### Operations between Series with different index objects

# <codecell>

ser_1 = pd.Series(data=[1,3,4], index=['A', 'B', 'C'])
ser_2 = pd.Series(data=[5,5,5], index=['A', 'G', 'C'])
print ser_1 + ser_2

# <markdowncell>

# ### DataFrame
# 
# The DataFrame is the 2-dimensional version of a Series.
# 
# #### Let's look at some creation examples for DataFrame
# 
# You can think of it as a spreadsheet whose columns are Series objects.

# <codecell>

# build from a dict of equal-length lists or ndarrays
pd.DataFrame({'col_1': [0.12, 7, 45, 10], 'col_2': [0.9, 9, 34, 11]})

# <markdowncell>

# You can explicitly set the column names and index values as well.

# <codecell>

pd.DataFrame(data={'col_1': [0.12, 7, 45, 10], 'col_2': [0.9, 9, 34, 11]},
             columns=['col_1', 'col_2', 'col_3'])

# <codecell>

pd.DataFrame(data={'col_1': [0.12, 7, 45, 10], 'col_2': [0.9, 9, 34, 11]},
             columns=['col_1', 'col_2', 'col_3'],
             index=['obs1', 'obs2', 'obs3', 'obs4'])

# <markdowncell>

# You can also think of it as a dictionary of Series objects.

# <codecell>

movie_rating = {
    'gender': 'F',
    'genres': 'Drama',
    'movie_id': 1193,
    'rating': 5,
    'timestamp': 978300760,
    'user_id': 1,
    }
ser_1 = pd.Series(movie_rating)
ser_2 = pd.Series(movie_rating)
df = pd.DataFrame({'r_1': ser_1, 'r_2': ser_2})
df.columns.name = 'rating_events'
df.index.name = 'rating_data'
df

# <codecell>

df = df.T
df

# <codecell>

df.columns 

# <codecell>

df.index

# <markdowncell>

# #### Adding/Deleting entries

# <codecell>

df = pd.DataFrame({'r_1': ser_1, 'r_2': ser_2})
df.drop('genres', axis=0)

# <codecell>

df.drop('r_1', axis=1)

# <codecell>

# careful with the order here
df['r_3'] = ['F', 'Drama', 1193, 5, 978300760, 1]
df

# <markdowncell>

# --> Go to question set

# <markdowncell>

# #### DataFrame indexing

# <markdowncell>

# You can index into a column using it's label, or with dot notation

# <codecell>

df['r_1']

# <codecell>

df.r_1

# <markdowncell>

# You can also use multiple columns to select a subset of them:

# <codecell>

df[['r_2', 'r_1']]

# <markdowncell>

# The .ix method gives you the most flexibility to index into certain rows, or
# even rows and columns:

# <codecell>

df.ix['gender']

# <codecell>

df.ix[0]

# <codecell>

df.ix[:2]

# <codecell>

df.ix[:2, 'r_1']

# <codecell>

df.ix[:2, ['r_1', 'r_2']]

# <markdowncell>

# --> Go to question set

# <markdowncell>

# ## The MovieLens dataset: loading and first look
# 
# Loading of the MovieLens dataset here is based on the intro chapter of 'Python
# for Data Analysis".
# 
# The MovieLens data is spread across three files. Using the `pd.read_table`
# method we load each file:

# <codecell>

import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('data/ml-1m/users.dat',
                      sep='::', header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('data/ml-1m/ratings.dat',
                        sep='::', header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('data/ml-1m/movies.dat',
                       sep='::', header=None, names=mnames)

# show how one of them looks
ratings.head(5)

# <markdowncell>

# Using `pd.merge` we get it all into  one big DataFrame.

# <codecell>

movielens = pd.merge(pd.merge(ratings, users), movies)
movielens

# <markdowncell>

# ## Evaluation
# 
# Before we attempt to express the basic equations for content-based or
# collaborative filtering we need a basic mechanism to evaluate the performance of
# our engine.

# <markdowncell>

# ### Evaluation: split ratings into train and test sets
# 
# This subsection will generate training and testing sets for evaluation. You do
# not need to understand every single line of code, just the general gist:
# 
# - take a smaller sample from the full 1M dataset for speed reasons;
# - make sure that we have at least 2 ratings per user in that subset;
# - split the result into training and testing sets.

# <codecell>

# let's work with a smaller subset for speed reasons
movielens = movielens.ix[np.random.choice(movielens.index, size=10000, replace=False)]
print movielens.shape
print movielens.user_id.nunique()
print movielens.movie_id.nunique()

# <codecell>

user_ids_larger_1 = pd.value_counts(movielens.user_id, sort=False) > 1
movielens = movielens[user_ids_larger_1[movielens.user_id]]
print movielens.shape
np.all(movielens.user_id.value_counts() > 1)

# <markdowncell>

# We now generate train and test subsets using groupby and apply.

# <codecell>

def assign_to_set(df):
    sampled_ids = np.random.choice(df.index,
                                   size=np.int64(np.ceil(df.index.size * 0.2)),
                                   replace=False)
    df.ix[sampled_ids, 'for_testing'] = True
    return df

movielens['for_testing'] = False
grouped = movielens.groupby('user_id', group_keys=False).apply(assign_to_set)
movielens_train = movielens[grouped.for_testing == False]
movielens_test = movielens[grouped.for_testing == True]
print movielens_train.shape
print movielens_test.shape
print movielens_train.index & movielens_test.index

# <markdowncell>

# Store these two sets in text files:

# <codecell>

movielens_train.to_csv('data/movielens_train.csv')
movielens_test.to_csv('data/movielens_test.csv')

# <markdowncell>

# ### Evaluation: performance criterion
# 
# Performance evaluation of recommendation systems is an entire topic all in
# itself. Some of the options include:
# 
# - RMSE: $\sqrt{\frac{\sum(\hat y - y)^2}{n}}$
# - Precision / Recall / F-scores
# - ROC curves
# - Cost curves

# <codecell>

def compute_rmse(y_pred, y_true):
    """ Compute Root Mean Squared Error. """
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))

# <markdowncell>

# ### Evaluation: the 'evaluate' method

# <codecell>

def evaluate(estimate_f):
    """ RMSE-based predictive performance evaluation with pandas. """
    ids_to_estimate = zip(movielens_test.user_id, movielens_test.movie_id)
    estimated = np.array([estimate_f(u,i) for (u,i) in ids_to_estimate])
    real = movielens_test.rating.values
    return compute_rmse(estimated, real)

# <markdowncell>

# ## Minimal reco engine v1.0: simple mean ratings

# <markdowncell>

# ### Content-based filtering using mean ratings
# 
# With this table-like representation of the ratings data, a basic content-based
# filter becomes a one-liner function.

# <codecell>

def estimate1(user_id, item_id):
    """ Simple content-filtering based on mean ratings. """
    return movielens_train.ix[movielens_train.user_id == user_id, 'rating'].mean()

print 'RMSE for estimate1: %s' % evaluate(estimate1)

# <markdowncell>

# ### Collaborative-based filtering using mean ratings

# <codecell>

def estimate2(user_id, movie_id):
    """ Simple collaborative filter based on mean ratings. """
    ratings_by_others = movielens_train[movielens_train.movie_id == movie_id]
    if ratings_by_others.empty: return 3.0
    #return  movielens_train.ix[movielens_train.movie_id == movie_id, 'rating'].mean()
    return ratings_by_others.rating.mean()

print 'RMSE for estimate2: %s' % evaluate(estimate2)

# <markdowncell>

# --> Go to question set

# <markdowncell>

# ## More formulas!
# 
# Here are some basic ways in which we can generalize the simple mean-based algorithms we discussed before. 

# <markdowncell>

# ### Content-based: generalizations of the aggregation function
# 
# Possibly incorporating metadata about items.
# 
# $$ r_{u,i} = k \sum_{i' \in I(u)} sim(i, i') \; r_{u,i'} $$
# 
# $$ r_{u,i} = \bar r_u + k \sum_{i' \in I(u)} sim(i, i') \; (r_{u,i'} - \bar r_u) $$
# 
# Here $k$ is a normalizing factor,
# 
# $$ k = \frac{1}{\sum_{i' \in I(u)} |sim(i,i')|} $$
# 
# and $\bar r_u$ is the average rating of user u:
# 
# $$ \bar r_u = \frac{\sum_{i \in I(u)} r_{u,i}}{|I(u)|} $$

# <markdowncell>

# ### Collaborative filtering: generalizations of the aggregation function
# 
# Possibly incorporating metadata about users.
# 
# $$ r_{u,i} = k \sum_{u' \in U(i)} sim(u, u') \; r_{u',i} $$
# 
# $$ r_{u,i} = \bar r_u + k \sum_{u' \in U(i)} sim(u, u') \; (r_{u',i} - \bar r_u) $$
# 
# Here $k$ is a normalizing factor,
# 
# $$ k = \frac{1}{\sum_{u' \in U(i)} |sim(u,u')|} $$
# 
# and $\bar r_u$ is the average rating of user u:
# 
# $$ \bar r_u = \frac{\sum_{i \in I(u)} r_{u,i}}{|I(u)|} $$

# <markdowncell>

# ## Aggregation in pandas

# <markdowncell>

# ### Groupby
# 
# The idea of groupby is that of *split-apply-combine*:
# 
# - split data in an object according to a given key;
# - apply a function to each subset;
# - combine results into a new object.

# <codecell>

print movielens.groupby('gender')['rating'].mean()

# <codecell>

print movielens.groupby(['gender', 'age'])['rating'].mean()

# <markdowncell>

# ### Pivoting
# 
# Let's start with a simple pivoting example that does not involve any
# aggregation. We can extract a ratings matrix as follows:

# <codecell>

# transform the ratings frame into a ratings matrix
ratings_mtx_df = movielens.pivot_table(values='rating',
                                       rows='user_id',
                                       cols='movie_id')
# with an integer axis index only label-based indexing is possible
ratings_mtx_df.ix[ratings_mtx_df.index[-15:],ratings_mtx_df.columns[:15]]

# <markdowncell>

# The more interesting case with `pivot_table` is as an interface to
# `groupby`:

# <codecell>

by_gender_title = movielens.groupby(['gender', 'title'])['rating'].mean()
print by_gender_title

# <codecell>

by_gender_title = movielens.groupby(['gender', 'title'])['rating'].mean().unstack('gender')
by_gender_title.head(10)

# <codecell>

by_gender_title = movielens.pivot_table('rating', rows='title', cols='gender')
by_gender_title.head(10)

# <markdowncell>

# ## Minimal reco engine v1.1: implicit sim functions

# <markdowncell>

# We're going to need a user index from the users portion of the dataset. This will allow us to retrieve information given a specific user_id in a more convenient way:

# <codecell>

user_info = users.set_index('user_id')
user_info.head(5)

# <rawcell>

# With this in hand, we can now ask what the gender of a particular user_id is like so:

# <codecell>

user_id = 6
user_info.ix[user_id, 'gender']

# <markdowncell>

# ### Collaborative-based filtering using implicit sim functions
# 
# Using the pandas aggregation framework we will build a collaborative filter that estimates ratings using an implicit `sim(u,u')` function to compare different users.

# <codecell>

def estimate3(user_id, movie_id):
    """ Collaborative filtering using an implicit sim(u,u'). """
    ratings_by_others = movielens_train[movielens_train.movie_id == movie_id]
    if ratings_by_others.empty: return 3.0
    means_by_gender = ratings_by_others.pivot_table('rating', rows='movie_id', cols='gender')
    user_gender = user_info.ix[user_id, 'gender']
    if user_gender in means_by_gender.columns: 
        return means_by_gender.ix[movie_id, user_gender]
    else:
        return means_by_gender.ix[movie_id].mean()

print 'RMSE for reco3: %s' % evaluate(estimate3)

# <markdowncell>

# At this point it seems worthwhile to write a `learn` that pre-computes whatever datastructures we need at estimation time.

# <codecell>

class Reco3:
    """ Collaborative filtering using an implicit sim(u,u'). """

    def learn(self):
        """ Prepare datastructures for estimation. """
        self.means_by_gender = movielens_train.pivot_table('rating', rows='movie_id', cols='gender')

    def estimate(self, user_id, movie_id):
        """ Mean ratings by other users of the same gender. """
        if movie_id not in self.means_by_gender.index: return 3.0
        user_gender = user_info.ix[user_id, 'gender']
        if ~np.isnan(self.means_by_gender.ix[movie_id, user_gender]):
            return self.means_by_gender.ix[movie_id, user_gender]
        else:
            return self.means_by_gender.ix[movie_id].mean()

reco = Reco3()
reco.learn()
print 'RMSE for reco3: %s' % evaluate(reco.estimate)

# <codecell>

class Reco4:
    """ Collaborative filtering using an implicit sim(u,u'). """

    def learn(self):
        """ Prepare datastructures for estimation. """
        self.means_by_age = movielens_train.pivot_table('rating', rows='movie_id', cols='age')

    def estimate(self, user_id, movie_id):
        """ Mean ratings by other users of the same age. """
        if movie_id not in self.means_by_age.index: return 3.0
        user_age = user_info.ix[user_id, 'age']
        if ~np.isnan(self.means_by_age.ix[movie_id, user_age]):
            return self.means_by_age.ix[movie_id, user_age]
        else:
            return self.means_by_age.ix[movie_id].mean()

reco = Reco4()
reco.learn()
print 'RMSE for reco4: %s' % evaluate(reco.estimate)

# <markdowncell>

# ## Minimal reco engine v1.2: custom similarity functions

# <markdowncell>

# ### A few similarity functions
# 
# These were all written to operate on two pandas Series, each one representing the rating history of two different users. You can also apply them to any two feature vectors that describe users or items. In all cases, the higher the return value, the more similar two Series are. You might need to add checks for edge cases, such as divisions by zero, etc.
# 
# - Euclidean 'similarity'
# 
# $$ sim(x,y) = \frac{1}{1 + \sqrt{\sum (x - y)^2}}$$

# <codecell>

def euclidean(s1, s2):
    """Take two pd.Series objects and return their euclidean 'similarity'."""
    diff = s1 - s2
    return 1 / (1 + np.sqrt(np.sum(diff ** 2)))

# <markdowncell>

# - Cosine similarity
# 
# $$ sim(x,y) = \frac{(x . y)}{\sqrt{(x . x) (y . y)}} $$

# <codecell>

def cosine(s1, s2):
    """Take two pd.Series objects and return their cosine similarity."""
    return np.sum(s1 * s2) / np.sqrt(np.sum(s1 ** 2) * np.sum(s2 ** 2))

# <markdowncell>

# - Pearson correlation
# 
# $$ sim(x,y) = \frac{(x - \bar x).(y - \bar y)}{\sqrt{(x - \bar x).(x - \bar x) * (y - \bar y)(y - \bar y)}} $$

# <codecell>

def pearson(s1, s2):
    """Take two pd.Series objects and return a pearson correlation."""
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

# <markdowncell>

# - Jaccard similarity
# 
# $$ sim(x,y) = \frac{(x . y)}{(x . x) + (y . y) - (x . y)} $$

# <codecell>

def jaccard(s1, s2):
    dotp = np.sum(s1 * s2)
    return dotp / (np.sum(s1 ** 2) + np.sum(s2 ** 2) - dotp)

def binjaccard(s1, s2):
    dotp = (s1.index & s2.index).size
    return dotp / (s1.sum() + s2.sum() - dotp)

# <markdowncell>

# ### Collaborative-based filtering using custom sim functions

# <codecell>

class Reco5:
    """ Collaborative filtering using a custom sim(u,u'). """

    def learn(self):
        """ Prepare datastructures for estimation. """
        self.all_user_profiles = movielens.pivot_table('rating', rows='movie_id', cols='user_id')

    def estimate(self, user_id, movie_id):
        """ Ratings weighted by correlation similarity. """
        ratings_by_others = movielens_train[movielens_train.movie_id == movie_id]
        if ratings_by_others.empty: return 3.0
        ratings_by_others.set_index('user_id', inplace=True)
        their_ids = ratings_by_others.index
        their_ratings = ratings_by_others.rating
        their_profiles = self.all_user_profiles[their_ids]
        user_profile = self.all_user_profiles[user_id]
        sims = their_profiles.apply(lambda profile: pearson(profile, user_profile), axis=0)
        ratings_sims = pd.DataFrame({'sim': sims, 'rating': their_ratings})
        ratings_sims = ratings_sims[ ratings_sims.sim > 0]
        if ratings_sims.empty:
            return their_ratings.mean()
        else:
            return np.average(ratings_sims.rating, weights=ratings_sims.sim)
        
reco = Reco5()
reco.learn()
print 'RMSE for reco5: %s' % evaluate(reco.estimate)

# <markdowncell>

# ## Mini-Challenge!
# 
# - Not a real challenge
# - Focus on understanding the different versions of our minimal reco
# - Try to mix and match some of the ideas presented to come up with a minimal reco of your own
# - Evaluate it!

# <markdowncell>

# ## PyTables
# ### What is it?
# 
# *PyTables is a package for managing hierarchical datasets and designed to
# efficiently and easily cope with extremely large amounts of data.*
# 
# ### HDF5
# From [hdfgroup.org](http://www.hdfgroup.org):
# *HDF5 is a Hierarchical Data Format consisting of a data format specification and a supporting library implementation.*
# 
# HDF5 files are organized in a hierarchical structure, with two primary structures: groups and datasets.
# 
# - HDF5 group: a grouping structure containing instances of zero or more groups or datasets, together with supporting metadata.
# - HDF5 dataset: a multidimensional array of data elements, together with supporting metadata.

# <markdowncell>

# ### Sample file structure

# <rawcell>

# / (RootGroup) ''
# /historic (Group) ''
# /historic/purchases (Table(79334510,)) 'Purchases table'
#   description := {
#   "product_id": StringCol(itemsize=16, shape=(), dflt='', pos=0),
#   "purchase_date": Float32Col(shape=(), dflt=0.0, pos=1),
#   "purchase_id": StringCol(itemsize=16, shape=(), dflt='', pos=2),
#   "user_id": StringCol(itemsize=16, shape=(), dflt='', pos=3)}
#   byteorder := 'little'
#   chunkshape := (1260,)
#   autoIndex := True
#   colindexes := {
#     "purchase_id": Index(9, full, shuffle, zlib(1)).is_CSI=False,
#     "user_id": Index(9, full, shuffle, zlib(1)).is_CSI=False,
#     "product_id": Index(9, full, shuffle, zlib(1)).is_CSI=False,
#     "purchase_date": Index(9, full, shuffle, zlib(1)).is_CSI=False}
# /itemops (Group) ''
# /userops (Group) ''
# /userops/recent_ui_idxs (CArray(2, 120388)) ''
#   atom := Float64Atom(shape=(), dflt=0.0)
#   maindim := 0
#   flavor := 'numpy'
#   byteorder := 'little'
#   chunkshape := (1, 8192)
# /userops/ui_idxs (CArray(2, 4170272)) ''
#   atom := Float64Atom(shape=(), dflt=0.0)
#   maindim := 0
#   flavor := 'numpy'
#   byteorder := 'little'
#   chunkshape := (1, 16384)
# /userops/inc_mtx (Group) ''
# /userops/inc_mtx/data (Array(1918289,)) ''
#   atom := Float64Atom(shape=(), dflt=0.0)
#   maindim := 0
#   flavor := 'numpy'
#   byteorder := 'little'
#   chunkshape := None
# /userops/inc_mtx/indices (Array(1918289,)) ''
#   atom := Int32Atom(shape=(), dflt=0)
#   maindim := 0
#   flavor := 'numpy'
#   byteorder := 'little'
#   chunkshape := None
# /userops/inc_mtx/indptr (Array(446122,)) ''
#   atom := Int32Atom(shape=(), dflt=0)
#   maindim := 0
#   flavor := 'numpy'
#   byteorder := 'little'
#   chunkshape := None
# /userops/inc_mtx/shape (Array(2,)) ''
#   atom := Int64Atom(shape=(), dflt=0)
#   maindim := 0
#   flavor := 'python'
#   byteorder := 'little'
#   chunkshape := None

# <markdowncell>

# ### File creation

# <codecell>

import tables as tb

h5file = tb.openFile('data/tutorial.h5', mode='w', title='Test file')
h5file

# <markdowncell>

# ### Group and dataset creation

# <codecell>

group_1 = h5file.createGroup(h5file.root, 'group_1', 'Group One')
group_2 = h5file.createGroup('/', 'group_2', 'Group Two')
h5file

# <codecell>

h5file.createArray(group_1, 'random_arr_1', np.random.randn(30),
                   "Just a bunch of random numbers")

# <codecell>

h5file

# <codecell>

h5file.root.group_1.random_arr_1

# <codecell>

h5file.root.group_1.random_arr_1[:5]

# <markdowncell>

# ### Group attributes

# <codecell>

from datetime import datetime
h5file.setNodeAttr(group_1, 'last_modified', datetime.utcnow())
group_1._v_attrs

# <codecell>

h5file.getNodeAttr(group_1,'last_modified')

# <markdowncell>

# ### Handling things that don't fit in memory

# <codecell>

group_3 = h5file.createGroup(h5file.root, 'group_3', 'Group Three')
ndim = 6000000
h5file.createArray(group_3, 'random_group_3',
                   numpy.zeros((ndim,ndim)), "A very very large array")

# <codecell>

rows = 10
cols = 10
earr = h5file.createEArray(group_3, 'EArray', tb.Int8Atom(),
                           (0, cols), "A very very large array, second try.")

for i in range(rows):
    earr.append(numpy.zeros((1, cols)))

# <codecell>

earr

# <markdowncell>

# ## References and further reading
# 
# - Goldberg, D., D. Nichols, B. M. Oki, and D. Terry. “Using Collaborative Filtering to Weave an Information Tapestry.” Communications of the ACM 35, no. 12 (1992): 61–70.
# - Resnick, Paul, and Hal R. Varian. “Recommender Systems.” Commun. ACM 40, no. 3 (March 1997): 56–58. doi:10.1145/245108.245121.
# - Adomavicius, Gediminas, and Alexander Tuzhilin. “Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions.” IEEE Transactions on Knowledge and Data Engineering 17, no. 6 (2005): 734–749. doi:http://doi.ieeecomputersociety.org/10.1109/TKDE.2005.99.
# - Adomavicius, Gediminas, Ramesh Sankaranarayanan, Shahana Sen, and Alexander Tuzhilin. “Incorporating Contextual Information in Recommender Systems Using a Multidimensional Approach.” ACM Trans. Inf. Syst. 23, no. 1 (2005): 103–145. doi:10.1145/1055709.1055714.
# - Koren, Y., R. Bell, and C. Volinsky. “Matrix Factorization Techniques for Recommender Systems.” Computer 42, no. 8 (2009): 30–37.
# - William Wesley McKinney. Python for Data Analysis. O’Reilly, 2012.
# - Toby Segaran. Programming Collective Intelligence. O’Reilly, 2007.
# - Zhou, Tao, Zoltan Kuscsik, Jian-Guo Liu, Matus Medo, Joseph R Wakeling, and Yi-Cheng Zhang. “Solving the Apparent Diversity-accuracy Dilemma of Recommender Systems.” arXiv:0808.2670 (August 19, 2008). doi:10.1073/pnas.1000488107.
# - Shani, G., D. Heckerman, and R. I Brafman. “An MDP-based Recommender System.” Journal of Machine Learning Research 6, no. 2 (2006): 1265.
# - Joseph A. Konstan, John Riedl. "Deconstructing Recommender Systems." IEEE Spectrum, October 2012.


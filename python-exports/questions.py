# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Exercises for "A Beginner's Introduction to Pydata: How to Build a Minimal Recommendation System"
# ## Systems check

# <codecell>

import numpy as np
import pandas as pd
import tables as tb
!find ./data

# <markdowncell>

# ## How to load (a subset of) the MovieLens dataset

# <codecell>

# subset version (hosted notebook)
movielens_train = pd.read_csv('data/movielens_train.csv', index_col=0)
movielens_test = pd.read_csv('data/movielens_test.csv', index_col=0)
print movielens_train
print movielens_test

# <markdowncell>

# # Numpy Questions: Indexing
# ## 1. Access an individual element in a multidimensional array

# <codecell>

# given the following ndarray, access the its third element
arr = np.arange(10)
arr

# <markdowncell>

# ## 2. Access the last column of of a 2d array

# <codecell>

# given the following ndarray, access its last column
arr = np.array([[5,4,2,5],[4,5,1,12],[0,1,5,4]])
arr

# <markdowncell>

# ## 4. Select all elements from a 2d array that are larger than zero

# <codecell>

# given the following ndarray, obtain all elements that are larger than zero
arr = np.random.randn(5)
arr

# <markdowncell>

# ## 5. Set a portion of an array to the same scalar value

# <codecell>

# given the following ndarray, set the last two elements to 10
arr = np.zeros(6)
arr

# <markdowncell>

# # Numpy Questions: Operations

# <markdowncell>

# ## 1. Compute the sum of a 1D array

# <codecell>

# given the following ndarray, compute its sum
arr = np.random.randn(5)
arr

# <markdowncell>

# ## 2. Compute the mean of a 1D array

# <codecell>

# given the following ndarray, compute its mean
arr = np.random.randn(5)
arr

# <markdowncell>

# ## 3. How do you detect the presence of NANs in an array?

# <codecell>

# given the following ndarray, detect all elements that are nans
arr = np.array([np.nan] * 10)
arr[2:4] = 5
arr

# <markdowncell>

# # Pandas questions: Series and DataFrames
# ## 1. Adding and deleting a column in a dataframe

# <codecell>

# given the following DataFrame, add a new column to it
df = pd.DataFrame({'col1': [1,2,3,4]})
df

# <markdowncell>

# ## 2. Adding and deleting a row in a dataframe

# <codecell>

# given the following DataFrame, delete row 'd' from it
df = pd.DataFrame({'col1': [1,2,3,4]}, index = ['a','b','c','d'])
df

# <markdowncell>

# ## 3. Creating a DataFrame from a few Series

# <codecell>

# given the following three Series, create a DataFrame such that it holds them in its columns
ser_1 = pd.Series(np.random.randn(6))
ser_2 = pd.Series(np.random.randn(6))
ser_3 = pd.Series(np.random.randn(6))

# <markdowncell>

# # Pandas questions: indexing

# <markdowncell>

# ## 1. Indexing into a specific column

# <codecell>

# given the dataframe 'movielens' that we loaded in the previous step, try to index
# into the 'zip' column
movielens[?]

# <markdowncell>

# ## 2. Label-based indexing

# <codecell>

# using the same 'movielens' dataframe, index into the row whose index is 681564
movielens.ix[?]

# <markdowncell>

# # Reco systems questions: estimation functions
# ## 1. Simple content filtering using mean ratings

# <codecell>

# write an 'estimate' function that computes the mean rating of a particular user
def estimate(user_id, movie_id):
    # first, index into all ratings by this user
    # second, compute the mean of those ratings
    # return


# try it out for a user_id, movie_id pair
estimate(4653, 2648)

# <markdowncell>

# ## 2. Simple collaborative filtering using mean ratings

# <codecell>

# write an 'estimate' function that computes the mean rating of a particular user
def estimate(user_id, movie_id):
    # first, index into all ratings of this movie
    # second, compute the mean of those ratings
    # return

    
# try it out for a user_id, movie_id pair
estimate(4653, 2648)

# <markdowncell>

# # Pytables questions: file and node creation
# ## 1. Create a PyTables file in your working environment

# <codecell>

# write your answer in this code block

# <markdowncell>

# ## 2. Within the file you created, create a new group

# <codecell>

# write your answer in this code block

# <markdowncell>

# ## 3. Within the group you created, create a new array of integers and save it

# <codecell>

# write your answer in this code block

# <markdowncell>

# ## 4. For the group created, set a datetime attribute, with the value of ‘utcnow’

# <codecell>

# write your answer in this code block


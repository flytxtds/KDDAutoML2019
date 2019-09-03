import datetime
import CONSTANT
from util import log, timeit
import numpy as np

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
import random
from sklearn.preprocessing import OneHotEncoder
import pandas as pd




@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])


@timeit
def clean_df(df):
    fillna(df)



@timeit
def fillna(df):
    k=0.95
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        threshold = int(k*len(df[c]))
        if (df[c].isnull().sum()) < threshold:
             mean = df[c].mean(skipna = True)
             df[c].fillna(mean,inplace=True)
        else:      
             df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        threshold = int(k*len(df[c]))
        
        if (df[c].isnull().sum()) < threshold :
             mode = df[c].value_counts().index[0]
             df[c].fillna(str(mode),inplace=True)
        else:     
             df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        threshold = int(k*len(df[c]))
        if (df[c].isnull().sum()) < threshold :
             mode = df[c].value_counts().index[0]
             df[c].fillna(str(mode),inplace=True)
        else:     
             df[c].fillna("0", inplace=True)


@timeit
def feature_engineer(df, config, dropcols, numericmap, istrain,square_cubic_transform,skewness):
    
    transform_numeric(df, dropcols, numericmap, istrain,square_cubic_transform,skewness)
    transform_categorical_hash(df, dropcols,istrain)
   
    return 
    
  
                
@timeit
def transform_numeric(df, dropcols, numericmap, istrain,square_cubic_transform,skewness):
    
  
    
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        
        threshold = int(0.02*len(df[c])) 
        if len(set(df[c])) < threshold and istrain:
            dropcols.append(c)
        else:
            if (skewness):
            	df[c] = feature_transform(c, df[c], numericmap, istrain)
    
    if (square_cubic_transform):
    	for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        	df[c+'_squared'] = square(c, df[c])
        	df[c+'_cubed'] = cube(c, df[c])       
    
    return 

@timeit
def square(c, col):
    return col.apply(lambda x: x*x)


@timeit
def cube(c, col):
    return col.apply(lambda x: x*x*x)



   
  
@timeit
def feature_transform(c, col, numericmap, istrain):

    if istrain:
        skewness = col.skew()
        min_ = col.min()
        if skewness>1:
            col = col.apply(lambda x:np.log(x+abs(min_)+1))
            numericmap[c] = 'log'
        if skewness < -1:
            col = col.apply(lambda x:np.power(x+abs(min_),3))
            numericmap[c] = 'cubic'
    else:
        min_ = col.min()
        if c in numericmap:
            col = col.apply(lambda x: np.log(x+abs(min_)+1) if numericmap[c] == 'log' else np.power(x+abs(min_),3))
            
    return col
        


@timeit
def transform_categorical_hash(df, dropcols,istrain):

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
       
        cardinality = len(set(df[c]))
        col_threshold = 0.9*len(df[c]) #90% of dataset length
        if cardinality > col_threshold and istrain:
            dropcols.append(c)
        
   

        df[c] = df[c].astype('category')


    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        #df[c] = df[c].apply(lambda x: int(x.split(',')[0]))
        
        col_threshold = 0.9*len(df[c]) #90% of dataset length
        if cardinality > col_threshold and istrain:
            dropcols.append(c)
        
        df[c]=df[c].astype('category')

     
    if not istrain:
        df.drop(dropcols, axis=1, inplace=True)


    if istrain:
        df.drop(dropcols, axis=1, inplace=True)   

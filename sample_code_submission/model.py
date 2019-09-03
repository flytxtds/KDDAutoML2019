import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install scikit-learn==0.20.3")
os.system("pip3 install catboost")

import copy
import numpy as np
import pandas as pd

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table
from preprocess import clean_df, clean_tables, transform_numeric,transform_categorical_hash
from util import Config, log, show_dataframe, timeit
from model_automl import Model_NIPS
import time
from sklearn.preprocessing import OneHotEncoder



class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.diff_info=None
        self.model = None
        self.time={}
        self.training_data = None
        self.start_time = time.time()

    @timeit
    def fit(self, Xs, y, time_ramain):

        self.tables = copy.deepcopy(Xs)
        
        self.dropcols = []
       
        self.istrain = True
        
        self.numericmap = {}        
        self.square_cubic_transform = True
       
        self.skewness = True
     
        clean_tables(Xs)
        enc = OneHotEncoder(handle_unknown='ignore')

        self.ohe = enc
      
        start = time.time()
        X = merge_table(Xs, self.config)
        self.time['merging_train']= time.time() -start
        clean_df(X)
      
        start = time.time()
        #feature_engineer(X, self.config, self.dropcols, self.numericmap, self.istrain,self.square_cubic_transform,self.skewness)
        transform_numeric(X, self.dropcols, self.numericmap, self.istrain,self.square_cubic_transform,self.skewness)
        transform_categorical_hash(X, self.dropcols,self.istrain)
 
        self.time['feature_engineer']= time.time() -start
      

        numerical_list = list()
        date_time = list()
        categorical=list()
    
        for term,col in enumerate(X.columns):
           if ((X[col].dtype == "int64") or (X[col].dtype=="float64")):
                 numerical_list.append(term)
           if ((X[col].dtype=="datetime64[ns]")):
                 date_time.append(term)
           if ((X[col].dtype.name=="category")):
                 categorical.append(term)      
                
        
        datainfo={}


        
        datainfo['loaded_feat_types'] = list()
        datainfo['loaded_feat_types'].append(date_time)
        datainfo['loaded_feat_types'].append(numerical_list)
        datainfo['loaded_feat_types'].append(categorical)
        datainfo['time_budget'] = self.config['time_budget']

        self.diff_info = datainfo
 
        self.training_data = X
        self.model = Model_NIPS(datainfo)
        start = time.time()
        self.model.fit(X, y,datainfo)

        self.time['fitting']= time.time() -start 
      

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        self.istrain = False
   
        
     
        Xs[MAIN_TABLE_NAME] = X_test

        clean_tables(Xs)
        start = time.time()

        X = merge_table(Xs, self.config)

        self.time['merging_test']= time.time() -start

        clean_df(X)

        
        #feature_engineer(X, self.config, self.dropcols,self.numericmap, self.istrain,self.square_cubic_transform,self.skewness)

        transform_numeric(X, self.dropcols, self.numericmap, self.istrain,self.square_cubic_transform,self.skewness)
        transform_categorical_hash(X, self.dropcols,self.istrain)
        
        start = time.time()
        result =self.model.predict(X,self.diff_info,self.start_time)

        self.time['result_predict']= time.time() -start  

        return pd.Series(result)

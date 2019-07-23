#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:07:36 2019

@author: olashileadebimpe
"""

import os
import csv
import math
import time
import sys
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt
from scipy import stats
from scipy.special import inv_boxcox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,r2_score , mean_absolute_error
import statsmodels.api as sm
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive, drift,  mean,  seasonal_naive)
from tsfresh import extract_features 
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import xgboost as xgb
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings('ignore')

from Modified_timeseries import smape , mase , get_decomposotion , my_holt_trend ,\
 my_holtwinter_seas , split_into_train_test_out_tsfresh , in_sampling_test , \
 my_LinearModel , my_SupportVector,  my_gradientboosting , mlp_bench , my_LinearModel,\
 my_xgboost , get_inv_box_transform, feature_extraction , get_all_stas_result , get_ML_result


start_time = time.time()


data_dir = './Data//M4DataSet2/M4-info.csv'

data_title = []
with open(data_dir) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        if(line[0].startswith('M')):
            data_title.append(line[:2])
        else:
            pass

#load the testing points
data_dir_test = './Data//M4DataSet2/Monthly-test.csv'

data_test = []
with open(data_dir_test) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        line3 = [x.strip('"') for x in line]
        data_test.append(line3)
        
        
#load the testing points
data_dir_train = './Data//M4DataSet2/Monthly-train.csv'

data_train = []
with open(data_dir_train) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        line3 = [x.strip('"') for x in line]
        data_train.append(line3)


#Naive
data_dir_naive = './Data//M4DataSet2/submission-Naive.csv'

data_naive = []
with open(data_dir_naive) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        if(line[0].startswith('M')):
            data_naive.append(line[0:19])
        else:
            pass

#holt
data_dir_holt = './Data//M4DataSet2/submission-Holt.csv'

data_holt = []
with open(data_dir_holt) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        if(line[0].startswith('M')):
            data_holt.append(line[0:19])
        else:
            pass

#SES
data_dir_ETS = './Data//M4DataSet2/submission-ETS.csv'

data_ETS = []
with open(data_dir_ETS) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        line = [x.strip('"') for x in line]
        if(line[0].startswith('M')):
            data_ETS.append(line[0:19])
        else:
            pass


#load the testing points
data_dir_damp = './Data//M4DataSet2/submission-Damped.csv'

data_damp = []
with open(data_dir_damp) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        if(line[0].startswith('M')):
            data_damp.append(line[0:19])
        else:
            pass

#comination
data_dir_comb = './Data//M4DataSet2/submission-Com-1.csv'

data_comb = []
with open(data_dir_comb) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        if(line[0].startswith('M')):
            data_comb.append(line[0:19])
        else:
            pass

#RNN
data_dir_RNN = './Data/M4DataSet2/submission-RNN.csv'

data_RNN = []
with open(data_dir_RNN) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        line = [x.strip('"') for x in line]
        if(line[0].startswith('M')):
            data_RNN.append(line[0:19])
        else:
            pass

#MLP
data_dir_MLP = './Data//M4DataSet2/submission-MLP.csv'

data_MLP = []
with open(data_dir_MLP) as f:
    next(f)
    for line in f:
        line = line.strip()
        line = line.split(",")  
        line = list(filter(None, line))
        line = [x.strip('"') for x in line]
        if(line[0].startswith('M')):
            data_MLP.append(line[0:19])
        else:
            pass
        

      
# =============================================================================
# loop_start = int(sys.argv[1])
# loop_end = int(sys.argv[2])  
# =============================================================================
loop_start = 30800
loop_end = 31000 
freq = 1
monthly = 12
fh = 18
in_size = 40

f = open('./result/'+str(fh)+'_'+str(in_size)+'_'+str(loop_start)+'_'+str(loop_end)+'_'+'M4_monthlydata_v2_200_samples.csv','a')

while loop_start < loop_end:
    
    all_ids = set([data_train[loop_start][0], 
                data_title[loop_start][0],
                data_test[loop_start][0],
                data_naive[loop_start][0],
                data_holt[loop_start][0],
                data_damp[loop_start][0],
                data_comb[loop_start][0],
                data_RNN[loop_start][0],
                data_MLP[loop_start][0]])
    
    if(len(all_ids) == 1):
        train = np.array(data_train[loop_start][1:] , dtype= "float" )
        test = np.array(data_test[loop_start][1:] , dtype= "float" )
        #id for data title
        print(data_title[loop_start])
        
        
        #runn all experient and commit to file your predictions
        
        
        #error metrics for naive 
        naive_ = np.array(data_naive[loop_start][1:] , dtype= "float" )
        smape_naive = smape(test , naive_ )
        mase_naive = mase(train , test , naive_ , freq)
        
        #error metrics for holt 
        holt = np.array(data_holt[loop_start][1:] , dtype= "float" )
        smape_holt = smape(test , holt )
        mase_holt = mase(train , test , holt , freq)
        
        #error metrics for data_ETS 
        ETS = np.array(data_ETS[loop_start][1:] , dtype= "float" )
        smape_ets = smape(test , ETS )
        mase_ets = mase(train , test , ETS , freq)
        
        #error metrics for data_damp
        damp = np.array(data_damp[loop_start][1:] , dtype= "float" )
        smape_damp = smape(test , damp )
        mase_damp = mase(train , test , damp , freq)
        
        #error metrics for data_comb
        comb = np.array(data_comb[loop_start][1:] , dtype= "float" )
        smape_comb = smape(test , comb )
        mase_comb = mase(train , test , comb , freq)
        
        #error metrics for RNN
        RNN = np.array(data_RNN[loop_start][1:] , dtype= "float" )
        smape_RNN = smape(test , RNN )
        mase_RNN = mase(train , test , RNN , freq)
        
        #error metrics for data_comb
        MLP = np.array(data_MLP[loop_start][1:] , dtype= "float" )
        smape_MLP = smape(test , MLP )
        mase_MLP = mase(train , test , MLP , freq)
        
        
        
        
        try:
            
        
            to_transform = True
            
            if to_transform:
                data_log , lambda_ , value = stats.boxcox(train , alpha=0.05)
                
                if(len(set(data_log)) == 1):
                    to_transform =  False
                    data_log =  train
                    
            trend_ , seas_ , resid_ = get_decomposotion(data_log , monthly)
                
            yhat_holt =  my_holt_trend(trend_ , fh)
            
            #seasonality forcast
            yhat_wn_holt = my_holtwinter_seas(seas_ , fh)
            
            x_train, y_train, x_test = split_into_train_test_out_tsfresh(resid_ , in_size)
            #x_train, y_train, x_test = feature_extraction(resid_ , in_size)
            
            #for the insample test to get the best method
            id__ = in_sampling_test(x_train, y_train)
            
            if(id__ == 1):
                pred_ =  my_gradientboosting(x_train, y_train, x_test,in_size, fh)
                
            elif(id__ == 2):
                pred_ =  mlp_bench(x_train, y_train, x_test,in_size, fh)
                
            elif(id__ == 3):
                pred_ =  my_LinearModel(x_train, y_train, x_test,in_size, fh)
            
            elif(id__ == 4):
                pred_ =  my_xgboost(x_train, y_train, x_test,in_size, fh)
                    
            elif(id__ == 5):
                pred_ =  my_SupportVector(x_train, y_train, x_test,in_size, fh)
                
            else:
                print("Retuened a value not in the insample testing")
                
                
            pred__ = np.vstack((yhat_holt, yhat_wn_holt, pred_.reshape(len(pred_))))
                
            if to_transform:
                pred__ = get_inv_box_transform([sum(d) for d in zip(*pred__)] , lambda_)
            else:
                pred__ =  [sum(d) for d in zip(*pred__)]
                
            smape_method = smape(test , pred__ )
            mase_method = mase(train , test , pred__ , freq)
            
            temp, temp_mase = get_all_stas_result(train , test , fh)
            
            temp_ML , temp_ML_mase = get_ML_result(train , test , in_size , fh)
            
            
            output = []     
    
            [output.append(d) for d in data_title[loop_start]]
            output.append("_")
            [output.append(d) for d in test]
            output.append("_")
            output.append("_")
            [output.append(d) for d in pred__]
            output.append("_")
            output.append("_")
            
            output.append((smape_method))
            output.append("_")
            output.append((smape_naive))
            output.append((smape_holt))
            output.append((smape_ets))
            output.append((smape_damp))
            output.append((smape_comb))
            output.append((smape_RNN))
            output.append((smape_MLP))
            output.append("_")
            [output.append(d) for d in temp]
            [output.append(d) for d in temp_ML]
            
            output.append("_")
            output.append("_")
            output.append("_")
            output.append((mase_method))
            output.append("_")
            output.append((mase_naive))
            output.append((mase_holt))
            output.append((mase_ets))
            output.append((mase_damp))
            output.append((mase_comb))
            output.append((mase_RNN))
            output.append((mase_MLP))
            output.append("_")
            [output.append(d) for d in temp_mase]
            [output.append(d) for d in temp_ML_mase]
            
#            plt.figure(figsize=(36,5))
#            plt.plot(np.concatenate((train , np.array(test)))[-fh*3:], label='Actual', linewidth=3)
#            plt.plot(np.concatenate((train , np.array(pred__)))[-fh*3:] , label='forcast' ,marker='o', linestyle='--', linewidth=3)
#            plt.legend(loc='best')
#            plt.savefig("./images/M4_DATASET__ALL__"+str(in_size)+str(data_title[loop_start][0])+".png")
#            plt.close()

                        
            
            
            wr = csv.writer(f, dialect='excel')
            wr.writerow(output)
    
            
        except:
            pass
        
        loop_start+=1
        
        
    else:
        print("Something went wrong in the naming convention")
    
    
f.close()
print("--- %s seconds ---" % (time.time() - start_time))


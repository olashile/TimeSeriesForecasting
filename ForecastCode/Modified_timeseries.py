#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:40:04 2019

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



# =============================================================================
#  all functions used in this project
# =============================================================================


def my_holtwinter_seas(insampledata , fh):
    fit1 = ExponentialSmoothing(insampledata ,seasonal_periods=12 ,trend='add', seasonal='add',damped=True).fit()
    y_hat_avg = fit1.forecast(fh)
    return y_hat_avg


def my_holt_trend(insampledata , fh):
    fit_1 = Holt(insampledata).fit(smoothing_level=1.0, smoothing_slope=1.0, optimized=False)
    fcast1 = fit_1.forecast(fh)
    return fcast1


def get_box_transform(insample_data):
    value = False 
    
    if((np.where(insample_data <= 0)[0]).size == 0):
        transform , lambda_ = stats.boxcox(insample_data)
        
    elif((((np.where(insample_data <= 0)[0]).size)/ len(insample_data)) < 0.05):
        for x in np.where(insample_data <= 0)[0]:
            insample_data[x] = insample_data[x]+0.001
        transform , lambda_ = stats.boxcox(insample_data)
    else:
        value = True
        # add a certain value to all the data and do the box cos transformation
        print("I havn't dont this implemetation")
    return transform , lambda_ , value



def get_inv_box_transform(insample_data , lambda_):
    return inv_boxcox(insample_data, lambda_)

    



def get_decomposotion(insamaple_data , p):
    dec = decompose(insamaple_data , period=p)
    return  dec.trend ,  dec.seasonal ,  dec.resid


def smape(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    
    Values close to zero are the best
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(200 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()



def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
        
    A scaled error is less than one if it arises from a better forecast
    than the average one-step, naïve forecast computed insample. Conversely, 
    it is greater than one if the forecast
    is worse than the average one-step, naïve forecast
    computed in-sample.
    
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep

def get_next_test_data(data , in_num ):
    """
    input data is usually the y_train used for the last prediction
    in_num is the window sixe you want to extract
    returns the test data for the next prediction
    """
    x_train, y_train, x_test = split_into_train_test_out_tsfresh(data, in_num)
    
    return x_test

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]   
    
	dff = pd.DataFrame(data)
    
    
	cols, names = list(), list()

	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	agg = pd.concat(cols, axis=1)
	agg.columns = names

	if dropnan:
		agg.dropna(inplace=True)
	return agg

def feature_extraction(data, in_num):
    """
    Get the time series to be used for feature extraction
    y_train is the y value of the data fitting data
    
    """
    data1 = list(data.copy())
    temp= list(data)
    
    values = series_to_supervised(temp , in_num-1)
    values = values.loc[:, 'var1(t-18)':'var1(t)']
    
    
    data1.append(data1[-1])
    #data1 = np.roll(data, -1) # roll the data once just to have a data in the y_train axis
    
    #make the dataframe using Tsfresh package
    df_shift_small , y_train = make_forecasting_frame(data1, kind="price", max_timeshift=in_num, rolling_direction=1)
    
    #create the features needed for the
    result = extract_features(df_shift_small, column_id="id", column_sort="time", 
                              column_value="value", impute_function=impute,
                              show_warnings=False, disable_progressbar=False, 
                              n_jobs=1,chunksize = 1,
                              default_fc_parameters=EfficientFCParameters())
    
    #result_without_zero = result.loc[:, (result != 0).any(axis=0)]
    #the 50 columns i only need out tsfresh
    columl_list = [  
            'value__absolute_sum_of_changes',
                     'value__agg_autocorrelation__f_agg_"mean"',
                     'value__agg_autocorrelation__f_agg_"median"',
                     'value__agg_autocorrelation__f_agg_"var"',
                     'value__autocorrelation__lag_0',
                     'value__autocorrelation__lag_1',
                     'value__autocorrelation__lag_2',
                     'value__binned_entropy__max_bins_10',
                     'value__cid_ce__normalize_False',
                     'value__cid_ce__normalize_True',
                     'value__count_above_mean',
                     'value__count_below_mean',
                     'value__fft_aggregated__aggtype_"centroid"',
                     'value__fft_aggregated__aggtype_"variance"',
                     'value__fft_coefficient__coeff_0__attr_"abs"',
                     'value__fft_coefficient__coeff_0__attr_"real"',
                     'value__fft_coefficient__coeff_1__attr_"abs"',
                     'value__fft_coefficient__coeff_1__attr_"angle"',
                     'value__fft_coefficient__coeff_1__attr_"imag"',
                     'value__fft_coefficient__coeff_1__attr_"real"',
                     'value__first_location_of_maximum',
# =============================================================================
#                      'value__large_standard_deviation__r_0.05',
#                      'value__large_standard_deviation__r_0.1',
#                      'value__large_standard_deviation__r_0.15000000000000002',
#                      'value__large_standard_deviation__r_0.2',
#                      'value__large_standard_deviation__r_0.25',
#                      'value__large_standard_deviation__r_0.30000000000000004',
#                      'value__large_standard_deviation__r_0.35000000000000003',
#                      'value__large_standard_deviation__r_0.4',
#                      'value__large_standard_deviation__r_0.45',
# =============================================================================
                     'value__linear_trend__attr_"intercept"',
                     'value__linear_trend__attr_"pvalue"',
                     'value__linear_trend__attr_"rvalue"',
                     'value__linear_trend__attr_"slope"',
                     'value__longest_strike_above_mean',
                     'value__longest_strike_below_mean',
                     'value__max_langevin_fixed_point__m_3__r_30',
                     'value__maximum',
                     'value__mean',
                     'value__mean_abs_change',
                     'value__mean_change',
                     'value__median',
                     'value__minimum',
                     'value__number_cwt_peaks__n_5',
                     'value__partial_autocorrelation__lag_0',
                     'value__partial_autocorrelation__lag_1',
                     'value__partial_autocorrelation__lag_2',
                     'value__standard_deviation',
                     'value__sum_values',
                     'value__variance']
    
    #extract just only those colums
    result_without_zero = result[columl_list]
    
    result_without_zero_ = result_without_zero.iloc[-len(values):]
    
    result_combined = pd.concat([values.reset_index().drop(columns='index'), 
                        result_without_zero_.reset_index().drop(columns='id')], axis=1)
    
    x_train =  result_combined[:-1]
    x_test =   result_combined[-1:]
    y_train =  y_train[in_num-1:-1]
     
    return x_train, y_train, x_test


def split_into_train_test_out_tsfresh(data, in_num):
    """
    Get the time series to be used for feature extraction
    y_train is the y value of the data fitting data
    
    """
    
    data1 = np.roll(data, -1) # roll the data once
    
    #make the dataframe using Tsfresh package
    df_shift_small , y_train = make_forecasting_frame(data1, kind="price", max_timeshift=in_num, rolling_direction=1)
    
    #create the features needed for the
    result = extract_features(df_shift_small, column_id="id", column_sort="time", 
                              column_value="value", impute_function=impute,
                              show_warnings=False, disable_progressbar=False, 
                              n_jobs=5,chunksize = 1,
                              default_fc_parameters=EfficientFCParameters())
    
    #result_without_zero = result.loc[:, (result != 0).any(axis=0)]
    #the 50 columns i only need out tsfresh
    columl_list = [  
#            'value__absolute_sum_of_changes',
# =============================================================================
# =============================================================================
                       'value__agg_autocorrelation__f_agg_"mean"',
                       'value__agg_autocorrelation__f_agg_"median"',
                       'value__agg_autocorrelation__f_agg_"var"',
                       'value__autocorrelation__lag_0',
                       'value__autocorrelation__lag_1',
                       'value__autocorrelation__lag_2',
                       'value__binned_entropy__max_bins_10',
# =============================================================================
# =============================================================================
#                     'value__cid_ce__normalize_False',
#                     'value__cid_ce__normalize_True',
#                     'value__count_above_mean',
#                     'value__count_below_mean',
#                     'value__fft_aggregated__aggtype_"centroid"',
                     'value__fft_aggregated__aggtype_"variance"',
                     'value__fft_coefficient__coeff_0__attr_"abs"',
                     'value__fft_coefficient__coeff_0__attr_"real"',
                     'value__fft_coefficient__coeff_1__attr_"abs"',
                     'value__fft_coefficient__coeff_1__attr_"angle"',
                     'value__fft_coefficient__coeff_1__attr_"imag"',
                     'value__fft_coefficient__coeff_1__attr_"real"',
                     'value__first_location_of_maximum',
 #=============================================================================
# =============================================================================
                       'value__large_standard_deviation__r_0.05',
                       'value__large_standard_deviation__r_0.1',
                       'value__large_standard_deviation__r_0.15000000000000002',
                       'value__large_standard_deviation__r_0.2',
                       'value__large_standard_deviation__r_0.25',
#                       'value__large_standard_deviation__r_0.30000000000000004',
#                       'value__large_standard_deviation__r_0.35000000000000003',
#                       'value__large_standard_deviation__r_0.4',
#                       'value__large_standard_deviation__r_0.45',
# =============================================================================
# =============================================================================
                     'value__linear_trend__attr_"intercept"',
                     'value__linear_trend__attr_"pvalue"',
                     'value__linear_trend__attr_"rvalue"',
                     'value__linear_trend__attr_"slope"',
                     'value__longest_strike_above_mean',
                     'value__longest_strike_below_mean',
                     'value__max_langevin_fixed_point__m_3__r_30',
                     'value__maximum',
                     'value__mean',
                     'value__mean_abs_change',
                     'value__mean_change',
                     'value__median',
                     'value__minimum',
                     'value__number_cwt_peaks__n_5',
                     'value__partial_autocorrelation__lag_0',
                     'value__partial_autocorrelation__lag_1',
                     'value__partial_autocorrelation__lag_2',
                     'value__standard_deviation',
                     'value__sum_values',
                     'value__variance']
    #extract just only those colums
    result_without_zero = result[columl_list]
    
    #return these values
    x_train =  result_without_zero[:-1]
    x_test =   result_without_zero[-1:]
    y_train =  y_train[:-1]
    
    return x_train, y_train, x_test


def mlp_bench(x_train, y_train, x_test,in_num, fh):
    """
    Forecasts using a simple MLP which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []

    model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)
    model.fit(x_train, y_train)

    last_prediction = model.predict(x_test)
    
    for i in range(0, fh):
        y_hat_test.append(last_prediction[0])
        temp = np.concatenate((y_train , last_prediction))
        x_test = get_next_test_data(temp , in_num )
        last_prediction = model.predict(x_test)

    return np.asarray(y_hat_test)



def my_gradientboosting(x_train, y_train, x_test, in_num ,fh):
    """
    Forecasts using a simple Ensembles which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []
    
    model_em = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.5, loss='ls', max_depth=5, max_features=4,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=0.1,
             min_samples_split=0.5, min_weight_fraction_leaf=0.0,
             n_estimators=500, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
    
    model_em.fit(x_train, y_train)
    
    last_prediction = model_em.predict(x_test)
    
    for i in range(0, fh):
        y_hat_test.append(last_prediction[0])
        temp = np.concatenate((y_train , last_prediction))
        x_test = get_next_test_data(temp , in_num )
        last_prediction = model_em.predict(x_test)

    return np.asarray(y_hat_test)




def my_xgboost(x_train, y_train, x_test, in_num, fh):
    """
    Forecasts using a simple Ensembles which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []
    
    model_em = model_em = xgb.XGBRegressor(learning_rate=0.5,  max_depth=5, n_estimators=500)
    
    model_em.fit(x_train, y_train)
    
    last_prediction = model_em.predict(x_test)
    
    for i in range(0, fh):
        y_hat_test.append(last_prediction[0])
        temp = np.concatenate((y_train , last_prediction))
        x_test = get_next_test_data(temp , in_num )
        last_prediction = model_em.predict(x_test)

    return np.asarray(y_hat_test)



def my_SupportVector(x_train, y_train, x_test,in_num , fh):
    """
    Forecasts using a simple Ensembles which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []
    
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.002)
    svr_rbf.fit(x_train, y_train)
    
    last_prediction = svr_rbf.predict(x_test)
    
    for i in range(0, fh):
        y_hat_test.append(last_prediction[0])
        temp = np.concatenate((y_train , last_prediction))
        x_test = get_next_test_data(temp , in_num )
        last_prediction = svr_rbf.predict(x_test)

    return np.asarray(y_hat_test)

# =============================================================================
# 
# =============================================================================
#using pmdarima autoarima
import pmdarima as pm
def my_autorima(x_train, fh):
    arima = pm.auto_arima(x_train, error_action='ignore', trace=1,seasonal=True, m=12)
    y_hat_autoarima = arima.predict(n_periods= fh)
    
    return y_hat_autoarima


def my_basic_arima(x_train, fh):
    arima = sm.tsa.statespace.SARIMAX(x_train, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
    y_hat_autoarima = arima.predict(start=len(x_train), end=len(x_train)+fh-1)
    
    return y_hat_autoarima


def get_all_stas_result(series__ , y_true , fh):
    res = []
    res_mase = []
    #Arima
    y_arima =  my_autorima(series__ , fh)
    #y_arima =  y_true
    res.append((smape(y_true, y_arima)))
    res_mase.append((mase(series__ , y_true, y_arima , 1)))
    
    #holts
    y_holts = my_holt_trend(series__ , fh)
    res.append((smape(y_true, y_holts)))
    res_mase.append((mase(series__ , y_true, y_holts , 1)))
    
    #winter
    y_winter = my_holtwinter_seas(series__ , fh)
    res.append((smape(y_true, y_winter)))
    res_mase.append((mase(series__ , y_true, y_winter , 1)))
    
    #average
    temp = np.vstack((y_arima, y_holts,y_winter ))
    
    y_average =  [mean(d) for d in zip(*temp)]
    res.append((smape(y_true, y_average)))
    res_mase.append((mase(series__ , y_true, y_average , 1)))
    
    return res , res_mase



def split_into_train_test(data, in_num ,fh):
    """
    Splits the series into train and test sets. Each step takes multiple points as inputs
    :param data: an individual TS
    :param fh: number of out of sample points
    :param in_num: number of input points for the forecast
    :return:
    """
    train =  data
    x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
    x_test = train[-in_num:]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    x_train = np.reshape(x_train, (-1, 1))
    x_test = np.reshape(x_test, (-1, 1))
    temp_test = np.roll(x_test, -1)
    temp_train = np.roll(x_train, -1)
    for x in range(1, in_num):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
        temp_test = np.roll(temp_test, -1)[:-1]
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train, x_test 


def my_svr_test(x_train, y_train, x_test, fh):
    """
    Forecasts using a simple MLP which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []
    
    model_em = SVR(kernel='rbf', C=1e3, gamma=0.002)
    
    model_em.fit(x_train, y_train)
    
    last_prediction = model_em.predict(x_test)[0]
    
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model_em.predict(x_test)[0]
    
    return np.asarray(y_hat_test)


def my_gradientboosting_test(x_train, y_train, x_test, fh):
    """
    Forecasts using a simple MLP which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []
    
    model_em = ensemble.GradientBoostingRegressor()
    
    model_em.fit(x_train, y_train)
    
    last_prediction = model_em.predict(x_test)[0]
    
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model_em.predict(x_test)[0]
    
    return np.asarray(y_hat_test)



def get_ML_result(data , test , insize , fh):
    
    res = []
    res_mase = []
    #get the data into test and train components
    x_train, y_train, x_test = split_into_train_test(data, insize, fh)
    
    yhat_gbm = my_gradientboosting_test(x_train, y_train, x_test, fh)
    
    yhat_svr = my_svr_test(x_train, y_train, x_test, fh)
    
    res.append(smape(test ,yhat_gbm))
    
    res.append(smape(test ,yhat_svr))
    
    res_mase.append(mase(data , test ,yhat_gbm , 1))
    res_mase.append(mase(data , test ,yhat_svr , 1))
    
    return res , res_mase
    

# =============================================================================
# 
# =============================================================================

def my_LinearModel(x_train, y_train, x_test,in_num, fh):
    """
    Forecasts using a simple Ensembles which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []
    
    ridgeReg = Ridge(alpha=0.05, normalize=True)
    ridgeReg.fit(x_train, y_train)
    
    last_prediction = ridgeReg.predict(x_test)
    
    for i in range(0, fh):
        y_hat_test.append(last_prediction[0])
        temp = np.concatenate((y_train , last_prediction))
        x_test = get_next_test_data(temp , in_num )
        last_prediction = ridgeReg.predict(x_test)

    return np.asarray(y_hat_test)

def in_sampling_test(x_train, y_train , n= 10):
    
    pr_ = pd.DataFrame(columns=['MAE'])
    
    #in_sample_size = [12,18,24,36,40]
    
    #extract the last 5 records of the x train and y train.
    in_sample_test_x = x_train.tail(n)
    in_sample_test_y = y_train.tail(n)
    
    #build several models for the extrain and the y train
    
    #gradient boosting
    model_em = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.5, loss='ls', max_depth=5, max_features=5,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=0.1,
             min_samples_split=0.5, min_weight_fraction_leaf=0.0,
             n_estimators=100, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
    
    model_em.fit(x_train, y_train)
    prediction = model_em.predict(in_sample_test_x)   
    
    #print(mean_absolute_error(in_sample_test_y, prediction))
    pr_.loc[1] = smape(in_sample_test_y.values, prediction)
    
    
    #using MLP
    model_mlp = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)
    model_mlp.fit(x_train, y_train)
    prediction = model_mlp.predict(in_sample_test_x)  
    #print(mean_absolute_error(in_sample_test_y, prediction))
    pr_.loc[2] = smape(in_sample_test_y.values, prediction)
    
    #using ridge regression
    ridgeReg = Ridge(alpha=0.05, normalize=True)
    ridgeReg.fit(x_train, y_train)
    
    prediction = ridgeReg.predict(in_sample_test_x)
    #print(mean_absolute_error(in_sample_test_y, prediction))
    pr_.loc[3] = smape(in_sample_test_y.values, prediction)
    
    #using xgb regression
    model_xgb = xgb.XGBRegressor(n_estimators=100)
    model_xgb.fit(x_train, y_train)
    
    prediction = model_xgb.predict(in_sample_test_x)
    #print(mean_absolute_error(in_sample_test_y, prediction))
    pr_.loc[4] = smape(in_sample_test_y.values, prediction)
    
    #using Support vector machine
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.002)
    svr_rbf.fit(x_train, y_train)
    
    prediction = svr_rbf.predict(in_sample_test_x)  
    #print(mean_absolute_error(in_sample_test_y, prediction))
    pr_.loc[5] = smape(in_sample_test_y.values, prediction)
    
    
    return pr_['MAE'].idxmin()



if __name__ == '__main__':
    
    # Data directory for the M3 Monthly dataset
    #data_dir2 = './Data/M3C.csv'
    
    # Data directory for the M3 Monthly dataset
    #data_dir2 = './Data/NN3_reduced.csv'
    data_dir2 = './Data/NN3_FINAL_DATASET_WITH_TEST_DATA.csv'
    
    
    
    
    #database decleation
    m3data = []
    m3data_ = []
    
    #loading all the excel file
    with open(data_dir2) as f:
        #next(f)
        for line in f:
            line = line.strip()
            line = line.split(",")  
            line = list(filter(None, line)) 
            line3 = [x.strip('"') for x in line]
            #line3 = list(map(int, line3))
            m3data.append(line3)
            
    #extracting only the time series part of the data set       
    for i in range(len(m3data)):
        m3data_.append(np.array(m3data[i][6:] , dtype= "float" ))
        
        
    # =============================================================================
    # fh = int(sys.argv[1])
    in_size = int(sys.argv[1])
    # 
    # f = open(str(fh)+str(in_size)+'output.csv','a')
    # =============================================================================
    
    fh = 18     # forecasting horizon
    #in_size = 40    # number of points used as input for each forecast
    
    freq = 1       # data frequency
    monthly = 12
    series =  m3data[0]
    f = open('./result/'+str(fh)+'_'+str(in_size)+'_'+'NN3_FINAL_WITH_TEST_DATA.csv','a')
    start_time = time.time()
    counter = 0
    for series in m3data:
        try:
            
            output = []
            
            id = np.array(series[:1])[0]
            series_ = np.array(series[6:] , dtype= "float" )
            
            f_train = series_[:-fh]
            f_test = series_[-fh:]
            
            to_transform = True
        
            if to_transform:
                data_log , lambda_ , value = stats.boxcox(f_train , alpha=0.05)
                
                if(len(set(data_log)) == 1):
                    to_transform =  False
                    data_log =  f_train
                    
            trend_ , seas_ , resid_ = get_decomposotion(data_log , monthly)
            
            yhat_holt =  my_holt_trend(trend_ , fh)
            
            #seasonality forcast
            yhat_wn_holt = my_holtwinter_seas(seas_ , fh)
            
            x_train, y_train, x_test = split_into_train_test_out_tsfresh(resid_ , in_size)
            
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
                
            temp, temp_mase = get_all_stas_result(f_train , f_test , fh)
            
            temp_ML , temp_ML_mase = get_ML_result(f_train , f_test , in_size , fh)
                
            print(id)
            print(id__)    
            print(smape(f_test, pred__))
                
            print(mase(f_train ,f_test, pred__ , freq))
            
            output.append(id)
            [output.append(d) for d in f_test]
            output.append("_")
            output.append("_")
            [output.append(d) for d in pred__]
            output.append("_")
            output.append("_")
            
            output.append((smape(f_test, pred__)))
        
            output.append("_")
            output.append((mase(f_train ,f_test, pred__ , freq)))
            output.append("#########")
            output.append("_")
            [output.append(d) for d in temp]
            [output.append(d) for d in temp_ML]
            output.append("_")
            [output.append(d) for d in temp_mase]
            [output.append(d) for d in temp_ML_mase]
            
            plt.figure(figsize=(36,5))
            plt.plot(series_[-fh*3:], label='Actual', linewidth=3)
            plt.plot(np.concatenate((f_train , np.array(pred__)))[-fh*3:] , label='forcast' ,marker='o', linestyle='--', linewidth=3)
            plt.legend(loc='best')
            plt.savefig("./images/NN3__ALL__"+str(in_size)+id+".png")
            plt.close()
                          
   
            
            
            
            wr = csv.writer(f, dialect='excel')
            wr.writerow(output)
        except:
            pass
        

        
    f.close()
    print("--- %s seconds ---" % (time.time() - start_time))

from abc import ABC
from abc import abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


class TimeSeriesBaseAlgorithm(ABC):
    """Base class for all time series algorithms

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """
    
    def data_split(self, data, split_proportion):
        """
        Splits the data into train and test datasets

        Parameters
        ----------
        data : DataFrame
            A dataframe to split into 2 parts, where the dataframe has an index is a 
            time based data type e.g datetime or period(M) and only has one feature 
            which is the target feature

        split_proportion : float
            The proportation to use for the train dataset, must be between 0 and 1

        Returns
        -------
        train : DataFrame
            A dataframe which can be used for training the model

        test : DataFrame
            A dataframe which can be used for testing the model
        """
        
        # Creating the train and validation set
        train = data[:int(split_proportion*(len(data)))]
        test = data[int(split_proportion*(len(data))):]

        return train, test
        
    
    def dickey_fuller_test(self, timeseries):
        """
        Runs the dickey-fuller test to determine whether the data is stationary or not. 
        Only if the p-value is less than 0.05 can you reject the null hypothesis. If it
        is more then 0.05 then the data is considered to be stationary. 

        Parameters
        ----------
        timeseries : DataFrame
            A dataframe where the index is a time based data type 
            e.g datetime or period(M) and only has one feature which is the target feature

        Returns
        -------
        df_res_output : Series
            A table of the results of the Test Statistic, p-value, 
            Critical Value(1%, 5% and 10%) as well as the number 
            of observations and lags used
        """
        
        timeseries = timeseries.iloc[:,0].values    

        #Perform Dickey-Fuller test:
        
        df_test_res = adfuller(timeseries, autolag='AIC')
        df_res_output = pd.Series(df_test_res[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in df_test_res[4].items():
            df_res_output['Critical Value (%s)'%key] = value

        return df_res_output


    def display_rolling_plot(self, timeseries, rolling_period):
        """
        Generates and display a line plot with 3 line, the original data points, the rolling mean and the rolling standard deviation. 

        Parameters
        ----------
        timeseries : DataFrame
            A dataframe where the index is a time based data type 
            e.g datetime or period(M) and only has one feature which is the target feature
        
        rolling_period : int
            The number of data points to use for the rolling period
        
        Returns
        -------
        fig : plot
            the line plot for the rolling mean and standard deviation

        """
        print("Displaying rolling plot...")
        timeseries = timeseries.iloc[:,0].values             # only selects the value data points
        fig, ax = plt.subplots(figsize=(20, 10))

        #Determing rolling statistics
        rolmean = pd.Series(timeseries).rolling(rolling_period).mean()
        rolstd = pd.Series(timeseries).rolling(rolling_period).std()

        #Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

        return fig

    
    def mean_absolute_percentage_error(self, y_true, y_pred):
        """
        Returns the mean absolute percentage error (MAPE) between the actual value/s and 
        the predicted value/s. MAPE is the average percentage error between the actual
        and predicted value/s. 

        Parameters
        ----------
        y_true : list or array  
            A list or array of actual value/s for the target variable
        
        y_pred : list or array  
            A list or array of predicted value/s for future observations 
            of the target variable

        Returns
        -------
        mape : float
            The mean of the absolute percentage error between the actual and predicted value/s to 2 decimal places
        """
        
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
    
        mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

        return mape


    @abstractmethod   
    def grid_search_run(self):
        pass


    @abstractmethod   
    def model_build(self):
        pass


    @abstractmethod   
    def forecast_vs_actual_plot(self):
        pass


    @abstractmethod   
    def cross_validate(self):
        pass


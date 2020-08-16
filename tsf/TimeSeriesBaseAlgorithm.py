

# Libraries
from abc import ABC
from abc import abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


class TimeSeriesBaseAlgorithm(ABC):
    """
    Base class for all time series algorithms. Every algorithm
    should inherit from this class and those become a subclass of 
    TimeSeriesBaseAlgorithm. The subclass needs to implement the
    abstract methods. Where required the subclass can overwrite 
    any of the concrete methods here.

    Notes
    -----
    
    """
    
    @abstractmethod   
    def cross_validate(self):
        pass
    
    
    @abstractmethod   
    def grid_search_run(self):
        pass


    @abstractmethod   
    def model_build(self):
        pass


    @abstractmethod   
    def plot_forecasts(self):
        pass


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
        
        # Creating the train and validation set by splitting the data
        train = data[:int(split_proportion*(len(data)))]
        test = data[int(split_proportion*(len(data))):]

        return train, test
        
    
    def dickey_fuller_test(self, time_series):
        """
        Runs the dickey-fuller test to determine whether the data is stationary or not. 
        Only if the p-value is less than 0.05 can you reject the null hypothesis. If it
        is more then 0.05 then the data is considered to be stationary. 

        Parameters
        ----------
        time_series : DataFrame
            A dataframe where the index is a time based data type 
            e.g datetime or period(M) and only has one feature which is the target feature

        Returns
        -------
        df_res_output : Series
            A table of the results of the Test Statistic, p-value, 
            Critical Value(1%, 5% and 10%) as well as the number 
            of observations and lags used
        """
        
        time_series = time_series.iloc[:,0].values    

        # Performs Dickey-Fuller test
        df_test_res = adfuller(time_series, autolag='AIC')

        # Creates Series and specifies the index
        df_res_output = pd.Series(df_test_res[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

        # For each critical value is given the 'Critical Value' title plus the relevant percentage
        for key,value in df_test_res[4].items():
            df_res_output['Critical Value (%s)'%key] = value

        return df_res_output


    def display_rolling_plot(self, time_series, rolling_period):
        """
        Generates and display a line plot with 3 line, the original data points, the rolling mean and the rolling standard deviation. 

        Parameters
        ----------
        time_series : DataFrame
            A dataframe where the index is a time based data type 
            e.g datetime or period(M) and only has one feature which is the target feature
        
        rolling_period : int
            The number of data points to use for the rolling period
        
        Returns
        -------
        fig : plot
            The line plot for the rolling mean and standard deviation

        """

        print("Displaying rolling plot...")
        # Selects only the value data points and creates an empty plot
        time_series = time_series.iloc[:,0].values             
        fig, ax = plt.subplots(figsize=(20, 10))

        # Calculates the rolling statistics for both the mean and standard deviation
        rolmean = pd.Series(time_series).rolling(rolling_period).mean()
        rolstd = pd.Series(time_series).rolling(rolling_period).std()

        # Plots rolling statistics
        orig = plt.plot(time_series, color='blue',label='Original')
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
        
        # Returns a contiguous flattened array
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
    
        # Calculates the MAPE
        mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

        return mape


    


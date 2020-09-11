
# Libraries
from tsf.TimeSeriesBaseAlgorithm import TimeSeriesBaseAlgorithm

import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt

import statsmodels as sm
import statsmodels.tsa.api as tsa
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose


class SARIMA_tsf(TimeSeriesBaseAlgorithm):
    """
    This class uses the SARIMA algorithm for time series forecasting and
    it contains methods which allow you to check for stationarity, create 
    models, perform a grid search on models, cross validate models, creates
    plot for forecasts and more. 
    
    Notes
    -----
    This class inherits from TimeSeriesBaseAlgorithm base class and it 
    provides implementation for the base class's abstract methods. This
    subclass also has specific methods for SARIMA time series algorithm.

    """
    
    def __init__(self):
        pass


    def cross_validate(self, time_series, order_param, seasonal_order_param, 
                        freq_type_start, freq_type_end):
        """
        Runs cross validation for a specified duration (validation_period) 
        and predicts values for a specified number steps (no_forecasts). Cross 
        validation for time series, trains a model for every observation from 
        the validation period using observations prior to each observation in 
        the validation period as training data. It then makes predictions equal
        to the number of no_forecast and compares the predictions to the actual 
        values using MAPE on the results.

        Parameters
        ----------
        time_series : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature.
            
        order_param : tuple
            A tuple of 3 values representing p, d and q. This represents the
            number of AR parameters (p), differences (d), and MA (q) parameters.

        seasonal_order_param : tuple
            A tuple of 4 values representing P, D, Q and M. The 4 values are
            specifically for the seasonality conponent of the time series.
            Number of AR parameters (P), differences (D), MA (Q) parameter and
            single seasonality period (M).
 
        freq_type_start : dict 
            A dictionary where the key is the frequency of observations and 
            value is validation_period - the number of cross validation iterations 
            to use for testing the model's stabilty overtime which should be a 
            negative value. e.g. {'months' : -12} 

        freq_type_end : dict 
            A dictionary where the key is the frequency of observations and 
            value is no_forecasts - the number of observations to forecast for
            e.g. {'months' : 6} 

        Returns
        -------
        cv_mape_res : DataFrame
            A dataframe containing the results of the cross validation. 
            There are 4 columns 'order_param', 'seasonal_order_param', 
            'mape_total', 'mape_freq'. The word 'freq' is replaced with the actual
            frequency of the data.

        mape_df : DataFrame
            A dataframe containing the results of the cross validation. 
            There are 6 columns 'order_param', 'seasonal_order_param', cv_datestart', 
            'cv_dateend', 'mape_total', 'mape_freq'. The word 'freq' is replaced 
            with the actual frequency of the data.

        """

        # Extracts the validation_period, no_forecasts and frequency from the input 
        # dictionaries
        validation_period = abs(list(freq_type_start.values())[0])
        no_forecasts = list(freq_type_end.values())[0]
        freq = list(freq_type_start.keys())[0]
        
        # Sorts the input time series
        time_series = time_series.sort_index(ascending=True)

        # Gets the newest observations and creates a list
        mape = []
        time_series_temp = time_series.tail(validation_period)
        datelist = time_series_temp.index.tolist()

        # Performs the cross validation for each date in the validation period
        for date in datelist:
            # Sets the start and end date for the test data
            cv_date = pd.to_datetime(date)
            cv_datestart = cv_date + relativedelta(**freq_type_start)
            cv_dateend = cv_datestart + relativedelta(**freq_type_end)

            # Everything prior to the start date is the training data
            cv_df= time_series.loc[time_series.index < cv_datestart]
            # Everything between the start and end date is the test data
            actual = time_series.loc[(time_series.index >= cv_datestart) & (time_series.index < cv_dateend)]

            # Builds the model and makes a forecast equal to the no_forecasts
            model_cv = self.model_build(cv_df, order_param, seasonal_order_param)
            pred_forecast = model_cv.get_forecast(steps=no_forecasts)

            # Gets the predicted values and actual values
            pred = pred_forecast.predicted_mean
            actual = actual[actual.columns[0]]

            # Calculates the MAPE total (sum of the actuals vs sum of predictions) 
            # and MAPE freq (the MAPE for individual observations)
            mape_total = super().mean_absolute_percentage_error(sum(actual),sum(pred))
            mape_freq = super().mean_absolute_percentage_error(actual,pred)

            # Adds the values to a list
            mape_res = [order_param, seasonal_order_param, cv_datestart, cv_dateend, mape_total, mape_freq]
            mape.append(mape_res)

        # Creates an empty dataframe
        mape_df = pd.DataFrame(mape, columns=['order_param', 'seasonal_order_param', 'cv_datestart', \
                'cv_dateend', 'mape_total', 'mape_'+str(freq)])

        # Creates an empty dataframe
        cv_mape_res = pd.DataFrame(columns=['order_param', \
                                'seasonal_order_param', 'mape_total', 'mape_'+str(freq)])
        
        # Calculates the MAPE total and MAPE freq for the entire validation period
        lst_tot_mape = str(round(np.mean([item[4] for item in mape]),2))
        lst_freq_mape = str(round(np.mean([item[5] for item in mape]),2))

        # Creates a series and adds the data as a row to the 'cv_mape_res' dataframe
        mape_row = pd.Series(data=[order_param, seasonal_order_param, lst_tot_mape, lst_freq_mape], \
            index=['order_param', 'seasonal_order_param', 'mape_total', 'mape_'+str(freq)])

        cv_mape_res = cv_mape_res.append(mape_row, ignore_index=True)

        return cv_mape_res, mape_df

    
    def decompose(self, time_series, model_type):
        """
        Runs decompose on the time series to extract the trend component,
        seasonal component and the residual component. 
        
        Parameters
        ----------
        time_series : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature.
            
        model_type : str
            There are 2 types: 'additive' is useful when the seasonal variation
            is relatively constant over time 'multiplicative' is useful when 
            the seasonal variation increases over time. 

        Returns
        -------
        decomposition : DecomposeResult
            Results class for seasonal decompositions which contains trend 
            component,cseasonal component and the residual component. 

        fig : plot
            A plot which shows the original, trend, seasonal and residual
            components of the time series.

        """
        # Creates an empty plot and performs seasonal decomposion of the 
        # input time series
        fig, ax = plt.subplots(figsize=(20, 10))
        decomposition = seasonal_decompose(time_series, model = model_type)

        # Extracts the trend, seasonal, residual components of the
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Creates individual subplots and adds them to the big plot
        plt.subplot(411)
        plt.plot(time_series, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal,label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        return decomposition, fig


    def grid_search_CV(self, time_series, order_range, seasonal_order_list,
                                freq_type_start, freq_type_end):
        """
        Generates a grid search of parameters and then builds a 
        model with every combination of parameters. It then saves
        the AIC score and parameters into a list.

        Parameters
        ----------
        time_series : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature.

        order_range : range
            A range of values to assign for p, d and q. This represents the
            number of AR parameters (p), differences (d), and MA (q) parameters.

        seasonal_order_list : list
            A list of values to assign for M. This represents a single 
            seasonality period (M). Note it will use order_range for the
            values for the number of AR parameters (P), differences (D)
            and MA (Q) parameter.    

        freq_type_start : dict 
            A dictionary where the key is the frequency of observations and 
            value is validation_period - the number of cross validation iterations 
            to use for testing the model's stabilty overtime. e.g. {'months' : -12} 

        freq_type_end : dict 
            A dictionary where the key is the frequency of observations and 
            value is no_forecasts - the number of observations to forecast for
            e.g. {'months' : 6}
            
        Returns
        -------
        grid_cv_df : DataFrame
            A dataframe containing the results of the cross validation. 
            There are 4 columns 'order_param', 'seasonal_order_param', 
            'mape_total', 'mape_freq'. The word 'freq' is replaced with the actual
            frequency of the data.
            
        """   

        # Creates a combination of all the different values for order param
        # and seasonal order param (the grid)
        p = d = q = order_range
        M = seasonal_order_list
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, M))]
        
        # Gets the frequency and creates an empty dataframe
        freq = list(freq_type_start.keys())[0]
        grid_cv_df = pd.DataFrame(columns=['order_param', 'seasonal_order_param',\
                                'mape_total', 'mape_'+str(freq)])

        # Using the grid, it builds a model for each unique combination 
        # to search for the model which the best performance using AIC metric
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                cv_res, mape_df = self.cross_validate(time_series, param, param_seasonal, freq_type_start, freq_type_end)
                grid_cv_df = grid_cv_df.append(cv_res, ignore_index=True)

        return grid_cv_df


    def grid_search_run(self, time_series_train, order_range, seasonal_order_list):
        """
        Generates a grid search of parameters and then builds a 
        model with every combination of parameters. It then saves
        the AIC score and parameters into a list.

        Parameters
        ----------
        time_series_train : DataFrame
            A dataframe to be used for training the model where index is 
            a time based data type e.g datetime or period(M) and only has
            one feature which is the target feature.
            
        order_range : range
            A range of values to assign for p, d and q. This represents the
            number of AR parameters (p), differences (d), and MA (q) parameters.

        seasonal_order_list : list
            A list of values to assign for M. This represents a single 
            seasonality period (M). Note it will use order_range for the
            values for the number of AR parameters (P), differences (D)
            and MA (Q) parameter.

        Returns
        -------
        aic_scores : list
            A list of the order parameters, seasonal parameters and the respective AIC score
            
        """   

        # Creates a combination of all the different values for order param
        # and seasonal order param (the grid)
        aic_scores = []
        p = d = q = order_range
        M = seasonal_order_list
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, M))]
        
        # Using the grid, it builds a model for each unique combination 
        # to search for the model which the best performance using AIC metric
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    results = self.model_build(time_series_train, param, param_seasonal)

                    aic = [param, param_seasonal, results.aic] 
                    aic_scores.append(aic)
                    
                # Captures any errors
                except Exception as e:
                    print('Error: '+ str(e))
                    continue
                    
        return aic_scores


    def model_build(self, time_series_train, order_param, seasonal_order_param):
        """
        Builds the model and returns summary statistics.

        Parameters
        ----------
        time_series_train : DataFrame
            A dataframe to be used for training the model where index is 
            a time based data type e.g datetime or period(M) and only has
            one feature which is the target feature.

        order_param : tuple
            A tuple of 3 values representing p, d and q. This represents the
            number of AR parameters (p), differences (d), and MA (q) parameters.

        seasonal_order_param : tuple
            A tuple of 4 values representing P, D, Q and M. The 4 values are
            specifically for the seasonality conponent of the time series.
            Number of AR parameters (P), differences (D), MA (Q) parameter and
            single seasonality period (M).

        Returns
        -------
        results : MLEResults
            Class to hold results from fitting a state space model. The is the 
            output from calling .fit() on the model.
            
        """

        # Builds the SARIMA time series model
        mod = sm.tsa.statespace.sarimax.SARIMAX(time_series_train,
                                order = order_param,
                                seasonal_order = seasonal_order_param,
                                enforce_stationarity = False,
                                enforce_invertibility = False)
        # Trains the model
        results = mod.fit()
    
        return results        


    def model_diagnostics(self, model_result):
        """
        Plots the model diagnostics

        Parameters
        ----------
        model_result : MLEResults
            Class to hold results from fitting a state space model. The is the 
            output from calling .fit() on the model. 
        
        Returns
        -------
        fig : plot
            A plot of the diagnostics for the model

        """ 
        # Plots the model diagnostics
        fig = model_result.plot_diagnostics(figsize=(20, 8))

        return fig
        
        
    def plot_actuals_vs_forecast(self, full_time_series, order_param, seasonal_order_param, 
                    forecast_steps, y_lab, test_date, plot_date, datetime_col, significance_level=0.05):
        """
        Splits the time series into train and test based on the test date. It
        then builds the model with the train data and predicts for a specified 
        number of steps returning the forecasted values. Then 

        Parameters
        ----------
        full_time_series : DataFrame
            A dataframe to be used for training and testing the model where 
            index is a time based data type e.g datetime or period(M) and 
            only has one feature which is the target feature.

        order_param : tuple
            A tuple of 3 values representing p, d and q. This represents the
            number of AR parameters (p), differences (d), and MA (q) parameters.

        seasonal_order_param : tuple
            A tuple of 4 values representing P, D, Q and M. The 4 values are
            specifically for the seasonality conponent of the time series.
            Number of AR parameters (P), differences (D), MA (Q) parameter and
            single seasonality period (M).
   
        forecast_steps : int
            The number of observations to forecast for.

        y_lab : str
            Name of the y-axis

        test_date : str
            A date to start the testing the model from. Any date after 
            and including this will be used as part of the testing the 
            model. This should be in the same format as the datetime 
            column.

        plot_date : str
            A date to plot the results from. This date should be prior to 
            the test_date. This should be in the same format as the datetime 
            column.

        datetime_col : str
            Name of the date time column e.g. Date, Year-Month.

        significance_level : float, optional
            This significance level is used for the confidence interval. ie., 
            default value = 0.05 returns a 95% confidence interval. Confidence 
            interval shows the area where the actual value could be between, 
            the smaller the significance level, the larger the confidence interval 
            which returns a larger area. 
        
        Returns
        -------
        forecast_actuals : DataFrame
            A dataframe containing 3 columns the datetime column, the forecasted 
            observation values and the actual observation values.
            
        """ 

        # Splits the time series data based on the test_date
        train_data = full_time_series.loc[(full_time_series.index < test_date)]
        test_data = full_time_series.loc[(full_time_series.index >= test_date)]
        
        # Builds and trains the model
        res = self.model_build(train_data, order_param, seasonal_order_param)    
        
        # Plots the forecasted and actuals observation values
        forecast_r = self.plot_forecasts(res, forecast_steps, full_time_series, y_lab, \
                            plot_date, significance_level)
        plt.show(forecast_r)
        
        # Gets the forecasted and actual values into one dataframe
        forecasted_values = res.get_forecast(steps=forecast_steps).predicted_mean.to_frame().reset_index()
        forecasted_values.columns = (datetime_col, 'Forecasted')
        test_data = test_data.reset_index()
        test_data.columns = (datetime_col, 'Actuals')
        forecast_actuals = forecasted_values.merge(test_data, on = datetime_col)
        
        return forecast_actuals


    def plot_forecasts(self, results, forecast_steps, full_time_series, 
                                y_lab, date, significance_level=0.05):
        """
        Using the forecasted values and the actual values generates a plot 
        to visulise the Forecasted vs Actuals.
  
        Parameters
        ----------
        results : MLEResults
            Class to hold results from fitting a state space model. The is the 
            output from calling .fit() on the model.
            
        forecast_steps : int
            The number of observations to forecast for.

        full_time_series : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature. This
            should contain all the observations. 

        y_lab : str
            Name of the y-axis 
        
        date : str
            A date to start the plotting from. This doesn't have to be from the 
            start of the time series but it should be before the start point for
            the forecast. This should be in the same format as the datetime 
            column.

        significance_level : float, optional
            This significance level is used for the confidence interval. ie., 
            default value = 0.05 returns a 95% confidence interval. Confidence 
            interval shows the area where the actual value could be between, 
            the smaller the significance level, the larger the confidence interval 
            which returns a larger area. 

        Returns
        -------
        m_pred.get_figure() : plot
            A plot which shows the actual values, the forecasted values and the
            confidence interval for the forecasted values.

        """

        # Gets a set of forecasts and the associated confidence intervals
        pred_forecast = results.get_forecast(steps=forecast_steps)
        pred_ci = pred_forecast.conf_int(alpha=significance_level)
        
        # Selects the desired obervations to plot
        plot_data = full_time_series.loc[(full_time_series.index >= date)]

        # Plots the actual and forecasted values
        ax = plot_data.plot(label='Actuals', figsize=(20, 10))
        m_pred = pred_forecast.predicted_mean.plot(ax=ax, label='Forecast')

        # Adds the confidence intervals and shades them grey
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(y_lab)
        plt.title(("Forecasted Values using {0}% confidence interval").format(str(int(100-(significance_level*100)))))
        plt.legend()
        
        return m_pred.get_figure()
        
        
    def run_model_validation(self, grid_search_scores, time_series_train, freq_type_start, freq_type_end):
        """
        Runs the model build and cross validation functions for the top 5 aic scores

        Parameters
        ----------
        grid_search_scores : list
            A list containing a list of order_param, seasonal_order_param, 
            and results.aic which is the output of calling aic on model.fit().

        time_series_train : DataFrame
            A dataframe to be used for training the model where index is 
            a time based data type e.g datetime or period(M) and only has
            one feature which is the target feature.

        freq_type_start : dict 
            A dictionary where the key is the frequency of observations and 
            value is validation_period - the number of cross validation iterations 
            to use for testing the model's stabilty overtime. e.g. {'months' : -12} 

        freq_type_end : dict 
            A dictionary where the key is the frequency of observations and 
            value is no_forecasts - the number of observations to forecast for
            e.g. {'months' : 6} 
            
        Returns
        -------
        run_model_df : DataFrame
            A Dataframe containing the results of running cross_validation, the 
            dataframe contains the following columns 'order_param', 
            'seasonal_order_param', 'mape_total', 'mape_freq'. The word 'freq' is 
            replaced with the actual frequency of the data. 
            
        """ 

        # Gets the top 5 parameters using the AIC metric
        hyperparams = self.top_aic_scores(grid_search_scores) 

        # Gets the frequency and creates an empty dataframe
        freq = list(freq_type_start.keys())[0]
        run_model_df = pd.DataFrame(columns=['order_param', 'seasonal_order_param', 'mape_total', \
                                'mape_'+str(freq)])

        # Performs cross validation for each of the top 5 param 
        for order_x, seasonal_order_y, z in hyperparams:
            cv_res, mape_df = self.cross_validate(time_series_train, order_x, seasonal_order_y, freq_type_start, freq_type_end)
                    
            run_model_df = run_model_df.append(cv_res, ignore_index=True)

        return run_model_df


    def top_aic_scores(self, scores_list):
        """
        Returns the top 5 AIC scores

        Parameters
        ----------
        scores_list : list
            A list containing a list of order_param, seasonal_order_param, 
            and results.aic which is the output of calling aic on model.fit().
            
        Returns
        -------
        scores_list : list
            Returns the top 5 AIC score from the scores_list.

        """
        
        # Function to the 3rd element
        def top_score(elem):
            return elem[2]

        # Sorts the list by the 3rd element
        scores_list.sort(key = top_score)
    
        return scores_list[0:5] 

    
        


    
        
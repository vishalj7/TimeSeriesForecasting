
from tsf.TimeSeriesBaseAlgorithm import TimeSeriesBaseAlgorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import statsmodels.tsa.api as tsa
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose


class SARIMA_tsf(TimeSeriesBaseAlgorithm):
    
    def __init__(self):
        pass


    def cross_validate(self, timeseries, order_param, seasonal_order_param, 
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
        timeseries : DataFrame
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
            to use for testing the model's stabilty overtime. e.g. {'months' : -12} 

        freq_type_end : dict 
            A dictionary where the key is the frequency of observations and 
            value is no_forecasts - the number of observations to forecast for
            e.g. {'months' : 6} 

        Returns
        -------
        valuelist_df : DataFrame
            A dataframe containing the results of the cross validation. 
            There are 4 columns 'cross_validate_start_date', 'cross_validate_end_date',
            'mape_total', 'mape_freq'

        """

        validation_period = abs(list(freq_type_start.values())[0])
        no_forecasts = list(freq_type_end.values())[0]
        freq = list(freq_type_start.keys())[0]
        timeseries = timeseries.sort_index(ascending=True)

        valuelist = []
        timeseries_temp = timeseries.tail(validation_period)

        datelist = timeseries_temp.index.tolist()

        for date in datelist:
            cv_date = pd.to_datetime(date)
            cv_datestart = cv_date + relativedelta(**freq_type_start)
            cv_dateend = cv_datestart + relativedelta(**freq_type_end)

            cv_df= timeseries.loc[timeseries.index < cv_datestart]
            actual12 = timeseries.loc[(timeseries.index >= cv_datestart) & (timeseries.index < cv_dateend)]

            mod_cv = sm.tsa.statespace.sarimax.SARIMAX(cv_df,
                                        order=order_param,
                                        seasonal_order=seasonal_order_param,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

            results_cv = mod_cv.fit()
            pred_uc_cv = results_cv.get_forecast(steps=no_forecasts)

            pred12 = pred_uc_cv.predicted_mean
            actual12 = actual12[actual12.columns[0]]

            mape_total = super().mean_absolute_percentage_error(sum(actual12),sum(pred12))
            mape_freq = super().mean_absolute_percentage_error(actual12,pred12)

            values = [cv_datestart, cv_dateend, mape_total, mape_freq]
            valuelist.append(values)

        valuelist_df = pd.DataFrame(valuelist, columns=['cross_validate_start_date', 'cross_validate_end_date', 'mape_total', 'mape_'+str(freq)])
        
        lst_tot_mape = [item[2] for item in valuelist]
        print("Mean Absolute Percentage Error Total = " + str(round(np.mean(lst_tot_mape),2)))

        lst_freq_mape = [item[3] for item in valuelist]
        print("Mean Absolute Percentage Error {0} on {0} = ".format([list(freq_type_start)[0]]) + str(round(np.mean(lst_freq_mape),2)))

        return valuelist_df

    
    def decompose(self, timeseries, model_type):
        """
        Runs decompose on the time series to extract the trend component,
        seasonal component and the residual component. 

        
        Parameters
        ----------
        timeseries : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature.
            
        model_type : str
            There are 2 types: 'additive' is useful when the seasonal variation
            is relatively constant over time 'multiplicative' is useful when 
            the seasonal variation increases over time.

        frequency : tuple
            A tuple of 4 values representing P, D, Q and M. The 4 values are
            specifically for the seasonality conponent of the time series.
            Number of AR parameters (P), differences (D), MA (Q) parameter and
            single seasonality period (M).

        Returns
        -------
        decomposition : DecomposeResult
            Results class for seasonal decompositions which contains trend 
            component,cseasonal component and the residual component. 

        fig : plot
            A plot which shows the original, trend, seasonal and residual
            components of the time series.

        """

        fig, ax = plt.subplots(figsize=(20, 10))
        decomposition = seasonal_decompose(timeseries, model = model_type)

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.subplot(411)
        plt.plot(timeseries, label='Original')
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



    def forecast_vs_actual_plot(self, results, forecast_steps, full_timeseries, 
                                y_lab, date, confidence_interval=0.05):
        """
        Runs decompose on the time series to extract the trend component,
        seasonal component and the residual component. 

        
        Parameters
        ----------
        results : 
            
        forecast_steps : str
            There are 2 types: 'additive' is useful when the seasonal variation
            is relatively constant over time 'multiplicative' is useful when 
            the seasonal variation increases over time.

        full_timeseries : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature.

        y_lab : str
            Name of the y-axis 
        
        date : 

        confidence_interval : float
            The 

        Returns
        -------
        decomposition : DecomposeResult
            Results class for seasonal decompositions which contains trend 
            component,cseasonal component and the residual component. 

        fig : plot
            A plot which shows the original, trend, seasonal and residual
            components of the time series.

        """

        pred_uc = results.get_forecast(steps=forecast_steps)
        pred_ci = pred_uc.conf_int(alpha=confidence_interval)
        
        plot_data = all_data.loc[(all_data.index >= date)]

        ax = plot_data.plot(label='Acutals', figsize=(20, 10))
        m_pred = pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(y_lab)
        plt.legend()
        
        return m_pred



    def grid_search_run(self):
        print("done grid search")    
    


    def model_build(self):
        print("done build ")
        


    def model_forecast_result(self):
        pass 
    


    def result_evaluation(self):
        pass 



    def run_model_validation(self):
        pass 



    def top_aic_scores(self):
        pass 

    
        


    
        
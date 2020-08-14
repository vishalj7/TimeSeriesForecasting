
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

        valuelist_df = pd.DataFrame(valuelist, columns=['cross_validate_start_date', \
                                'cross_validate_end_date', 'mape_total', 'mape_'+str(freq)])
        
        lst_tot_mape = [item[2] for item in valuelist]
        print("Mean Absolute Percentage Error Total = " + str(round(np.mean(lst_tot_mape),2)))

        lst_freq_mape = [item[3] for item in valuelist]
        print("Mean Absolute Percentage Error {0} on {0} = ".format([list(freq_type_start)[0]]) + \
                str(round(np.mean(lst_freq_mape),2)))

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
                                y_lab, date, significance_level=0.05):
        """
        Using the forecasted values and the actual values generates a plot 
        to visulise the Forecasted vs Actuals.
  
        Parameters
        ----------
        results : MLEResults
            Class to hold results from fitting a state space model. The is the 
            output from calling .fit() on the model.
            
        forecast_steps : str
            The number of observations to forecast for.

        full_timeseries : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature. This
            should contain all the observations. 

        y_lab : str
            Name of the y-axis 
        
        date : str
            A date to start the plotting from. This doesn't have to be from the 
            start of the timeseries but it should be before the start point for
            the forecast. This should be in the same format as the datetime 
            column.

        significance_level : float
            This significance level is used for the confidence interval. ie., 
            default value = 0.05 returns a 95% confidence interval. Confidence 
            interval shows the area where the actual value could be between, 
            the smaller the significance level, the larger the confidence interval 
            which returns a larger area. 

        Returns
        -------
        m_pred.get_figure() : plot
            A plot which shows the acutal values, the forecasted values and the
            confidence interval for the forecasted values.

        """

        pred_uc = results.get_forecast(steps=forecast_steps)
        pred_ci = pred_uc.conf_int(alpha=significance_level)
        
        plot_data = full_timeseries.loc[(full_timeseries.index >= date)]

        ax = plot_data.plot(label='Acutals', figsize=(20, 10))
        m_pred = pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(y_lab)
        plt.title(("Forecasted vs Actual using {0} confidence interval").format(str(significance_level)))
        plt.legend()
        
        return m_pred.get_figure()


    def grid_search_run(self, timeseries_train):
        """
        Generates a grid search of parameters and then builds a 
        model with every combination of parameters. It then saves
        the AIC score and parameters into a list.

        Parameters
        ----------
        timeseries_train : DataFrame
            A dataframe to be used for training the model where index is 
            a time based data type e.g datetime or period(M) and only has
            one feature which is the target feature.
            
        Returns
        -------
        aic_scores : list
            A list of the order parameters, seasonal parameters and the respective AIC score
            
        """   

        aic_scores = []
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        
        # Using the grid search to perform the model build
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.sarimax.SARIMAX(timeseries_train,
                                                    order = param,
                                                    seasonal_order = param_seasonal,
                                                    enforce_stationarity = False,
                                                    enforce_invertibility = False)
                    results = mod.fit()
                    aic = [param, param_seasonal, results.aic] 
                    aic_scores.append(aic)
                    
                except Exception as e:
                    print('Error: '+ str(e))
                    continue
                    
        return aic_scores


    def grid_search_CV(self, timeseries, freq_type_start, freq_type_end):
        """
        Generates a grid search of parameters and then builds a 
        model with every combination of parameters. It then saves
        the AIC score and parameters into a list.

        Parameters
        ----------
        timeseries : DataFrame
            A dataframe where index is a time based data type e.g datetime or 
            period(M) and only has one feature which is the target feature.
            
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
        aic_scores : list
            A list of the order parameters, seasonal parameters and the respective AIC score
            
        """   

        aic_scores = []
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        
        freq = list(freq_type_start.keys())[0]
        grid_cv_df = pd.DataFrame(columns=['cross_validate_start_date', 
                            'cross_validate_end_date', 'mape_total', 'mape_'+str(freq), 
                            'order_param', 'seasonal_param'])

        # Using the grid search to perform the model build
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                print("Order: {0}, Seasonal_Order: {1}".format(param, param_seasonal))            
                cv_res = self.cross_validate(timeseries, param, param_seasonal, freq_type_start, freq_type_end)

                cv_res['order_param'] = str(param)
                cv_res['seasonal_param'] = str(param_seasonal)
                print(cv_res)

                grid_cv_df = grid_cv_df.append(cv_res, ignore_index=True)


        return grid_cv_df



    def model_build(self, timeseries_train, order_param, seasonal_order_param):
        """
        Builds the model and returns summary statistics.

        Parameters
        ----------
        timeseries_train : DataFrame
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

        mod = sm.tsa.statespace.sarimax.SARIMAX(timeseries_train,
                                order = order_param,
                                seasonal_order = seasonal_order_param,
                                enforce_stationarity = False,
                                enforce_invertibility = False)
        results = mod.fit()
        print("Summary Results Table.") 
        print(results.summary().tables[1])
    
        return results        


    def model_forecast_result(self, full_timeseries, order_param, seasonal_order_param, 
                    forecast_steps, y_lab, test_date, plot_date, datetime_col, significance_level=0.05):
        """
        Builds the model with all the data and predicts for a specified number 
        of steps returning the forecasted values.

        Parameters
        ----------
        full_timeseries : DataFrame
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
   
        forecast_steps : str
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

        significance_level : float
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

        train_data = full_timeseries.loc[(full_timeseries.index < test_date)]
        test_data = full_timeseries.loc[(full_timeseries.index >= test_date)]

        res = self.model_build(train_data, order_param, seasonal_order_param)    
        
        forecast_r = self.forecast_vs_actual_plot(res, forecast_steps, full_timeseries, y_lab, \
                            plot_date, significance_level)
        plt.show(forecast_r)

        forecasted_values = res.get_forecast(steps=forecast_steps).predicted_mean.to_frame().reset_index()
        forecasted_values.columns = (datetime_col, 'Forecasted')
        test_data = test_data.reset_index()
        test_data.columns = (datetime_col, 'Actuals')
        forecast_actuals = forecasted_values.merge(test_data, on = datetime_col)
        
        return forecast_actuals


    def result_evaluation(self, cr_result):
        """
        Plots the diagnostics for each set of hyperparameters

        Parameters
        ----------
        cr_result : list
            A list containing result of performing cross validation on each 
            of the top 5 grid score searches. 
            
        """ 
        # for x in cr_result:
        #     print("Hyperparameters: Order{0}, Seasonal_Order{1}".format(x[0][0], x[0][1]))
        #     r = x[1]
        #     r.plot_diagnostics(figsize=(20, 8))
        #     plt.show()
        
        # print("")



    def run_model_validation(self, grid_search_scores, timeseries_train, freq_type_start, freq_type_end):
        """
        Runs the model build and cross validation functions for the top 5 aic scores

        Parameters
        ----------
        grid_search_scores : list
            A list containing a list of order_param, seasonal_order_param, 
            and results.aic which is the output of calling aic on model.fit().

        timeseries_train : DataFrame
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
        cr_result : list
            A list containing result of performing cross validation on each 
            of the top 5 grid score searches. 
            
        """ 

        hyperparams = self.top_aic_scores(grid_search_scores) 

        cr_result = []
        for order_x, seasonal_order_y, z in hyperparams:
            print("Order: {0}, Seasonal_Order: {1}".format(order_x, seasonal_order_y))            
            cr_res = self.cross_validate(timeseries_train, order_x, seasonal_order_y, freq_type_start, freq_type_end)
            cr_result.append([(order_x, seasonal_order_y) ,cr_res])
            print("")
            print("")

        return cr_result


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
        def top_score(elem):
            return elem[2]
    
        scores_list.sort(key = top_score)
    
        return scores_list[0:5] 

    
        


    
        
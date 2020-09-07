## SARIMA

This document covers what SARIMA is and when it should be used. To see how to run this algorithm, please view the scripts in the sample_script folder, the scripts will provide a guide on how to run this algorithm for 2 different datetime formats daily and monthly.
<br></br>

### What is SARIMA?

SARIMA or Seasonal Autoregressive Integrated Moving Average is a univariate time series algorithm that supports seasonal component within the time series data. It is based of ARIMA (or Autoregressive Integrated Moving Average) where ARIMA can't handle time series data that has a seasonal component. 

SARIMA can handle time series data that contains a seasonal pattern as the algorithm can perform differencing on the data to essentially remove the effect of the seasonal pattern. 



<br></br>
### When should SARIMA be used?

When you want to use a single data column to make predicitons (univariate), it doesn't matter if the data has more than one column but before using the SARIMA algorithm, you need to make sure there is only a single data column and the index is in a datetime format. 

If after performing the dickey_fuller_test and/or seeing a trend and seasonal pattern using the decompose function, it is then deemed the data isn't stationary and it has a seasonal pattern. If this is the case for your data, this is when you should be using SARIMA as it has parameters where you can specify the seasonal patter frequency and apply differencing. 

There are 2 set of parameters for the SARIMA model the first is the order parameters which is a tuple of 3 values representing p, d and q. This represents the number of AR parameters (p), differences (d), and MA (q) parameters. These parameters are to handle the trend pattern within the time series data just like ARIMA.

The second set of parameters is the seasonal order parameters which is a tuple of 4 values representing P, D, Q and M. The 4 values are specifically for the seasonality conponent of the time series. Number of AR parameters (P), differences (D), MA (Q) parameter and single seasonality period (M). The value passed of D is the differencing applied to the seasonal pattern of the time series data. The seasonal pattern is specified by the value M.  

# TimeSeriesForecasting

TimeSeriesForecasting is a Python library for performing time series analysis and 
forecasting future values.

## Dependencies
TimeSeriesForecasting requires:

+ Python (>= 3.5)
+ Pandas (>= 1.0.1)
+ NumPy (>= 1.18.1)
+ MatplotLib
+ Statsmodels (>= 0.11.0)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install TimeSeriesForecasting.

First, download the package via github and then extract the folder. Next step would be to change directory to the extracted folder and run the following command. 

```bash
cd Downloads/TimeSeriesForecasting

pip install .
```

## Usage

The first step would be to import the package and the particular algorithm you would like i.e. SARIMA_tsf. Next, you would need to create the object based on the algorithm and this will allow you to call the relevant functions. 

```python
from tsf.SARIMA_tsf import SARIMA_tsf

tsf = SARIMA_tsf()

tsf.dickey_fuller_test(time_series_data) # returns the result of the dickey fuller test
```

## Data Format

This package requires the data in a dataframe where the index is a datetime format and the frequency of the data has been set using '.asfreq()' and the column/s to be in a numeric format whether that be as an integer or floating point

## Sample Scripts

There are samples scripts in the sample_scripts folder which provides examples on how to use this package for various datetime frequencies. The sample scripts provide a full in depth usage of the TimeSeriesForecasting package as it:
+ imports the package
+ creates the object
+ performs the stationary checks
+ performs hyperparameter tuning
+ builds the model with the optimal hyperparameters
+ performs forecasting of future observations
+ performs model diagnostic

These scripts can then be used as a template and there are 2 versions. One where the datetime frequency is montly and another script where the frequency is daily. 

### Note 
If you would like to run these scripts please make sure the data folder and the files inside is kept in the same directory as the sample_scritps folder. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/) General Public License v3.0 or later
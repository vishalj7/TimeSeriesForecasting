import unittest
import random
import pandas
import numpy
import statsmodels
import matplotlib

from tsf.SARIMA_tsf import SARIMA_tsf

class sarima_test(unittest.TestCase):

    def test_object(self):
        tsf = SARIMA_tsf()
        self.assertIs(type(tsf), SARIMA_tsf)

    def test_method_availability(self):
        tsf = SARIMA_tsf()
        mape = tsf.mean_absolute_percentage_error([2,45, 675, 6345, 45463], [5, 44, 657, 5464, 46536])
        self.assertIsNotNone(mape)
        self.assertEqual(mape, 34.23)

    def test_method_existance(self):
        tsf = SARIMA_tsf()
        try:
            method_test = tsf.made_up_method('abc', 123)
        except Exception as e:
            method_test = e
        self.assertIs(type(method_test), AttributeError)

    def test_top_aic_scores_less_than_5(self):
        test_list = [[(0,0,1), (0,0,0,1), 362], [(0,0,1), (0,0,0,1), 23], [(0,0,1), (0,0,0,1), 77]]

        tsf = SARIMA_tsf()
        top_scores = tsf.top_aic_scores(test_list)

        self.assertEqual(top_scores, [[(0, 0, 1), (0, 0, 0, 1), 23], [(0, 0, 1), (0, 0, 0, 1), 77], [(0, 0, 1), (0, 0, 0, 1), 362]])

    def test_model_build(self):
        random.seed(42)
        x = random.random()
        
        data = []
        for x in range(500):
            value = round(random.random(),3)
            data.insert((x+1), value)

        idx = pandas.date_range('2010-01-01', periods=500, freq='D')
        ts = pandas.Series(data=data, index=idx).to_frame()
        ts.columns = ['value']

        tsf = SARIMA_tsf()
    
        try:
            res = tsf.model_build(ts, (1,1,0), (1,0,1,14))
        except Exception:
            self.fail("tsf.model_build() raised Exception unexpectedly!")

        self.assertIs(type(res), statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper)

    
    def test_decompose_returns_pic(self):
        random.seed(42)
        x = random.random()
        
        data = []
        for x in range(500):
            value = round(random.random(),3)
            data.insert((x+1), value)

        idx = pandas.date_range('2010-01-01', periods=500, freq='D')
        ts = pandas.Series(data=data, index=idx).to_frame()
        ts.columns = ['value']

        tsf = SARIMA_tsf()

        decompose, decom_fig = tsf.decompose(ts, 'additive')
        
        self.assertIs(type(decom_fig), matplotlib.figure.Figure)


    def test_plot_actuals_vs_forecast(self):
        random.seed(42)
        x = random.random()
        
        data = []
        for x in range(365):
            value = round(random.random(),3)
            data.insert((x+1), value)

        idx = pandas.date_range('2020-01-01', periods=365, freq='D')
        ts = pandas.Series(data=data, index=idx).to_frame()
        ts.columns = ['value']

        ts.index.rename('daily')
        tsf = SARIMA_tsf()

        try:
            forecast_actuals = tsf.plot_actuals_vs_forecast(ts, (1,1,0), (1,0,1,14), 14, \
             "test values", "2020-12-17", "2020-01-01", 'daily', 0.05)
        except Exception:
            self.fail("tsf.plot_actuals_vs_forecast() raised Exception unexpectedly!")

        self.assertIs(type(forecast_actuals), pandas.core.frame.DataFrame)

if __name__ == '__main__':
    unittest.main()
import unittest
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

    

if __name__ == '__main__':
    unittest.main()
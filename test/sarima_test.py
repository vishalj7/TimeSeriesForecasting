import unittest
from tsf.SARIMA_tsf import SARIMA_tsf

class sarima_test(unittest.TestCase):

    def test_object(self):
        tsf = SARIMA_tsf()
        self.assertIs(type(tsf), SARIMA_tsf)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
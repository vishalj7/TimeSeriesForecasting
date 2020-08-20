from distutils.core import setup

DESCRIPTION = 'A package that helps to perform time series analysis and forecasting'

setup(
    name='TimeSeriesForecasting',
    version='0.1.0',
    author='V. JHAVERI',
    author_email='vishal.jhaveri@capgemini.com',
    packages=['tsf', 'test'],
    license='LICENSE.txt',
    data_files=[('data', ['data/daily_temp.csv', 'data/car_sales_by_month.csv'])],
    description='A package that helps to perform time series analysis and forecasting',
    long_description=open('README.txt').read(),
    python_requires='>=3.5, <4',
    install_requires=[
        "pandas >= 1.0.1",
        "numpy >= 1.18.1",
        "matplotlib",
        "statsmodels >= 0.11.0"
    ]
)

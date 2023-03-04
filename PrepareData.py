from datetime import datetime

from pandas import read_csv


# load data
# def parse(x):
#     return datetime.strptime(x, '%Y %m %d')


dataset = read_csv(r'TheData/Data\alldata.csv', delimiter=';', header=0)

# print(dataset.columns.values.tolist())
# dataset = dataset[['MaxLoad']]  ### , 'max_temp', 'min_temp', 'weekday','Month'
# dataset['MaxLoad'] = dataset['MaxLoad'] / 1000
print(dataset.columns.values.tolist())
#
# # removing seasonal data
# import statsmodels.api as sm
#
# # https://stackoverflow.com/questions/20672236/time-series-decomposition-function-in-python
# # res = sm.tsa.seasonal_decompose(dataset.MaxLoad, model='additive')
# # resplot = res.plot()
# # resplot.savefig("tt.png")
# # dataset['MaxLoad'] = res.resid
# # dataset.dropna(inplace=True)
#
#
# from stldecompose import decompose, forecast
# from stldecompose.forecast_funcs import (naive, drift, mean, seasonal_naive)
#
# stl = decompose(dataset.MaxLoad, period=365)
# stlplot = stl.plot()
# stlplot.savefig("stl_tt.png")

# save to file
dataset.to_csv('Load.csv')

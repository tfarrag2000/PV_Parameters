from pandas import read_csv
from datetime import datetime
import numpy as np


# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d')


dataset = read_csv(r'All_Data.csv', delimiter=',')

print(dataset.columns.values.tolist())
print(dataset)

# save to file
dataset.to_csv('Load.csv')

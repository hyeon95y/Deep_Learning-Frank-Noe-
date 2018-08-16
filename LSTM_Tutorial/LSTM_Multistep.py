# reference
# https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/

# load and plot dataset
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
import os

# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
series = read_csv(os.path.join(os.path.dirname(__file__), "shampoo-sales.csv"),
                          header=0,
                          parse_dates=[0],
                          index_col=0,
                          squeeze=True,
                          skipfooter=1,
                          date_parser=parser
                            )
# summarize first few rows
print(series.head())
# line plot
series.plot()
#plt.show()

# convert time series into supervised learning problem
# deafult value : n_in=1, n_out=1, dropnan=True
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True) :
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        print('* i : %d, n_in : %d, n_out : %d' % (i, n_in, n_out))
        print('* input')
        print('* df.shift\n', df.shift(i))
        print('* names\n', names)
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out) :
        cols.append(df.shift(-i))
        if i == 0 :
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else :
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        print('* forcast')
        print(df.shift(-i))
        print(names)
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    print('* raw_values \n', raw_values)
    raw_values = raw_values.reshape(len(raw_values), 1)
    # transform into supervised learning problem x, y
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    supervised_values = supervised.values
    print('supervised_values\n', supervised_values)
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test

# configure
n_lag = 1
n_seq = 3
n_test = 10

# prepare dataset
train, test = prepare_data(series, n_test, n_lag, n_seq)
print(test)
print('Train: %s, Test: %s' % (train.shape, test.shape))
print(train[0])

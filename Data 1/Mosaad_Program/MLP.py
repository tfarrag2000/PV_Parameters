from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import numpy as np



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def removing_seasonal_data(data, columnname, dropnan=True):
    import statsmodels.api as sm

    # https://stackoverflow.com/questions/20672236/time-series-decomposition-function-in-python
    res = sm.tsa.seasonal_decompose(data[columnname])

    resplot = res.plot()
    # save to file
    resplot.savefig("seasonal_1.png")
    data[columnname] = res.resid
    data.dropna(inplace=dropnan)
    return data


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def prepare_data(n_days=3, n_features=1, n_train_days=365 * 11, n_epochs=2, n_batch=64, n_neurons=4, verbosevalue=2):
    # load dataset
    dataset = read_csv('load.csv', header=0, index_col=0, parse_dates=True)
    # dataset = removing_seasonal_data(dataset, 'MaxLoad')
    values = dataset.values
    print(dataset)
    # ensure all data is float
    # values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_days, 1, dropnan=True)

    # split into train and test sets
    values = reframed.values
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]

    # split into input and outputs
    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    # print(train_X.shape, len(train_X), train_y.shape)

    # design network
    model = Sequential()
    model.add(Dense(21, activation='relu', input_dim=train_X.shape[1]))
    # model.add(Dropout(0.2))
    model.add(Dense(5, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch, validation_data=(test_X, test_y),
                        verbose=verbosevalue,
                        shuffle=False)

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_days * n_features))

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -n_features + 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -n_features + 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    MAPE = mean_absolute_percentage_error(inv_y, inv_yhat)
    print('Test MAPE: ' + str(MAPE))
    print(model.summary())
    print(list(zip(inv_y, inv_yhat)))

    # plot history loss
    pyplot.close()
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig("yyy.png")
    pyplot.close()

    # plot actual vs predicted
    pyplot.plot(inv_y, label='actual')
    pyplot.plot(inv_yhat, label='predicted')
    pyplot.legend()
    pyplot.savefig("yyy1.png")
    pyplot.close()


for i in range(1, 2):
    prepare_data(n_features=4, n_neurons=50, verbosevalue=2)

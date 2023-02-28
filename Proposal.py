import numpy as np
import os
import tensorflow as tf
import uuid
from math import sqrt
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import ExcelWriter
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Sequential

os.environ["PATH"] += os.pathsep + r'C:/Program Files (x86)/Graphviz2.38/bin/'
from tensorflow.keras.utils import plot_model

# experiment parameters
parameters = dict()


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


# def removing_seasonal_data(data, columnname, dropnan=True):
#     from stldecompose import decompose
#     # https://stackoverflow.com/questions/20672236/time-series-decomposition-function-in-python
#     stl = decompose(data[columnname])
#     stlplot = stl.plot()
#     # save to file
#     stlplot.savefig("stl_seasonal_1.png")
#     data[columnname] = stl.resid
#     data.dropna(inplace=dropnan)
#     return data


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def load_prepare_data():
    # load dataset
    dataset = read_csv('Load.csv', header=0, index_col=0, parse_dates=True)

    values = dataset.values
    print(dataset)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, parameters["n_days"], 1, dropnan=True)

    return reframed, scaler


def plotting_save_experiment_data(model, history, y_actual, y_predicted):
    print(model.summary())
    # save data
    y_actual = y_actual[:-1]
    y_predicted = y_predicted[1:]
    writer = ExcelWriter('experimentOutput\\' + parameters["ID"] + 'results.xlsx')
    df = DataFrame.from_dict(parameters, orient='index')
    df.columns = ['value']
    df.to_excel(writer, 'parameters')
    df = DataFrame(list(zip(y_actual, y_predicted)), columns=['y_actual', 'y_predicted'])
    df.to_excel(writer, 'predicted')
    df = DataFrame(list(zip(history.history['loss'], history.history['val_loss'])), columns=['loss', 'val_loss'])
    df.to_excel(writer, 'loss_history')
    writer.save()
    writer.close()

    # plot history
    plot_model(model, to_file='experimentOutput\\' + parameters["ID"] + 'model_fig.png', show_shapes=True,
               show_layer_names=True)

    # plot history loss
    pyplot.close()
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='test_loss')
    # pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='MAPE')
    pyplot.legend()
    pyplot.savefig('experimentOutput\\' + parameters["ID"] + "loss_fig.png")
    pyplot.close()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    print('Test RMSE: %.3f' % rmse)
    # calculate MAPE
    MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    print('Test MAPE: %.3f' % MAPE)
    min_train_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    print('min val_loss: %.3f' % min_val_loss)

    # plot actual vs predicted
    pyplot.plot(y_actual, label='actual')
    pyplot.plot(y_predicted, label='predicted')
    pyplot.legend()
    pyplot.title('RMSE: %.3f' % rmse + " , " + 'MAPE: %.3f' % MAPE)
    pyplot.savefig('experimentOutput\\' + parameters["ID"] + "forcast_fig.png")
    pyplot.close()

    # save to database
    if parameters["save_to_database"]:
        import MySQLdb
        db = MySQLdb.connect(host="localhost", user="tfarrag", passwd="mansoura", db="mydata")
        # prepare a cursor object using cursor() method
        cursor = db.cursor()
        s = model.summary()
        sql = """INSERT INTO experiments (experiment_ID, n_days, n_features, n_traindays, n_epochs, n_batch, 
        n_neurons,earlystop, RMSE, MAPE, min_train_loss, min_val_loss, Model_summary) VALUES ('{}',{},{},{}, {}, {}, 
        {}, {}, {:.4f},{:.4f}, {:.4f}, {:.4f}, '{}')""" \
            .format(parameters["ID"], parameters["n_days"], parameters["n_features"], parameters["n_traindays"],
                    parameters["n_epochs"], parameters["n_batch"], parameters["n_neurons"], parameters["earlystop"],
                    rmse, MAPE, min_train_loss, min_val_loss, s)

        try:
            cursor.execute(sql)
            db.commit()
            print("** saved to database")
        except:
            db.rollback()

        db.close()


def create_fit_model(data, scaler):
    # split into train and test sets
    values = data.values
    print(values.shape)
    train = values[:parameters["n_traindays"], :]
    test = values[parameters["n_traindays"]:, :]
    # split into input and outputs
    n_obs = parameters["n_days"] * parameters["n_features"]
    train_X, train_y = train[:, :n_obs], train[:, -parameters["n_features"]]
    test_X, test_y = test[:, :n_obs], test[:, -parameters["n_features"]]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], parameters["n_days"], parameters["n_features"]))
    test_X = test_X.reshape((test_X.shape[0], parameters["n_days"], parameters["n_features"]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(parameters["n_neurons"], input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='Adam')
    # fit network
    clbs = None
    if parameters["earlystop"]:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
        clbs = [earlyStopping]

    history = model.fit(train_X, train_y, epochs=parameters["n_epochs"], batch_size=parameters["n_batch"],
                        validation_data=(test_X, test_y),
                        verbose=parameters["model_train_verbose"],
                        shuffle=False, callbacks=clbs)

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], parameters["n_days"] * parameters["n_features"]))

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -parameters["n_features"] + 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -parameters["n_features"] + 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    plotting_save_experiment_data(model, history, inv_y, inv_yhat)


def run_experiment():
    data, scaler = load_prepare_data()

    create_fit_model(data, scaler)


def main():
    import datetime

    now = datetime.datetime.now()
    parameters["ID"] = now.strftime("%Y%m%d%H%M")  # uuid.uuid4().hex
    parameters["n_days"] = 4
    parameters["n_features"] = 4
    parameters["n_traindays"] = 365 * 11
    parameters["n_epochs"] = 50
    parameters["n_batch"] = 128
    parameters["n_neurons"] = 64
    parameters["model_train_verbose"] = 2
    parameters["earlystop"] = True
    parameters["save_to_database"] = False

    # https: // tensorflow.rstudio.com / blog / time - series - forecasting -
    # with-recurrent - neural - networks.html

    run_experiment()
    import gc
    gc.collect()


main()

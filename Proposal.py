import datetime
import logging
import os
import traceback
from math import sqrt
import MySQLdb

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import plot_model
from matplotlib import pyplot
from numpy import concatenate
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

os.environ["PATH"] += os.pathsep + r'C:/Program Files (x86)/Graphviz2.38/bin/'


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, col_names=None):
    """
    Convert time series data to a supervised learning format.

    Arguments:
    data -- the time series data (array or DataFrame)
    n_in -- the number of lagged time steps to include as input (default 1)
    n_out -- the number of time steps to predict (default 1)
    dropnan -- whether to drop rows with NaN values (default True)
    col_names -- list of column names for the resulting DataFrame (optional)

    Returns:
    DataFrame -- the converted time series data in a supervised format
    """

    # ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # get number of columns in data
    n_vars = data.shape[1]

    # set column names
    if col_names is None:
        col_names = []
        for i in range(n_vars):
            col_names += [('var%d' % (i + 1))]
    elif len(col_names) != n_vars:
        raise ValueError("Number of column names must match number of variables in data.")

    # initialize empty lists for columns and column names
    cols, names = [], []

    # create input sequence columns (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col_names[j], i)) for j in range(n_vars)]

    # create output sequence columns (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('%s(t)' % col_names[j]) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (col_names[j], i)) for j in range(n_vars)]

    # concatenate columns and set column names
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


# convert series to supervised learning
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg


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


def load_prepare_data(parameters):
    # load dataset
    dataset = read_csv('TheData/Load.csv', header=0, index_col=0, parse_dates=True)

    values = dataset.values
    print(dataset)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, parameters["n_days"], 5, dropnan=True)

    return reframed, scaler


def plotting_save_experiment_data(model, history, y_actual, y_predicted, parameters):
    """
    Save the experiment data to files and a database, and plot the results.

    Args:
        model: A Keras model object.
        history: A Keras history object.
        y_actual: A NumPy array of actual y values.
        y_predicted: A NumPy array of predicted y values.
        parameters: A dictionary of experiment parameters.

    Returns:
        None
    """
    # Save data
    y_actual = y_actual[:-1]
    y_predicted = y_predicted[1:]
    data_dict = {
        'parameters': pd.DataFrame.from_dict(parameters, orient='index', columns=['value']),
        'predicted': pd.DataFrame({'y_actual': y_actual, 'y_predicted': y_predicted}),
        'loss_history': pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']}),
    }
    writer = pd.ExcelWriter(f'experimentOutput/{parameters["ID"]}results.xlsx')
    for sheet_name, data in data_dict.items():
        data.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    writer.close()

    # Create output directory if it does not exist
    if not os.path.exists('experimentOutput'):
        os.makedirs('experimentOutput')

    # Plot model and save to file
    plot_model(model, to_file=f'experimentOutput/{parameters["ID"]}model_fig.png',
               show_shapes=True, show_layer_names=True)

    # Plot history loss and save to file
    pyplot.close()
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='test_loss')
    pyplot.legend()
    pyplot.savefig(f'experimentOutput/{parameters["ID"]}loss_fig.png')
    pyplot.close()

    # Calculate RMSE and MAPE
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    print(f'Test RMSE: {rmse:.3f}')
    MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    print(f'Test MAPE: {MAPE:.3f}')
    min_train_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    print(f'min val_loss: {min_val_loss:.3f}')

    # Plot actual vs predicted and save to file
    pyplot.plot(y_actual, label='actual')
    pyplot.plot(y_predicted, label='predicted')
    pyplot.legend()
    pyplot

    # save to database
    if parameters["save_to_database"]:
        with MySQLdb.connect(host="localhost", user="root", passwd="M@nsoura2000", db="pv_db") as db:
            cursor = db.cursor()
            s = model.summary()
            sql = """INSERT INTO experiments (experiment_ID, n_days, n_features, n_traindays, n_epochs, n_batch, 
            n_neurons,earlystop, RMSE, MAPE, min_train_loss, min_val_loss, Model_summary) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            values = (parameters["ID"], parameters["n_days"], parameters["n_features"], parameters["n_traindays"],
                      parameters["n_epochs"], parameters["n_batch"], parameters["n_neurons"], parameters["earlystop"],
                      rmse, MAPE, min_train_loss, min_val_loss, s)
            try:
                cursor.execute(sql, values)
                db.commit()
                print("** saved to database")
            except:
                db.rollback()

    # if parameters["save_to_database"]:
    #     import MySQLdb
    #     db = MySQLdb.connect(host="localhost", user="root", passwd="M@nsoura2000", db="pv_db")
    #     # prepare a cursor object using cursor() method
    #     cursor = db.cursor()
    #     s = model.summary()
    #     sql = """INSERT INTO experiments (experiment_ID, n_days, n_features, n_traindays, n_epochs, n_batch,
    #     n_neurons,earlystop, RMSE, MAPE, min_train_loss, min_val_loss, Model_summary) VALUES ('{}',{},{},{}, {}, {},
    #     {}, {}, {:.4f},{:.4f}, {:.4f}, {:.4f}, '{}')""" \
    #         .format(parameters["ID"], parameters["n_days"], parameters["n_features"], parameters["n_traindays"],
    #                 parameters["n_epochs"], parameters["n_batch"], parameters["n_neurons"], parameters["earlystop"],
    #                 rmse, MAPE, min_train_loss, min_val_loss, s)
    #
    #     try:
    #         cursor.execute(sql)
    #         db.commit()
    #         print("** saved to database")
    #     except:
    #         db.rollback()
    #
    #     db.close()


# def create_fit_model(data, scaler):
#     # split into train and test sets
#     values = data.values
#     print(values.shape)
#     train = values[:parameters["n_traindays"], :]
#     test = values[parameters["n_traindays"]:, :]
#     # split into input and outputs
#     n_obs = parameters["n_days"] * parameters["n_features"]
#     train_X, train_y = train[:, :n_obs], train[:, -parameters["n_features"]]
#     test_X, test_y = test[:, :n_obs], test[:, -parameters["n_features"]]
#     # reshape input to be 3D [samples, timesteps, features]
#     train_X = train_X.reshape((train_X.shape[0], parameters["n_days"], parameters["n_features"]))
#     test_X = test_X.reshape((test_X.shape[0], parameters["n_days"], parameters["n_features"]))
#     print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#
#     # design network
#     model = Sequential()
#     model.add(LSTM(parameters["n_neurons"], input_shape=(train_X.shape[1], train_X.shape[2])))
#     model.add(Dropout(0.5))
#     model.add(Dense(32))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#
#     model.compile(loss='mean_squared_error', optimizer='Adam')
#     # fit network
#     clbs = None
#     if parameters["earlystop"]:
#         earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
#         clbs = [earlyStopping]
#
#     history = model.fit(train_X, train_y, epochs=parameters["n_epochs"], batch_size=parameters["n_batch"],
#                         validation_data=(test_X, test_y),
#                         verbose=parameters["model_train_verbose"],
#                         shuffle=False, callbacks=clbs)
#
#     # make a prediction
#     yhat = model.predict(test_X)
#     test_X = test_X.reshape((test_X.shape[0], parameters["n_days"] * parameters["n_features"]))
#
#     # invert scaling for actual
#     test_y = test_y.reshape((len(test_y), 1))
#     inv_y = concatenate((test_y, test_X[:, -parameters["n_features"] + 1:]), axis=1)
#     inv_y = scaler.inverse_transform(inv_y)
#     inv_y = inv_y[:, 0]
#
#     # invert scaling for forecast
#     inv_yhat = concatenate((yhat, test_X[:, -parameters["n_features"] + 1:]), axis=1)
#     inv_yhat = scaler.inverse_transform(inv_yhat)
#     inv_yhat = inv_yhat[:, 0]
#
#     plotting_save_experiment_data(model, history, inv_y, inv_yhat)

def create_fit_model(data, scaler, parameters):
    try:
        # split into train and test sets
        values = data.values
        logging.info(f"Data shape: {values.shape}")
        train = values[:parameters["n_traindays"], :]
        test = values[parameters["n_traindays"]:, :]
        # split into input and outputs
        n_obs = parameters["n_days"] * parameters["n_features"]
        train_X, train_y = train[:, :n_obs], train[:, -parameters["n_features"]]
        test_X, test_y = test[:, :n_obs], test[:, -parameters["n_features"]]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], parameters["n_days"], parameters["n_features"]))
        test_X = test_X.reshape((test_X.shape[0], parameters["n_days"], parameters["n_features"]))
        logging.info(
            f"Train X shape: {train_X.shape}, Train y shape: {train_y.shape}, Test X shape: {test_X.shape}, Test y shape: {test_y.shape}")

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

        plotting_save_experiment_data(model, history, inv_y, inv_yhat, parameters)

    except Exception as e:
        logging.error(f"Error in create_fit_model: {e}")
        logging.error(traceback.format_exc())
        raise


def run_experiment(parameters):
    data, scaler = load_prepare_data(parameters)
    create_fit_model(data, scaler, parameters)


def main():
    """
    Set up and run a machine learning experiment.
    """
    try:
        now = datetime.datetime.now()
        parameters = dict()
        parameters = {
            "ID": now.strftime("%Y%m%d%H%M"),
            "n_days": 1,
            "n_features": 7,
            "n_traindays": 365 * 11,
            "n_epochs": 50,
            "n_batch": 128,
            "n_neurons": 64,
            "model_train_verbose": 2,
            "earlystop": True,
            "save_to_database": False
        }
        run_experiment(parameters)
    except Exception as e:
        print("Error running experiment:", e)


if __name__ == "__main__":
    main()

    # https: // tensorflow.rstudio.com / blog / time - series - forecasting -
    # with-recurrent - neural - networks.html

    # import gc
    # gc.collect()

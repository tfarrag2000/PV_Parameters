import numpy as np
import os
import uuid
from math import sqrt
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import ExcelWriter
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.layers import LSTM, GRU
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import regularizers
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

os.environ["PATH"] += os.pathsep + r'C:/Program Files (x86)/Graphviz2.38/bin/'
# from tensorflow.keras.utils import plot_model

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


# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#

def load_prepare_data():
    # load dataset
    dataset = read_csv('Load.csv', header=0, index_col=0, parse_dates=True)
    pp = parameters
    values = dataset.values
    print(dataset)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    # reframed = series_to_supervised(scaled, parameters["n_days"], 1, dropnan=True)
    df = DataFrame(scaled)
    df.dropna(inplace=True)

    return df, scaler


def plotting_save_experiment_data(model, y_actual, y_predicted):
    # print(model.summary())
    # save data
    # y_actual = y_actual[:-1]
    # y_predicted = y_predicted[1:]
    writer = ExcelWriter('experimentOutput\\' + parameters["ID"] + 'results.xlsx')
    df = DataFrame.from_dict(parameters, orient='index')
    df.columns = ['value']
    df.to_excel(writer, 'parameters')
    df = DataFrame(list(zip(y_actual, y_predicted)), columns=['y_actual', 'y_predicted'])
    df.to_excel(writer, 'predicted')
    # df = DataFrame(list(zip(history.history['loss'], history.history['val_loss'])), columns=['loss', 'val_loss'])
    # df.to_excel(writer, 'loss_history')
    writer.save()
    writer.close()

    # plot history
    # plot_model(model, to_file='experimentOutput\\' + parameters["ID"] + 'model_fig.png', show_shapes=True,
    #            show_layer_names=True)

    # plot history loss
    pyplot.close()
    # pyplot.plot(history.history['loss'], label='train_loss')
    # pyplot.plot(history.history['val_loss'], label='test_loss')
    # # pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='MAPE')
    # pyplot.legend()
    # pyplot.savefig('experimentOutput\\' + parameters["ID"] + "loss_fig.png")
    # pyplot.close()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    print('Test RMSE: {}'.format(rmse))
    # calculate MAPE
    MAPE = mean_absolute_percentage_error(y_actual, y_predicted, multioutput='raw_values')
    print('Test MAPE: {}'.format(MAPE))
    MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    print('Test MAPE: {}'.format(MAPE))
    # min_train_loss = min(history.history['loss'])
    # min_val_loss = min(history.history['val_loss'])
    # print('min val_loss: %.3f' % min_val_loss)

    # plot actual vs predicted
    pyplot.plot(y_actual, label='actual')
    pyplot.plot(y_predicted, label='predicted')
    pyplot.legend()
    pyplot.title('RMSE: %.3f' % rmse + " , " + ' MAPE: {}'.format(MAPE))
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
    pp = parameters
    values = data.values
    print(values.shape)
    train = values[:parameters["n_traindays"], :]
    test = values[parameters["n_traindays"]:, :]
    # split into input and outputs
    n_obs = parameters["n_days"] * parameters["n_features"]
    train_X, train_y = train[:, :n_obs], train[:, n_obs:]
    test_X, test_y = test[:, :n_obs], test[:, n_obs:]

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model = MLPRegressor(random_state=1, max_iter=500).fit(train_X, train_y)
    # make a prediction

    yhat = model.predict(test_X)
    # invert scaling for actual
    test_ = concatenate((test_X, test_y), axis=1)
    inv_y_ = scaler.inverse_transform(test_)
    inv_y = inv_y_[:, n_obs:]

    # invert scaling for forecast
    test_hat = concatenate((test_X, yhat), axis=1)
    inv_yhat_ = scaler.inverse_transform(test_hat)
    inv_yhat = inv_yhat_[:, n_obs:]

    ############# Module 1 , 2, 3 #####################################################
    test = [[2.22639, 0.4861, 4.58, 0.61, 4.97, -0.374, 0.088994, 0, 0, 0, 0, 0],
            [3.1281, 0.48125, 6.5, 0.6, 7.12, -0.356, 0.045, 0, 0, 0, 0, 0],
            [3.40278, 0.4861, 7, 0.575, 8.06, -0.32, 0.04, 0, 0, 0, 0, 0]]
    test = scaler.transform(test)
    test1 = [t[:n_obs] for t in test]
    prediction = model.predict(test1)
    test_ = concatenate((test1, prediction), axis=1)
    prediction_ = scaler.inverse_transform(test_)
    prediction_ = prediction_[:, n_obs:]
    print(prediction_)
    ####################################################################################

    plotting_save_experiment_data(model, inv_y, inv_yhat)


def run_experiment():
    data, scaler = load_prepare_data()

    create_fit_model(data, scaler)


def main():
    import datetime

    now = datetime.datetime.now()
    parameters["ID"] = "ANN_" + now.strftime("%Y%m%d%H%M")  # uuid.uuid4().hex
    parameters["n_days"] = 1
    parameters["n_features"] = 7
    parameters["n_traindays"] = 365 * 11
    parameters["n_epochs"] = 50
    # parameters["n_batch"] = 128
    parameters["n_neurons"] = 15
    parameters["model_train_verbose"] = 2
    parameters["earlystop"] = False
    parameters["save_to_database"] = False

    run_experiment()
    import gc
    gc.collect()


main()

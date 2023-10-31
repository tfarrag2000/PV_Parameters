import datetime
import json
from math import sqrt
import os
from pickle import dump, load

import mysql.connector
import numpy
import numpy as np
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import ExcelWriter
from pandas import read_csv
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.callbacks import EarlyStopping ,ModelCheckpoint,TensorBoard
from keras.layers import Dense, Dropout
from keras.models import load_model,Sequential
from keras.utils import plot_model


mainDir = os.path.dirname(os.path.abspath(__file__))

class ParametersForecasting:
    parameters = dict()
    id = ""
    MAPE = -1
    outputIndex = -1
    ANN_arch_list = []
    model = None

    def __init__(self,
                 n_epochs=250,
                 Dropout=0,
                 n_batch=256,
                 model_train_verbose=2,
                 earlystop=True,
                 save_to_database=True,
                 optimizer="Adam",
                 ActivationFunctions='tanh',
                 ANN_arch=None,
                 comment="",
                 outputIndex=-1
                 , model=None):
        '''

         outputIndex:
             -1 : all  (default)
             1 : Rs
             2: Iph
             3: I0
             4: Rp
             5: n
        '''

        self.model = model
        self.ANN_arch_list = []
        now = datetime.datetime.now()
        self.id = now.strftime("%Y%m%d%H%M%S")  # uuid.uuid4().hex
        self.parameters["ID"] = self.id
        self.parameters["n_epochs"] = int(n_epochs)
        self.parameters["Dropout"] = Dropout
        self.parameters["n_batch"] = int(n_batch)
        self.parameters["model_train_verbose"] = model_train_verbose
        self.parameters["earlystop"] = earlystop
        self.parameters["save_to_database"] = save_to_database
        self.parameters["optimizer"] = optimizer
        self.parameters["ANN_arch"] = ANN_arch
        self.parameters["comment"] = comment

        self.parameters["ActivationFunctions"] = ActivationFunctions

        self.outputIndex = outputIndex
        # convert series to supervised learning

    def mean_absolute_percentage_error(self, y_true, y_pred):
        # Ensure that denominators are not zero
        non_zero_denominators = np.where(y_true != 0)
        
        # Calculate absolute percentage errors only for non-zero denominators
        absolute_percentage_errors = np.abs((y_true[non_zero_denominators] - y_pred[non_zero_denominators]) / y_true[non_zero_denominators])
        
        # Calculate the mean MAPE for non-zero denominators
        mean_mape = np.mean(absolute_percentage_errors) * 100
        
        return mean_mape

    def load_prepare_data(self):
        # load dataset
        path=os.path.join(mainDir,'TheData\\data_without_outliers.csv')
        dataset = read_csv(path, header=0, index_col=0, parse_dates=True)

        self.cols = ['Pm', 'Vm', 'Im', 'Voc', 'Isc', 'Voc_coeff', 'Isc_coeff', 'Rs', 'Iph', 'I0', 'Rp', 'n']
        if (self.outputIndex != -1):
            self.cols = ['Pm', 'Vm', 'Im', 'Voc', 'Isc', 'Voc_coeff', 'Isc_coeff', self.cols[6 + self.outputIndex]]


        dataset = dataset[self.cols]
        self.col_inputs = self.cols[0:7]
        self.col_actual = [x + '_actual' for x in self.cols[7:]]
        self.col_predicted = [x + '_predicted' for x in self.cols[7:]]

        ## remove outliers  we do this step before and save the clean data
        # z_scores = zscore(dataset)
        # abs_z_scores = np.abs(z_scores)
        # filtered_entries = (abs_z_scores < 3).all(axis=1)
        # dataset = dataset[filtered_entries]

        values = dataset.values
        # ensure all data is float
        values = values.astype('float32')
        valuesInputs=values[:, 0:7]
        valuesOutputs=values[:, 7:]

        # normalize features
        scalerInputs = MinMaxScaler(feature_range=(-1, 1))
        scalerOutputs = MinMaxScaler(feature_range=(-1, 1))
        scalerInputs.fit(valuesInputs)
        scalerOutputs.fit(valuesOutputs)
        scaledInputs = scalerInputs.transform(valuesInputs)
        scaledOutputs = scalerOutputs.transform(valuesOutputs)
        dump(scalerInputs, open(mainDir + 'results\\Models\\' + self.parameters["ID"] + '_scalerInputs.pkl', 'wb'))
        dump(scalerOutputs, open(mainDir + 'results\\Models\\' + self.parameters["ID"] + '_scalerOutputs.pkl', 'wb'))

        return scaledInputs, scaledOutputs, scalerInputs, scalerOutputs

    def plotting_save_experiment_data(self, model, history, actual_inputs, actual_outputs, predicted_outputs):
        # plot history
        plot_model(model, to_file=mainDir + 'results\\experimentOutput\\' + self.parameters["ID"] + 'model_fig.png'
                   , show_shapes=False, show_layer_names=True,
                   rankdir='TB', expand_nested=False, dpi=96)

        print(model.summary())
        # y_actual = y_actual[:-1,:]
        # y_predicted = y_predicted[1:,:]
        print("******************************************")
        print(actual_outputs.shape, predicted_outputs.shape)
        # save data to excel file
        writer = ExcelWriter(mainDir + '\\experimentOutput\\' + self.parameters["ID"] + 'results.xlsx')
        df = DataFrame.from_dict(self.parameters, orient='index')
        df.columns = ['value']
        df.to_excel(writer, 'self.parameters')

        writer = ExcelWriter(mainDir + '\\experimentOutput\\' + self.parameters["ID"] + 'results.xlsx')
        df = DataFrame.from_dict(self.parameters, orient='index')
        df.columns = ['value']
        df.to_excel(writer, 'self.parameters')
        df = DataFrame(list(actual_inputs), columns=self.col_inputs)
        df.to_excel(writer, 'inputs')
        df = DataFrame(list(actual_outputs), columns=self.col_actual)
        df.to_excel(writer, 'actual')
        df = DataFrame(list(predicted_outputs), columns=self.col_predicted)
        df.to_excel(writer, 'predicted')
        all = concatenate((actual_inputs, actual_outputs, predicted_outputs), axis=1)
        df = DataFrame(all, columns=self.col_inputs + self.col_actual + self.col_predicted)
        df.to_excel(writer, 'The Data')


        df = DataFrame(list(zip(history.history['loss'], history.history['val_loss'])),
                       columns=['loss', 'val_loss'])
        df.to_excel(writer, 'loss_history')

        # plot and save history loss
        pyplot.close()
        pyplot.plot(history.history['loss'], label='train_loss')
        pyplot.plot(history.history['val_loss'], label='test_loss')
        # pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='MAPE')
        pyplot.legend()
        pyplot.savefig(mainDir + '\\experimentOutput\\' + self.parameters["ID"] + "loss_fig.png")
        pyplot.close()
        # calculate RMSE
        mse = mean_squared_error(actual_outputs, predicted_outputs, multioutput='raw_values')
        rmse = sqrt(mean_squared_error(actual_outputs, predicted_outputs))
        # calculate MAPE
        MAPE = self.mean_absolute_percentage_error(actual_outputs, predicted_outputs)
        min_train_loss = min(history.history['loss'])
        min_val_loss = min(history.history['val_loss'])

        resultssummary = dict()
        resultssummary["Test RMSE"] = rmse
        resultssummary["Test MAPE"] = MAPE
        print(actual_outputs.shape[1])
        if actual_outputs.shape[1] == 1:
            resultssummary[" Median ABE"] = median_absolute_error(actual_outputs, predicted_outputs)
        resultssummary["min train_loss"] = min_train_loss
        resultssummary["val_loss"] = min_val_loss
        df = DataFrame.from_dict(resultssummary, orient='index')
        df.columns = ['value']
        df.to_excel(writer, 'results summary')

        print('Test RMSE: %.5f' % rmse)
        print('Test MAPE: %.5f' % MAPE)
        print('min train_loss: %.5f' % min_train_loss)
        print('min val_loss: %.5f' % min_val_loss)

        # writer.save()
        writer.close()

        # plot and save actual vs predicted
        for i in range(actual_outputs.shape[1]):
            pyplot.scatter(range(actual_outputs.shape[0]), actual_outputs[:, i], s=6, label="y{}_actual".format(i))
            pyplot.plot(predicted_outputs[:, i], label='y{}_predicted'.format(i))
        pyplot.legend()
        pyplot.title('RMSE: %.3f' % rmse + " , " + 'MAPE: %.3f' % MAPE)
        figure = pyplot.gcf()
        figure.set_size_inches(16, 7)
        pyplot.savefig(mainDir + '\\experimentOutput\\' + self.parameters["ID"] + "forcast_fig.png")
        pyplot.close()

        # save to database
        if self.parameters["save_to_database"]:

            # Load database connection details from the configuration file
            with open(mainDir + '\\config.json', 'r') as config_file:
                config = json.load(config_file)

            # Extract database connection parameters
            db_host = config['database']['host']
            db_user = config['database']['user']
            db_password = config['database']['password']
            db_name = config['database']['database_name']

            # Establish a database connection
            db = mysql.connector.connect(host=db_host, user=db_user, passwd=db_password, db=db_name)

            # prepare a cursor object using cursor() method
            cursor = db.cursor()

            sql = """INSERT INTO experiments (experiment_ID, n_epochs, n_batch, 
                outputIndex, ANN_arch,Dropout, earlystop,  RMSE, MAPE, 
                min_train_loss, min_val_loss, Model_summary,comment,optimizer,ActivationFunctions) VALUES ('{}',{},{},{}, '{}',{} ,{}, 
                {:.6f},{:.6f}, {:.6f}, {:.6f}, '{}', '{}', '{}','{}')""" \
                .format(self.parameters["ID"],
                        self.parameters["n_epochs"], self.parameters["n_batch"], self.outputIndex, self.ANN_arch_list,
                        self.parameters["Dropout"],
                        self.parameters["earlystop"],
                        rmse, MAPE, min_train_loss, min_val_loss,model.summary(),
                        self.parameters["comment"], self.parameters["optimizer"],
                        self.parameters["ActivationFunctions"])

            try:
                cursor.execute(sql)
                db.commit()
                print("** saved to database")
            except TypeError as e:
                print(e)
                db.rollback()
                print(sql)
                print("** Error saving to database")

            db.close()

            self.MAPE = MAPE
        return MAPE

    def checkDataBase(self):

        # Load database connection details from the configuration file
        with open(mainDir + '\\config.json', 'r') as config_file:
            config = json.load(config_file)
        # Extract database connection parameters
        db_host = config['database']['host']
        db_user = config['database']['user']
        db_password = config['database']['password']
        db_name = config['database']['database_name']

        # Establish a database connection
        db = mysql.connector.connect(host=db_host, user=db_user, passwd=db_password, db=db_name)
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        sql = "select experiment_ID, MAPE from experiments where ANN_arch='{}' and n_batch ={} and Dropout={}  and outputIndex={} and optimizer='{}' and ActivationFunctions='{}'" \
            .format(self.ANN_arch_list, self.parameters["n_batch"], self.parameters["Dropout"], self.outputIndex,
                    self.parameters["optimizer"], self.parameters["ActivationFunctions"])

        try:
            cursor.execute(sql)
            records = cursor.fetchall()
            for row in records:
                self.MAPE = row[1]
                return True
        except TypeError as e:
            print(e)

        return False

    def create_fit_model(self, scaledInputs, scaledOutputs, scalerInputs, scalerOutputs):

        inputs = numpy.array(scaledInputs)
        outputs = numpy.array(scaledOutputs)


        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=10)


        #############################Create The The Model########################################
        if self.model is None and self.parameters["ANN_arch"] is not None:

            inputs = keras.Input(shape=(7,))

            layer_pre = inputs

            for n in self.parameters["ANN_arch"]:
                if n != 0:
                    layer=Dense(n, activation=self.parameters["ActivationFunctions"])(layer_pre)
                    layer_pre=layer
                    if self.parameters["Dropout"] != 0:
                        layer=Dropout(self.parameters["Dropout"])(layer_pre)
                        layer_pre = layer
                    self.ANN_arch_list.append(n)


            layer=Dense(y_train.shape[1], activation=self.parameters["ActivationFunctions"])(layer_pre)
            self.ANN_arch_list.append(y_train.shape[1])

            self.model = keras.Model(inputs=inputs, outputs=layer, name=self.parameters["ID"])

            # self.model = Sequential(name=self.parameters["ID"])
            # i=1
            # for n in self.parameters["ANN_arch"]:
            #     if n != 0:
            #         self.model.add(Dense(n, activation=self.parameters["ActivationFunctions"],name="dense"+str(i)))
            #         if self.parameters["Dropout"] != 0:
            #             self.model.add(Dropout(self.parameters["Dropout"]))
            #         self.ANN_arch_list.append(n)
            #         i=i+1
            # self.model.add(Dense(y_train.shape[1], activation=self.parameters["ActivationFunctions"],name="dense"+str(i)))
            #
            # print(y_train.shape[1])
            # self.ANN_arch_list.append(y_train.shape[1])

            # if self.checkDataBase():
            #     print("###################### already exist model")
            #     return
        ##############################################################################
        #### Train the model
        self.model.compile(loss='mean_squared_error', optimizer=self.parameters["optimizer"])

        # callbacks
        mc = ModelCheckpoint(mainDir + 'results\\Models\\' + self.parameters["ID"] + '_best_model.h5',
                             monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
        logdir = mainDir + 'results\\logs\\scalars\\' + self.parameters["ID"]
        print(logdir)
        tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)
        clbs = [mc, tensorboard_callback]
        if self.parameters["earlystop"]:
            earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2, mode='auto')
            clbs.append(earlyStopping)

        # fit network

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)
        print("Model Input Layer Shape:", self.model.layers[0].input_shape)

        history = self.model.fit(X_train, y=y_train, epochs=self.parameters["n_epochs"],
                                 batch_size=self.parameters["n_batch"],
                                 validation_data=(X_test, y_test), verbose=self.parameters["model_train_verbose"],
                                 shuffle=True, callbacks=clbs)

        self.parameters["n_epochs"] = len(history.history["loss"])

        # self.model.save(mainDir + '\\Models\\' + self.parameters["ID"] + '_Last_model.h5')

        self.model = load_model(mainDir + 'results\\Models\\' + self.parameters["ID"] + '_best_model.h5')

        # make a prediction
        y_test_predicted = self.model.predict(X_test)

        # invert scaling for actual
        actual_inputs = scalerInputs.inverse_transform(X_test)
        actual_outputs= scalerOutputs.inverse_transform(y_test)
        # actual_data = concatenate((actual_inputs, actual_outputs), axis=1)

        # invert scaling for forecast
        predicted_outputs= scalerOutputs.inverse_transform(y_test_predicted)
        # Predicted_data = concatenate((actual_inputs, Predicted_outputs), axis=1)

        self.plotting_save_experiment_data(self.model, history, actual_inputs, actual_outputs,predicted_outputs)

    def start_experiment(self):
        scaledInputs, scaledOutputs, scalerInputs, scalerOutputs = self.load_prepare_data()

        print("Scaled Inputs shape:", scaledInputs.shape)
        print("Scaled Outputs shape:", scaledOutputs.shape)

        self.create_fit_model(scaledInputs, scaledOutputs, scalerInputs, scalerOutputs)
        return self.MAPE

if __name__ == "__main__":
    f = ParametersForecasting(n_epochs=500, n_batch=256, ANN_arch=[32, 64, 256, 256, 5], model_train_verbose=1,save_to_database=True, outputIndex=-1,
                            comment="test" ,ActivationFunctions='tanh' )
    f.start_experiment()

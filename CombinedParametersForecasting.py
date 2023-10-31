import datetime
from math import sqrt
import os
from tensorflow import keras
import mysql.connector
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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from keras.utils import plot_model
from tensorflow import keras


# os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# experiment self.parameters
mainDir = os.path.dirname(os.path.abspath(__file__))


class CombinedParametersForecasting:
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
        self.id = now.strftime("%Y%m%d%H%M%S") # uuid.uuid4().hex
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

    def  CreateCombinedModel(self):
        # list = ['20200726221917', '20200726140302', '20200726114513', '20200726051814', '20200726064251']
        # Dir = os.path.dirname(os.path.abspath(__file__))+'\\Models\\'
        # models = []
        # for f in list:
        #     fn = f + '_best_model.h5'
        #     funModel = load_model(Dir + fn)
        #     models.append(funModel)

        input_layer = keras.Input(shape=(7,))

        # merge input models
        # outModules = [f.output for f in models]
        # merge = keras.layers.concatenate(outModules)

        # layer_pre=merge
        i=1
        for n in self.parameters["ANN_arch"]:
            if n != 0:
                layer = Dense(n, activation=self.parameters["ActivationFunctions"],name="dense_c_"+str(i))(layer_pre)
                layer_pre = layer
                if self.parameters["Dropout"] != 0:
                    layer = Dropout(self.parameters["Dropout"])(layer_pre)
                    layer_pre = layer
                self.ANN_arch_list.append(n)
            i=i+1

        output = Dense(5, activation=self.parameters["ActivationFunctions"],name="dense_c_"+str(i))(layer_pre)
        self.ANN_arch_list.append(5)



        model = keras.Model(inputs=input_layer, outputs=output)
        # summarize layers
        # plot_model(model, to_file='D:\\multiple_inputsxxxxx.png')
        return model

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def load_prepare_data(self):
        # load dataset
        dataset = read_csv('TheData/data_without_outliers.csv', header=0, index_col=0, parse_dates=True)

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
        # normalize features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(values)
        return scaled, scaler

    def plotting_save_experiment_data(self, model, history, inputs, y_actual, y_predicted):
        # plot history
        plot_model(model, to_file=mainDir + '\\experimentOutput\\' + self.parameters["ID"] + 'model_fig.png'
                   , show_shapes=False, show_layer_names=True,
                   rankdir='TB', expand_nested=False, dpi=96)
        # stringlist = []
        # model.summary(print_fn=lambda x: stringlist.append(x))
        # short_model_summary = "\n".join(stringlist)

        print(model.summary())
        # y_actual = y_actual[:-1,:]
        # y_predicted = y_predicted[1:,:]
        print("******************************************")
        print(y_actual.shape, y_predicted.shape)
        # save data to excel file
        writer = ExcelWriter(mainDir + '\\experimentOutput\\' + self.parameters["ID"] + 'results.xlsx')
        df = DataFrame.from_dict(self.parameters, orient='index')
        df.columns = ['value']
        df.to_excel(writer, 'self.parameters')



        writer = ExcelWriter(mainDir + '\\experimentOutput\\' + self.parameters["ID"] + 'results.xlsx')
        df = DataFrame.from_dict(self.parameters, orient='index')
        df.columns = ['value']
        df.to_excel(writer, 'self.parameters')
        df = DataFrame(list(inputs), columns=self.col_inputs)
        df.to_excel(writer, 'inputs')
        df = DataFrame(list(y_actual), columns=self.col_actual)
        df.to_excel(writer, 'actual')
        df = DataFrame(list(y_predicted), columns=self.col_predicted)
        df.to_excel(writer, 'predicted')
        all = concatenate((inputs, y_actual, y_predicted), axis=1)
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
        mse_multioutput = mean_squared_error(y_actual, y_predicted, multioutput='raw_values')
        rmse = sqrt(mean_squared_error(y_actual, y_predicted))
        # calculate MAPE
        MAPE = self.mean_absolute_percentage_error(y_actual, y_predicted)
        MAPE_multioutput = []
        for i in range(y_actual.shape[1]):
            MAPE_multioutput.append(self.mean_absolute_percentage_error(y_actual[:, i], y_predicted[:, i]))

        print(MAPE_multioutput)

        min_train_loss = min(history.history['loss'])
        min_val_loss = min(history.history['val_loss'])

        resultssummary = dict()
        resultssummary["Test MSE_multioutput"] = str(mse_multioutput)
        resultssummary["Test MAPE_multioutput"] = str(MAPE_multioutput)
        resultssummary["Test RMSE"] = rmse
        resultssummary["Test MAPE"] = MAPE
        resultssummary["min train_loss"] = min_train_loss
        resultssummary["val_loss"] = min_val_loss
        df = DataFrame.from_dict(resultssummary, orient='index')
        df.columns = ['value']
        df.to_excel(writer, 'results summary')

        print('Test RMSE: %.5f' % rmse)
        print('Test MAPE: %.5f' % MAPE)
        print('Test RMSE: %.5f' % rmse)
        print('Test MAPE: %.5f' % MAPE)
        print('min train_loss: %.5f' % min_train_loss)
        print('min val_loss: %.5f' % min_val_loss)

        writer.save()
        writer.close()

        # plot and save actual vs predicted
        for i in range(y_actual.shape[1]):
            pyplot.scatter(range(y_actual.shape[0]), y_actual[:, i], s=6, label="y{}_actual".format(i))
            pyplot.plot(y_predicted[:, i], label='y{}_predicted'.format(i))
        pyplot.legend()
        pyplot.title('RMSE: %.3f' % rmse + " , " + 'MAPE: %.3f' % MAPE)
        figure = pyplot.gcf()
        figure.set_size_inches(16, 7)
        pyplot.savefig(mainDir + '\\experimentOutput\\' + self.parameters["ID"] + "forcast_fig.png")
        pyplot.close()

        # save to database
        if self.parameters["save_to_database"]:
            db = mysql.connector.connect(host="localhost", user="root", passwd="*********", db="mydata")
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

        db = mysql.connector.connect(host="localhost", user="root", passwd="*******", db="myData")
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        sql = "select experiment_ID, MAPE from experiments where ANN_arch='{}' and n_batch ={} and Dropout={}  and outputIndex={} and optimizer='{}' " \
              "and ActivationFunctions='{}' and comment like '%combined%' " \
            .format(self.ANN_arch_list, self.parameters["n_batch"], self.parameters["Dropout"], self.outputIndex,
                    self.parameters["optimizer"], self.parameters["ActivationFunctions"])
        print(sql)

        try:
            cursor.execute(sql)
            records = cursor.fetchall()
            for row in records:
                self.MAPE = row[1]
                return True
        except TypeError as e:
            print(e)

        return False

    def create_fit_model(self, data, scaler):
        # split into train and test sets
        data = numpy.array(data)
        # print(data[0, :])
        inputs = data[:, :7]
        outputs = data[:, 7:]
        # print(inputs[0, :])
        # print(outputs[0, :])

        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=10)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        #############################Create The The Model########################################
        self.model = self.CreateCombinedModel()

        # if self.checkDataBase():
        #     print("###################### already exist model")
        #     return

        self.model.compile(loss='mean_squared_error', optimizer=self.parameters["optimizer"])
        ##############################################################################

        # callbacks
        mc = ModelCheckpoint(mainDir + '\\Models\\' + self.parameters["ID"] + '_best_model.h5',
                             monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
        logdir = mainDir + 'logs\\scalars\\' + self.parameters["ID"]
        print(logdir)
        tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)
        clbs = [mc, tensorboard_callback]
        if self.parameters["earlystop"]:
            earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto')
            clbs.append(earlyStopping)

        # fit network
        history = self.model.fit(x=[X_train,X_train,X_train,X_train,X_train] , y=y_train, epochs=self.parameters["n_epochs"],
                                 batch_size=self.parameters["n_batch"],
                                 validation_data=([X_test,X_test,X_test,X_test,X_test], y_test), verbose=self.parameters["model_train_verbose"],
                                 shuffle=True, callbacks=clbs)

        self.parameters["n_epochs"] = len(history.history["loss"])

        self.model.save(mainDir + '\\Models\\' + self.parameters["ID"] + '_Last_model.h5')

        self.model = load_model(mainDir + '\\Models\\' + self.parameters["ID"] + '_best_model.h5')

        # make a prediction
        y_test_predicted = self.model.predict([X_test,X_test,X_test,X_test,X_test] )

        # invert scaling for actual
        actual_data = concatenate((X_test, y_test), axis=1)
        actual_data = scaler.inverse_transform(actual_data)
        print(y_test_predicted.shape)
        # invert scaling for forecast
        Predicted_data = concatenate((X_test, y_test_predicted), axis=1)
        Predicted_data = scaler.inverse_transform(Predicted_data)

        print("******************************************")
        print(actual_data.shape, Predicted_data.shape)

        y_test_actual = actual_data[:, 7:]
        y_test_predicted = Predicted_data[:, 7:]
        self.plotting_save_experiment_data(self.model, history,actual_data[:,0:7], y_test_actual, y_test_predicted)

    def start_experiment(self):
        data, scaler = self.load_prepare_data()
        self.create_fit_model(data, scaler)
        return self.MAPE



if __name__ == "__main__":
    f = CombinedParametersForecasting(n_epochs=1 , outputIndex=-1, earlystop=True, comment="Combined_Model", ANN_arch=[8,5],save_to_database=False)
    f.start_experiment()

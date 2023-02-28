import datetime
from math import sqrt

import mysql.connector
import mysql.connector
import numpy
import numpy as np
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import ExcelWriter
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# experiment self.parameters
mainDir = 'E:\\My Research Results\\Dr_Mosaad_Data3\\'


class TestModel:
    parameters = dict()
    id = ""
    MAPE = -1
    outputIndex = -1
    ANN_arch_list = []

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
                 ):
        '''

         outputIndex:
             -1 : all  (default)
             1 : Rs
             2: Iph
             3: I0
             4: Rp
             5: n
        '''

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
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def load_prepare_data(self):
        # load dataset
        dataset = read_csv('data_without_outliers.csv', header=0, index_col=0, parse_dates=True)

        self.cols = ['Pm', 'Vm', 'Im', 'Voc', 'Isc', 'Voc_coeff', 'Isc_coeff', 'Rs', 'Iph', 'I0', 'Rp', 'n']
        if (self.outputIndex != -1):
            self.cols = ['Pm', 'Vm', 'Im', 'Voc', 'Isc', 'Voc_coeff', 'Isc_coeff', self.cols[6 + self.outputIndex]]
            dataset = dataset[self.cols]

        self.col_inputs = self.cols[0:7]
        self.col_actual = [x + '_actual' for x in self.cols[7:]]
        self.col_predicted = [x + '_predicted' for x in self.cols[7:]]

        ## remove outliers
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




    def plotting_save_experiment_data(self, model, inputs, y_actual, y_predicted):
        # plot history
        plot_model(model, to_file=mainDir + 'experimentOutput\\' + self.parameters["ID"] + 'model_fig.png'
                   , show_shapes=False, show_layer_names=True,
                   rankdir='TB', expand_nested=False, dpi=96)

        # print(model.summary())
        # save data to excel file
        writer = ExcelWriter(mainDir + 'experimentOutput\\' + self.parameters["ID"] + 'results.xlsx')
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

        # calculate RMSE
        mse_multioutput = mean_squared_error(y_actual, y_predicted, multioutput='raw_values')
        rmse = sqrt(mean_squared_error(y_actual, y_predicted))
        # calculate MAPE
        MAPE = self.mean_absolute_percentage_error(y_actual, y_predicted)
        MAPE_multioutput = []
        for i in range(y_actual.shape[1]):
            MAPE_multioutput.append(self.mean_absolute_percentage_error(y_actual[:, i], y_predicted[:, i]))

        print(MAPE_multioutput)

        resultssummary = dict()
        resultssummary["Test MSE_multioutput"] = str(mse_multioutput)
        resultssummary["Test MAPE_multioutput"] = str(MAPE_multioutput)
        resultssummary["Test RMSE"] = rmse
        resultssummary["Test MAPE"] = MAPE
        df = DataFrame.from_dict(resultssummary, orient='index')
        df.columns = ['value']
        df.to_excel(writer, 'results summary')

        print('Test RMSE: %.5f' % rmse)
        print('Test MAPE: %.5f' % MAPE)

        self.MAPE = MAPE

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
        pyplot.savefig(mainDir + 'experimentOutput\\' + self.parameters["ID"] + "forcast_fig.png")
        pyplot.close()

        # save to database
        if self.parameters["save_to_database"]:
            db = mysql.connector.connect(host="localhost", user="root", passwd="mansoura", db="dr_mosaad_data3")
            # prepare a cursor object using cursor() method
            cursor = db.cursor()

            sql = """INSERT INTO experiments (experiment_ID, n_epochs, n_batch, 
                       outputIndex, ANN_arch,Dropout, earlystop,  RMSE, MAPE, Model_summary,comment,optimizer,ActivationFunctions) VALUES ('{}',{},{},{}, '{}',{} ,{}, 
                       {:.6f},{:.6f}, '{}', '{}', '{}','{}')""" \
                .format(self.parameters["ID"],
                        self.parameters["n_epochs"], self.parameters["n_batch"], self.outputIndex, self.ANN_arch_list,
                        self.parameters["Dropout"],
                        self.parameters["earlystop"],
                        rmse, self.MAPE, model.summary(),
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

    def checkDataBase(self):

        db = mysql.connector.connect(host="localhost", user="root", passwd="mansoura", db="dr_mosaad_data3")
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        sql = "select experiment_ID, MAPE from experiments where ANN_arch='{}' and n_batch ={} and Dropout={}  and " \
              "outputIndex={} and optimizer='{}' and ActivationFunctions='{}'" \
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

    def Predict_From_model(self, data, scaler, model):
        data = numpy.array(data)
        inputs = data[:, :7]
        outputs = data[:, 7:]

        # _, inputs, _, outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=10)

        y_test_predicted = model.predict([inputs for x in range(outputs.shape[1])])
        # y_test_predicted = model.predict(inputs)

        uu = numpy.array(y_test_predicted).transpose()
        yy = uu.reshape((-1, outputs.shape[1]))

        # invert scaling for actual
        actual_data = concatenate((inputs, outputs), axis=1)
        actual_data = scaler.inverse_transform(actual_data)

        # invert scaling for forecast
        Predicted_data = concatenate((inputs, yy), axis=1)
        Predicted_data = scaler.inverse_transform(Predicted_data)

        print("******************************************")
        print(actual_data.shape, Predicted_data.shape)

        y_test_actual = actual_data[:, 7:]
        y_test_predicted = Predicted_data[:, 7:]
        self.plotting_save_experiment_data(model, inputs, y_test_actual, y_test_predicted)

    def CreateModel(self):
        list = ['20200726221917', '20200726140302', '20200726114513', '20200726051814', '20200726064251']
        Dir = 'E:\\My Research Results\\Dr_Mosaad_Data3\\Models\\'
        models = []
        for f in list:
            fn = f + '_best_model.h5'
            funModel = load_model(Dir + fn)
            models.append(funModel)

        input_layer = keras.Input(shape=(7,))

        # merge input models
        outModules = [f.output for f in models]
        # merge = keras.layers.concatenate(outModules)
        model = keras.Model(inputs=[f.input for f in models], outputs=outModules)
        # summarize layers
        plot_model(model, to_file='E:\\ccc.png')
        return model

    def start_experiment(self):

        model = self.CreateModel()
        # model = load_model(mainDir + 'Models\\' + '20200726060320' + '_best_model.h5')
        data, scaler = self.load_prepare_data()
        self.Predict_From_model(data, scaler, model)
        return self.MAPE


f = TestModel(n_epochs=1000, outputIndex=-1, earlystop=True, comment="Test Merged Model")
f.start_experiment()

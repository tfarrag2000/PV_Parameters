import datetime
import os
from math import sqrt
from pickle import dump, load

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
                 save_to_database=True,
                 optimizer="Adam",
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

        self.parameters["save_to_database"] = save_to_database
        self.parameters["optimizer"] = optimizer
        self.parameters["comment"] = comment
        self.outputIndex = outputIndex
        # convert series to supervised learning

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



    def load_prepare_data_assesment(self):
        # prepare scalar
        dataset = read_csv('data_without_outliers.csv', header=0, index_col=0, parse_dates=True)

        self.cols = ['Pm', 'Vm', 'Im', 'Voc', 'Isc', 'Voc_coeff', 'Isc_coeff', 'Rs', 'Iph', 'I0', 'Rp', 'n']
        dataset = dataset[self.cols]
        values = dataset.values
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scalerInputs = MinMaxScaler(feature_range=(-1, 1))
        scalerOutputs= MinMaxScaler(feature_range=(-1, 1))
        scalerInputs.fit(values[:,0:7])
        scalerOutputs.fit(values[:,7:])


        # scalerInputs= load(open(mainDir + 'Models\\' + '20200820003813' + '_scalerInputs.pkl', 'rb'))
        # scalerOutputs= load(open(mainDir + 'Models\\' + '20200820003813' + '_scalerOutputs.pkl', 'rb'))

        # load dataset
        dataset = read_csv('PVs for assessments.csv', header=0, index_col=0, parse_dates=True)
        self.cols = ['Pm', 'Vm', 'Im', 'Voc', 'Isc', 'Voc_coeff', 'Isc_coeff', 'Rs', 'Iph', 'I0', 'Rp', 'n']
        # self.cols = ['Pm', 'Vm', 'Im', 'Voc', 'Isc', 'Voc_coeff', 'Isc_coeff', 'Iph']

        self.col_inputs = self.cols[0:7]
        self.col_predicted = [x + '_predicted' for x in self.cols[7:]]
        values = dataset.values[:,0:7]
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaled = scalerInputs.transform(values)
        return scaled, scalerInputs,scalerOutputs


    def plotting_save_experiment_data(self, model, inputs, y_predicted):
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

        df = DataFrame(list(y_predicted), columns=self.col_predicted)
        df.to_excel(writer, 'predicted')
        all = concatenate((inputs, y_predicted), axis=1)
        df = DataFrame(all, columns=self.col_inputs  + self.col_predicted)
        df.to_excel(writer, 'The Data')

        writer.save()
        writer.close()

        # save to database
        if self.parameters["save_to_database"]:
            db = mysql.connector.connect(host="localhost", user="root", passwd="mansoura", db="dr_mosaad_data3")
            # prepare a cursor object using cursor() method
            cursor = db.cursor()

            sql = """INSERT INTO experiments (experiment_ID, 
                       outputIndex,  Model_summary,comment) VALUES ('{}',{}, '{}','{}')""" \
                .format(self.parameters["ID"],
                        self.outputIndex,
                        model.summary(),
                        self.parameters["comment"])
            try:
                cursor.execute(sql)
                db.commit()
                print("** saved to database ID: "+self.parameters["ID"])
            except TypeError as e:
                print(e)
                db.rollback()
                print(sql)
                print("** Error saving to database")

            db.close()

    def Predict_From_model_assesment(self, data,  scalerInputs,scalerOutputs, model):
        inputs = numpy.array(data)[:, :7]
        y_test_predicted = model.predict([inputs ,inputs ,inputs ,inputs ,inputs])

        uu = numpy.array(y_test_predicted).transpose()
        yy = uu.reshape((-1,5 ))

        # invert scaling for actual

        # invert scaling for forecast
        inputs= scalerInputs.inverse_transform(inputs)
        yy = scalerOutputs.inverse_transform(yy)
        Predicted_data = concatenate((inputs, yy), axis=1)



        y_test_predicted = Predicted_data[:, 7:]
        self.plotting_save_experiment_data(model, Predicted_data[:,0:7], y_test_predicted)

    def CreateModel(self):
        list = ['20200726221917', '20200820003813', '20200726114513', '20200726051814', '20200726064251']
        # list = ['20200820003813']
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
        plot_model(model, to_file='E:\\kkkk.png')
        return model

    def start_experiment(self):

        model = self.CreateModel()
        data, scalerInputs,scalerOutputs = self.load_prepare_data_assesment()
        self.Predict_From_model_assesment(data,  scalerInputs,scalerOutputs, model)
        return self.MAPE


f = TestModel(outputIndex=-1,comment="assesment2")
f.start_experiment()


# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/#comment-546517
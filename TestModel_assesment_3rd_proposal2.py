import datetime
import json
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
from keras.models import load_model
from keras.utils import plot_model


mainDir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located




class TestModel:
    parameters = dict()
    id = ""
    MAPE = -1
    outputIndex = -1
    ANN_arch_list = []

    def __init__(self,
                 save_to_database=False,
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
        self.parameters["ID"] = self.id +'_testmodel_'

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
        dataset = read_csv(os.path.join(mainDir,'TheData\\data_without_outliers.csv'), header=0, index_col=0, parse_dates=True)
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


        # scalerInputs= load(open(mainDir + '\\Models\\' + '20200820003813' + '_scalerInputs.pkl', 'rb'))
        # scalerOutputs= load(open(mainDir + '\\Models\\' + '20200820003813' + '_scalerOutputs.pkl', 'rb'))

        # load dataset  PVs for assessments.csv
        # dataset = read_csv(os.path.join(mainDir,'TheData\\data_new_100_modules_30_Oct.csv'), header=0, index_col=0, parse_dates=True)
        dataset = read_csv(os.path.join(mainDir,'TheData\\data_new_100_modules_30_Oct.csv'), header=0, index_col=0, parse_dates=True)

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
        filepath = os.path.join(mainDir, 'results\\experimentOutput\\' + self.parameters["ID"] + 'model_fig.png')   # Joins the script directory with "dataset_RV2"

        plot_model(model, to_file=filepath   , show_shapes=True, show_layer_names=True,
                   rankdir='TB', expand_nested=False, dpi=300)



        # print(model.summary())
        # save data to excel file
        filepath = os.path.join(mainDir, 'results\\experimentOutput\\' + self.parameters["ID"] + 'results.xlsx')   # Joins the script directory with "dataset_RV2"
        writer = ExcelWriter(filepath)
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

        writer.close()

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

            sql = """INSERT INTO assestment_results (experiment_ID, 
                       outputIndex,  Model_summary,comments) VALUES ('{}',{}, '{}','{}')""" \
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

    def Predict_From_model_assesment(self, data,  scalerInputs,scalerOutputs, models):
        inputs = numpy.array(data)[:, :7]
        y_test_predicted=[]
        new_models=[]
        for m in models:
            # create a new input layer that accepts input of the actual shape
            actual_input_shape = inputs.shape
            new_input_layer = keras.layers.Input(shape=actual_input_shape)

            # create a new model with the modified input layer and the same remaining layers as the original model
            model_input = m.layers[1](new_input_layer)  # assume that the first layer is not an input layer
            for layer in m.layers[2:]:
                model_input = layer(model_input)
            new_model = keras.models.Model(inputs=new_input_layer, outputs=model_input)
            new_models.append(new_model)
            y= new_model.predict(np.expand_dims(inputs, axis=0))
            y_test_predicted.append(y)
            print ("done")

        model = keras.Model(inputs=[f.input for f in new_models], outputs=[f.output for f in new_models])


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
        list = ['20231104135032', '20231104184805', '20231104234439', '20231105002828', '20231105015201']

        Dir = os.path.join(mainDir, 'results\\Models')   # Joins the script directory with "dataset_RV2"
        models = []
        for f in list:
            fn = f + '_best_model.h5'
            funModel = load_model(os.path.join(Dir, fn))
            models.append(funModel)
        return models

    def start_experiment(self):

        models = self.CreateModel()
        data, scalerInputs,scalerOutputs = self.load_prepare_data_assesment()
        self.Predict_From_model_assesment(data,  scalerInputs,scalerOutputs, models)
        return self.MAPE


f = TestModel(outputIndex=-1,comment="assesment3_proposal2",save_to_database=True)
f.start_experiment()




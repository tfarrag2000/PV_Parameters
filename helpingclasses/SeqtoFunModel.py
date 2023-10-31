from tensorflow import keras
from tensorflow.python.keras.models import load_model

list=['20200717100010']
        Dir = os.path.dirname(os.path.abspath(__file__), 'Results\\Models\\')
for f in list:
    fn=f + '_best_model.h5'
    seqModel = load_model(Dir + fn)

    nfn=f + '_functional_best_model.h5'

    input_layer = keras.Input( shape=(7,))
    prev_layer = input_layer
    for layer in seqModel.layers:
        prev_layer = layer(prev_layer)
    funcModel = keras.Model([input_layer], [prev_layer])
    funcModel.save(Dir + nfn)
    print(funcModel.summary())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
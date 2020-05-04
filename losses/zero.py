import tensorflow.keras.backend as K

def zero_loss(yTrue,yPred):
    return 0*(K.sum(yTrue - yPred))
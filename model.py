import tensorflow

def get_model(num_outputs = 10, num_feats = [32, 64], input_shape = (28,28,1)):
    """ Returns a tf.keras.Model that computes a the probabilities for each one of the classes (num_outputs)
        Inputs:
            num_outputs: number of classes
            num_feats: a list containing the number of output dims (feat. maps) for the convolutational layers sequentially
            input_shape : shape of the input data, as we are using the MNIST images (28,28,1)
         
        Returns:
            tf.keras.Model that is later used to compute the class probabilities for an input image
    """
    
    inputs = tensorflow.keras.Input(shape=input_shape, name = 'input_1')
    x = inputs
    
    for num in num_feats:
        x = tensorflow.keras.layers.Conv2D(num, (3,3), padding = "same")(x)
        x = tensorflow.keras.layers.LeakyReLU(0.2)(x)
        x = tensorflow.keras.layers.MaxPool2D((2,2))(x)
    
    feats = tensorflow.keras.layers.GlobalAveragePooling2D(name = 'gap')(x)
    
    outs = tensorflow.keras.layers.Dense(num_outputs, activation = 'linear', name = 'cls', use_bias = False)(feats)
    outs = tensorflow.keras.layers.Activation('softmax')(outs)
    # decoupling models
    model = tensorflow.keras.Model(inputs=inputs, outputs=outs, name = 'mnist_model')

    return model


def get_cls_retrain_model(model, num_outputs = 10, input_shape = (28,28,1)):
    
    """ Returns a tf.keras.Model with the feature extraction freezed, while the classifier is trained
        Inputs:
            model: pre-trained model
            num_outputs: number of classes
            input_shape : shape of the input data, as we are using the MNIST images (28,28,1)
         
        Returns:
            tf.keras.Model that is later used to re-train the classifier layer that is initialized randomly
    """
    
    inputs = tensorflow.keras.Input(shape=input_shape, name = 'input_1')
    
    feat_model = tensorflow.keras.Model(inputs= model.inputs, outputs= model.get_layer('gap').output)
    feat_model.trainable = False

    feats = feat_model(inputs)
    outs = tensorflow.keras.layers.Dense(num_outputs, activation = 'softmax', name = 'cls')(feats)
    
    cls_retrain_model = tensorflow.keras.Model(inputs=inputs, outputs=outs, name = 'mnist_retrain')

    return cls_retrain_model

# Note: it is not possible to use a simple Lambda Layer because the weights are not considered during training 
class ScaleLayer(tensorflow.keras.layers.Layer):
    def __init__(self, num_outputs = 10):
        super(ScaleLayer, self).__init__()
        self.scale = tensorflow.Variable(tensorflow.ones(num_outputs), trainable= True)

    def call(self, inputs):
        return inputs * self.scale
    
    
def get_lws_model(model, num_outputs = 10, input_shape = (28,28,1)):
    
    """ Returns a tf.keras.Model with all the model freezed except a scaling layer (f in (1,c) ) that multiplies the classifier (learnable weight scaling ~ f*w*x)
        Inputs:
            model: pre-trained model
            num_outputs: number of classes
            input_shape : shape of the input data, as we are using the MNIST images (28,28,1)
         
        Returns:
            tf.keras.Model that is implemented using the learnable weight scaling (lws) strategy
    """
    
    inputs = tensorflow.keras.Input(shape= input_shape, name = 'input_1')
    cls_model = tensorflow.keras.Model(inputs= model.inputs, outputs= model.get_layer('cls').output)
    
    cls_model.trainable = False
    outs = cls_model(inputs)
    
    outs =  ScaleLayer(num_outputs)(outs)
    
    outs = tensorflow.keras.layers.Activation('softmax')(outs)
    
    lws_model = tensorflow.keras.Model(inputs=inputs, outputs=outs, name = 'mnist_lws')

    return lws_model
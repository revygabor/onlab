import numpy as np
import keras.backend as K
from keras.layers import Layer
from matplotlib import pyplot as plt
from keras.models import load_model, Model

def create_inference_model(name):
    model = load_model(name)
    input_layer = model.layers[0].input
    output_layer = model.layers[-1].output
    new_output_layer = Argmax()(output_layer)

    model = Model(inputs=input_layer, outputs=new_output_layer)
    # model.summary()
    return model


class Argmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return K.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def plot_confusion_matrix(conf_mtx, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    if not title:
            title = 'Normalized confusion matrix'

    print(conf_mtx)

    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(conf_mtx, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_mtx.shape[1]),
           yticks=np.arange(conf_mtx.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = conf_mtx.max() / 2.
    for i in range(conf_mtx.shape[0]):
        for j in range(conf_mtx.shape[1]):
            ax.text(j, i, format(conf_mtx[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_mtx[i, j] > thresh else "black")
    # fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    create_inference_model('model0.h5')
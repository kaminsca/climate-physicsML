import tensorflow as tf
from tensorflow import keras
from config import vars_mli, vars_mlo

def build_base_model():
    initializer = tf.keras.initializers.GlorotUniform()
    # input_length = initially 2 * 60 + 5 for including atmospheric levels, now just length of input vars
    input_length = len(vars_mli)
    output_length_relu = len(vars_mlo)
    input_layer = keras.layers.Input(shape=(input_length,), name='input')
    hidden_0 = keras.layers.Dense(768, activation='relu', kernel_initializer=initializer)(input_layer)
    hidden_1 = keras.layers.Dense(640, activation='relu', kernel_initializer=initializer)(hidden_0)
    hidden_2 = keras.layers.Dense(512, activation='relu', kernel_initializer=initializer)(hidden_1)
    hidden_3 = keras.layers.Dense(640, activation='relu', kernel_initializer=initializer)(hidden_2)
    hidden_4 = keras.layers.Dense(640, activation='relu', kernel_initializer=initializer)(hidden_3)
    output_pre = keras.layers.Dense(output_length_relu, activation='elu', kernel_initializer=initializer)(hidden_4)
    output_relu = keras.layers.Dense(output_length_relu, activation='relu', kernel_initializer=initializer)(output_pre)

    model = keras.Model(input_layer, output_relu, name='Emulator')
    model.summary()
    # print dimensions of input and output layers of model and exit
    print("Model dimensions of input and output layers:")
    print(model.input.shape)
    print(model.output.shape)
    return model

def load_model(model_path):
    custom_objects = {'CustomModel': CustomModel}
    model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    return model

@keras.utils.register_keras_serializable()
class CustomModel(tf.keras.Model):
    def __init__(self, base_model, initial_lambdas, constant_lambdas, exclude_these_losses):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.constant_lambdas = constant_lambdas  
        self.exclude_these_losses = exclude_these_losses

        # Initialize trainable lambda parameters
        self.lambda_mass_param = self.add_weight(
            name='lambda_mass_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_lambdas['mass']),
            trainable=('mass' not in constant_lambdas and 'mass' not in exclude_these_losses),
        )
        self.lambda_radiation_param = self.add_weight(
            name='lambda_radiation_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_lambdas['radiation']),
            trainable=('radiation' not in constant_lambdas and 'radiation' not in exclude_these_losses),
        )
        self.lambda_nonneg_param = self.add_weight(
            name='lambda_nonneg_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_lambdas['nonneg']),
            trainable=('nonneg' not in constant_lambdas and 'nonneg' not in exclude_these_losses),
        )

    def call(self, inputs):
        return self.base_model(inputs)

    def get_config(self):
        config = super(CustomModel, self).get_config()
        config.update({
            'base_model': keras.utils.serialize_keras_object(self.base_model),
            'initial_lambdas': {
                'mass': self.lambda_mass_param.numpy(),
                'radiation': self.lambda_radiation_param.numpy(),
                'nonneg': self.lambda_nonneg_param.numpy(),
            },
            'constant_lambdas': self.constant_lambdas,
            'exclude_these_losses': self.exclude_these_losses,
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_model = keras.utils.deserialize_keras_object(config['base_model'])
        initial_lambdas = config['initial_lambdas']
        constant_lambdas = config.get('constant_lambdas', [])
        exclude_these_losses = config.get('exclude_these_losses', [])
        return cls(base_model=base_model, initial_lambdas=initial_lambdas,
                   constant_lambdas=constant_lambdas,
                   exclude_these_losses=exclude_these_losses)



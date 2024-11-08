import torch
import tensorflow as tf

# Rin, Rout, pbuf_LHFLX, and pbuf_SHFLX are known and can be passed to the loss function

# Loss function combining MSE and energy conservation loss
def total_loss(Rin, Rout, pbuf_LHFLX, pbuf_SHFLX, ec_weight):
    def loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # MSE loss
        energy_loss = tf.reduce_mean(tf.abs(Rin - (Rout + pbuf_LHFLX + pbuf_SHFLX)))  # Energy conservation loss
        return mse_loss + ec_weight * energy_loss  # You can adjust the weights here if necessary
    return loss
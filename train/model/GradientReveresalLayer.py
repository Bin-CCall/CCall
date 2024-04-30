import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.custom_gradient
def gradient_reversal(x,alpha=1):
	def grad(dy):
		return -dy * alpha, None
	return x, grad

class GradientReversalLayer(Layer):

	def __init__(self,**kwargs):
		super(GradientReversalLayer,self).__init__(kwargs)

	def call(self, x,alpha=1.0):
		return gradient_reversal(x,alpha)

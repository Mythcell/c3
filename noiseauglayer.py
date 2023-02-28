import tensorflow as tf
from tensorflow.keras import layers

class NoiseAugmentation(layers.Layer):

    def __init__(self, mean_range=(-0.02,0.1), stddev_range=(1e-6,0.3), **kwargs):
        super().__init__(**kwargs)
        self.mean_range = mean_range
        self.stddev_range = stddev_range

    def call(self, x, training=True):
        
        noise = tf.random.normal(
            shape=(100,100,1),
            mean = tf.random.uniform(
                shape=[1],
                minval=self.mean_range[0],
                maxval=self.mean_range[1]
            ),
            stddev = tf.random.uniform(
                shape=[1],
                minval=self.stddev_range[0],
                maxval=self.stddev_range[1]
            )
        )
        x = tf.add(x, noise)
        return tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
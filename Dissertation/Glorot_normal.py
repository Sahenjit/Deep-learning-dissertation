import tensorflow as tf

tf.keras.initializers.GlorotNormal(seed=None)

initializer=tf.keras.initializers.GlorotNormal()
print(initializer)
value=initializer(shape=(2,2))

initializer=tf.keras.initializers.GlorotNormal()
layer=tf.keras.layers.Dense(3, kernel_initializer=initializer)
print(layer)


import tensorflow as tf

# 1. Verificar se o TensorFlow está enxergando a GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs disponíveis: {gpus}")

# 2. Forçar a execução de um bloco específico na GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.1, 0.2]])
    c = tf.matmul(a, b)
    print(c)
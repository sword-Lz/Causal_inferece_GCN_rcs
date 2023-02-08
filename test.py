import tensorflow as tf
dataset_location = 'data/amazon-book/train.txt'
record_defaults = [[1], [1], [0.]] # Sets the type of the resulting tensors and default values
# Dataset is in the format - UserID ProductID Rating
dataset = tf.data.TextLineDataset(dataset_location).map(lambda line: tf.io.decode_csv(line, record_defaults=record_defaults))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(1024)
dataset = dataset.repeat(10)
print(list(dataset.as_numpy_iterator()))
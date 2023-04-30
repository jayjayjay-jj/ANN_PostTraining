import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def load_dataset():
    df = pd.read_csv('arrival_from_australia.csv', index_col='Date')
    return df

dataset = load_dataset()

input_num = 1
output_num = 1
context_unit = 3 # Hidden Layer in RNN

# How many previous datas needed in predicting the next prediction
time_seq = 3

test_size = 30

train_data = dataset[:int(len(dataset)*0.7)]
test_data = dataset[len(train_data):]

minMaxScaler = MinMaxScaler()
norm_train_data = minMaxScaler.fit_transform(train_data)
norm_test_data = minMaxScaler.fit_transform(test_data)

# Input to Hidden
cell = tf.nn.rnn_cell.BasicRNNCell(context_unit, activation = tf.nn.relu)

# Hidden to Output
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = output_num, activation = tf.nn.relu)

feature_tensorflow = tf.placeholder(tf.float32, [None, time_seq, input_num])
target_tensorflow = tf.placeholder(tf.float32, [None, time_seq, output_num])

output, _ = tf.nn.dynamic_rnn(cell, feature_tensorflow, dtype=tf.float32)

error = tf.reduce_mean(0.5 * (target_tensorflow - output) ** 2)

learning_rate = 0.1
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

epoch = 1000
batch_size = 3 # Each train will use 3 datas

def next_bacth(dataset, batch_size):
    x_batch = np.zeros([batch_size, time_seq, input_num])
    y_batch = np.zeros([batch_size, time_seq, output_num])

    for i in range(batch_size):
        start = np.random.randint(0, len(dataset) - time_seq)
        x_batch[i] = dataset[start:start+time_seq]
        y_batch[i] = dataset[start+1:start+1+time_seq]

    return x_batch, y_batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, epoch + 1):
        x_batch, y_batch = next_bacth(norm_train_data, batch_size)

        train_data = {
            feature_tensorflow : x_batch,
            target_tensorflow : y_batch
        }

        sess.run(optimizer, feed_dict = train_data)

        if(i % 50 == 0):
            loss = sess.run(error, feed_dict = train_data)
            print(f'Loss : {loss}')

    seed_data = list(norm_test_data)

    for i in range(len(test_data)):
        x_batch = np.array(seed_data[-time_seq:]).reshape([1, time_seq, input_num])

        test_dict = {
            feature_tensorflow : x_batch
        }

        predict = sess.run(output, feed_dict=test_dict)
        seed_data.append(predict[0, -1, 0])

# From MinMaxScaler, inverse to real value
result = minMaxScaler.inverse_transform(np.array(seed_data[-len(test_data):]).reshape([len(test_data), 1]))

test_data['prediction'] = result
test_data.plot()
plt.show()


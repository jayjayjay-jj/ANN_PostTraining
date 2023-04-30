import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

def load_dataSet():
    df = pd.read_csv('foodDiet.csv')

    feature = df[['gender', 'age', 'height', 'pre-weight', 'weight6weeks']]

    target = df[['diet']]

    return feature, target

feature, target = load_dataSet()

# Normalize the big number
minMaxScaler = MinMaxScaler()
feature = minMaxScaler.fit_transform(feature)

ordinalEncoder = OrdinalEncoder()
feature = ordinalEncoder.fit_transform(feature)

oneHotEncoder = OneHotEncoder(sparse=False)
target = oneHotEncoder.fit_transform(target)

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

layers = {
    # Layer : Node
    'input': 5,
    'hidden': 5,  # Hidden Node is optional
    'output': 3
}

weight = {
    'input_to_hidden': tf.Variable(tf.random_normal([layers['input'], layers['hidden']])),
    'hidden_to_output': tf.Variable(tf.random_normal([layers['hidden'], layers['output']]))
}

bias = {
    'hidden': tf.Variable(tf.random_normal([layers['hidden']])),
    'output': tf.Variable(tf.random_normal([layers['output']]))
}

# Feature and Output both must be in tensorflow format so that it can be use using tensorflow
feature_tensorflow = tf.placeholder(tf.float32, [None, layers['input']])
target_tensorflow = tf.placeholder(tf.float32, [None, layers['output']])

def feed_forward():
    y1 = tf.matmul(feature_tensorflow, weight['input_to_hidden']) + bias['hidden']
    y1Act = tf.nn.sigmoid(y1)

    y2 = tf.matmul(y1Act, weight['hidden_to_output']) + bias['output']
    y2Act = tf.nn.sigmoid(y2)

    return y2Act

learning_rate = 0.1
epoch = 1000

output = feed_forward()

# Error is based on the formula
error = tf.reduce_mean((0.5 * target_tensorflow - output) ** 2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

with tf.Session() as sess:
    # Get into the Session, and activate the tf.Variable (Global Variable)
    # 1. Fill the data into both feature_tensorflow and target_tensorflow for the train data
    # 2. Run the optimizer with value from train_dict, optimizer will call the BNN and check error based on the error as a tensorflow object based on the formula above

    # feed_forward + back_tracking
    sess.run(tf.global_variables_initializer())

    for i in range(1, epoch + 1):
        train_dict = {
            feature_tensorflow : x_train,
            target_tensorflow : y_train
        }

        sess.run(optimizer, feed_dict = train_dict)
        loss = sess.run(error, feed_dict = train_dict)

        # Print the loss every 50 epoch
        if(i % 50 == 0):
            print(f'Loss = {loss}')

    test_dict = {
        feature_tensorflow : x_test,
        target_tensorflow : y_test
    }
    
    # Search the accuracy
    matches = tf.equal(tf.argmax(target_tensorflow, axis = 1), tf.argmax(output, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    print(f'Accuracy = {sess.run(accuracy, feed_dict = test_dict) * 100}%')
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations
from scipy.spatial.distance import cosine, euclidean
import face_data_generate


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    def next_train_batch():
        for feature, label in zip(features, labels):
            yield (feature, label)

    dataset = tf.data.Dataset.from_generator(
        next_train_batch, (tf.float32, tf.int32), (tf.TensorShape([2, 55, 47, 3]), tf.TensorShape([2])))
    # batch the examples.
    dataset = dataset.repeat().shuffle(1024).batch(batch_size)
    # Return the iterator
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def DeepID_model_fn(features, labels, mode, params):
    '''Model Function for DeepID'''
    # Input layer
    features_i = tf.map_fn(lambda x: x[0], features)
    features_j = tf.map_fn(lambda x: x[1], features)
    labels_i = tf.map_fn(lambda x: x[0], labels)
    labels_j = tf.map_fn(lambda x: x[1], labels)

    def cnn_model(inputs, params):
        # Hidden layer
        conv_1_outputs = layers.Conv2D(
            filters=20,
            kernel_size=[4, 4],
            padding='valid',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1))(inputs)

        pool_1_outputs = layers.MaxPool2D(
            pool_size=[2, 2],
            strides=[2, 2],
            padding='valid')(conv_1_outputs)

        conv_2_outputs = layers.Conv2D(
            filters=40,
            kernel_size=[3, 3],
            padding='valid',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1))(pool_1_outputs)

        pool_2_outputs = layers.MaxPool2D(
            pool_size=[2, 2],
            strides=[2, 2],
            padding='valid')(conv_2_outputs)

        conv_3_outputs = layers.Conv2D(
            filters=60,
            kernel_size=[3, 3],
            padding='valid',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1))(pool_2_outputs)

        pool_3_outputs = layers.MaxPool2D(
            pool_size=[2, 2],
            strides=[2, 2],
            padding='valid')(conv_3_outputs)

        conv_4_outputs = layers.Conv2D(
            filters=80,
            kernel_size=[2, 2],
            padding='valid',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1))(pool_3_outputs)

        # DeepID layer
        deepid = layers.Dense(
            units=160,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1))(tf.concat([tf.reshape(pool_3_outputs, [-1, 5 * 4 * 60]), tf.reshape(conv_4_outputs, [-1, 4 * 3 * 80])], 1))

        batch_normed = layers.BatchNormalization()(deepid)
        
        normed_deepid = activations.relu(batch_normed)
        
        # logits layer
        logits = layers.Dense(
            units=params['class_num'],
            activation=None,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1))(normed_deepid)

        return normed_deepid, logits

    # Compute deepid
    deepid_i, logits_i = cnn_model(features_i, params)
    deepid_j, logits_j = cnn_model(features_j, params)

    predicted_class_i = tf.argmax(logits_i, 1)
    predicted_class_j = tf.argmax(logits_j, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "DeepID_i": deepid_i,
            "DeepID_j": deepid_j
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Compute loss.
    ident_i = tf.losses.sparse_softmax_cross_entropy(
        labels=labels_i, logits=logits_i)
    ident_j = tf.losses.sparse_softmax_cross_entropy(
        labels=labels_j, logits=logits_j)

    verif = tf.reduce_sum(tf.map_fn(
        lambda x: 0.5 * tf.square(x[1]) if x[0] == 0 else 0.5 *
        tf.square(tf.maximum(0.0, 0.1 - x[1])),
        tf.transpose(tf.concat([[tf.cast(labels_i - labels_j, tf.float32)], [tf.norm(deepid_i - deepid_j, axis=1)]], axis=0))))

    loss = ident_i + ident_j + 0.1 * verif

    # Compute evaluation metrics.
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = {
            'accuracy': tf.metrics.accuracy(
                labels_i=tf.map_fn(lambda x: x[0], labels),
                labels_j=tf.map_fn(lambda x: x[1], labels),
                predictions_i=predicted_class_i,
                predictions_j=predicted_class_j)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=accuracy)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.RMSPropOptimizer(1e-4)  # AdamOptimizer(1e-4)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def DeepID_model_estimator():

    session_config = tf.ConfigProto(
        log_device_placement=True,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7

    run_config = tf.estimator.RunConfig().replace(session_config=session_config)

    # Build DeepID model
    DeepID_model = tf.estimator.Estimator(
        model_fn=DeepID_model_fn,
        model_dir='../models/',
        config=run_config,
        params={
            'class_num': 8192
        })

    return DeepID_model


def train_and_evaluate():

    # Fetch the data.
    (train_data, train_label), (eval_data,
                                eval_label) = face_data_generate.load_train_data()

    DeepID_model = DeepID_model_estimator()
    # Train the model
    DeepID_model.train(
        input_fn=lambda: train_input_fn(
            train_data, train_label, 512),
        steps=50000)

    # Evaluate the model
    eval_result = DeepID_model.evaluate(
        input_fn=lambda: eval_input_fn(eval_data, eval_label, 512))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def generate_deepid():
    pass


def main(argv=None):
    train_and_evaluate()
    # generate_deepid()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

import os
import time
import tensorflow as tf
from model_vgg import Model_Vgg19


class DataSet:
    '''
    queue runner 사용
    '''
    def __init__(self, sess, dataset, batch_size=16):
        self._sess = sess
        self._dataset = dataset
        self._batch_size = batch_size

        input_queue = tf.train.slice_input_producer(self._dataset)

        image_files = tf.read_file(input_queue[0])

        image_list = tf.image.decode_jpeg(image_files, channels=3)
        image_list.set_shape([150, 150, 3])

        if len(dataset) == 1:
            self._batch = tf.train.batch([image_list], batch_size=batch_size)
        elif len(dataset) == 2:
            self._batch = tf.train.batch([image_list, input_queue[1]], batch_size=batch_size)

    def countData(self):
        return len(self._dataset[0])

    def countBatch(self):
        return int(self.countData() / self._batch_size) + 1

    def start(self):
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(sess=self._sess, coord=self._coord)

    def init(self):
        print()

    def stop(self):
        self._coord.request_stop()
        self._coord.join(self._threads)

    def next_batch(self):
        return self._sess.run(self._batch)


def _parse_function (filename, label):
    #print('filename: ', filename)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    #image_resized = tf.image.resize_images(image_decoded, [150, 150])
    image_decoded.set_shape([150, 150, 3])
    return image_decoded, label


def input_fn(filenames, labels, batch_size = 16):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    dataset = dataset.map(_parse_function)

    return dataset

import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

class DataSet2:
    '''
    tf.dataset 사용
    '''
    def __init__(self, sess, dataset, batch_size=16):
        self._sess = sess
        self._dataset = dataset
        self._batch_size = batch_size

        tfdataset = tf.data.Dataset.from_tensor_slices((dataset[0], dataset[1]))
        tfdataset = tfdataset.shuffle(1000).repeat().batch(batch_size)
        #tfdataset = tfdataset.map(_parse_function)
        tfdataset = tfdataset.map(lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))

        self.iterator = tfdataset.make_one_shot_iterator()
        self.next_example, self.next_label = self.iterator.get_next()


    def countData(self):
        return len(self._dataset[0])

    def countBatch(self):
        return int(self.countData() / self._batch_size) + 1

    def start(self):
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(sess=self._sess, coord=self._coord)

    def init(self):
        _ = self._sess.run(self.iterator._initializer)

    def stop(self):
        print()

    def next_batch(self):
        return self._sess.run([self.next_example, self.next_label])

def train (model, train_dataset, valid_dataset, learning_rate, epochs):
    sess = model.sess
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    start_time = time.time()
    train_count = train_dataset.countData()
    total_batch_train = train_dataset.countBatch()
    print('train_count: ', train_count, 'total_batch_train: ', total_batch_train)
    valid_count = valid_dataset.countData()
    total_batch_valid = valid_dataset.countBatch()
    print('valid_count: ', valid_count, 'total_batch_valid: ', total_batch_valid)

    # usage
    # tensorboard --logdir=./log
    train_writer = tf.summary.FileWriter("./log/train")
    valid_writer = tf.summary.FileWriter("./log/valid")
    train_writer.add_graph(sess.graph)
    train_global_step = 0
    valid_global_step = 0

    #train_dataset.start()
    #valid_dataset.start()
    print('Learning started. It takes sometime.')
    for epoch in range(epochs):
        avg_cost_train = 0
        avg_cost_valid = 0
        accuracy_train = 0
        accuracy_valid = 0

        correct_count_train = 0
        correct_count_valid = 0

        start_time_epoch = time.time()

        train_dataset.init()
        for i in range(total_batch_train):
            batch_x_image, batch_y = train_dataset.next_batch()

            s, c, _ = model.train(batch_x_image, batch_y, learning_rate)
            avg_cost_train += c / total_batch_train
            correct_count_train += model.countCorrect(batch_x_image, batch_y)

            train_global_step += 1
            if train_global_step % 1000:
                train_writer.add_summary(s, global_step=train_global_step)

        accuracy_train = correct_count_train / train_count

        valid_dataset.init()
        for i in range(total_batch_valid):
            batch_x_image, batch_y = valid_dataset.next_batch ()
            c = model.get_cost(batch_x_image, batch_y)
            avg_cost_valid += c / total_batch_valid

            correct_count_valid += model.countCorrect(batch_x_image, batch_y)
            #s = model.summary(batch_x_image, batch_y)
            #valid_writer.add_summary(s, global_step=valid_global_step)
            #valid_global_step += 1

        accuracy_valid = correct_count_valid / valid_count

        print('Epoch:', '%04d' % (epoch + 1)
            , 'train [cost: ', '{:.9f}'.format(avg_cost_train), ', acc: %.4f]' % accuracy_train
            , 'valid [cost: ', '{:.9f}'.format(avg_cost_valid), ', acc: %.4f]' % accuracy_valid
            , " %.2f seconds" % (time.time() - start_time_epoch))

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join('./model/', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch + 1)
            print('Save checkpoint  : %s' %(checkpoint_path))

    print('Learning Finished!')
    print("--- %.2f seconds ---" %(time.time() - start_time))

    #train_dataset.stop()
    #valid_dataset.stop()

def eval(model, eval_dataset, epochs):
    sess = model.sess
    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    eval_count = eval_dataset.countData()
    total_batch_eval = eval_dataset.countBatch()
    print('eval_count: ', eval_count, 'total_batch_eval: ', total_batch_eval)

    #eval_dataset.start()
    eval_dataset.init()
    correct_count = 0
    for i in range(total_batch_eval):
        batch_x_image, batch_y = eval_dataset.next_batch()
        correct_count += model.countCorrect(batch_x_image, batch_y)

    #accuracy = correct_count / eval_count
    accuracy = correct_count / (total_batch_eval * eval_dataset._batch_size)

    print("eval-accuracy: %.4f" % accuracy)
    print("--- %.2f seconds ---" %(time.time() - start_time))
    #eval_dataset.stop()

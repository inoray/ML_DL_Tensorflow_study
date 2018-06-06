import tensorflow as tf
import time
import os

def train (model, train_dataset, valid_dataset, learning_rate=0.0001, epochs=16, tensorboard_logdir="./log"):
    """학습 수행

    args:
        train_dataset: 학습용 데이터. data.Dataset_image
        valid_dataset: 평가용 데이터. data.Dataset_image
        learning_rate: learning rate
        epochs: epochs
    """
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
    train_writer = tf.summary.FileWriter(tensorboard_logdir + "/train")
    valid_writer = tf.summary.FileWriter(tensorboard_logdir + "/valid")
    train_writer.add_graph(sess.graph)
    train_global_step = 0
    max_valid_acc = 0.

    print('Learning started. It takes sometime.')
    for epoch in range(epochs):
        avg_cost_train = 0
        avg_cost_valid = 0
        accuracy_train = 0
        accuracy_valid = 0

        start_time_epoch = time.time()

        train_dataset.init_iterator()
        for i in range(total_batch_train):
            batch_x_image, batch_y = train_dataset.next_batch()
            s, a, c, _ = model.train(batch_x_image, batch_y, learning_rate)
            accuracy_train += a / total_batch_train
            avg_cost_train += c / total_batch_train

            train_global_step += 1
            if train_global_step % 1000 == 0:
                train_writer.add_summary(s, global_step=train_global_step)

        valid_dataset.init_iterator()
        for i in range(total_batch_valid):
            batch_x_image, batch_y = valid_dataset.next_batch ()
            a, c = model.eval(batch_x_image, batch_y)
            accuracy_valid += a / total_batch_valid
            avg_cost_valid += c / total_batch_valid

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="cost", simple_value=avg_cost_valid),
            tf.Summary.Value(tag="accuracy", simple_value=accuracy_valid)])
        valid_writer.add_summary(summary, train_global_step)

        print('Epoch:', '%04d' % (epoch + 1)
            , 'train [cost: ', '{:.9f}'.format(avg_cost_train), ', acc: %.4f]' % accuracy_train
            , 'valid [cost: ', '{:.9f}'.format(avg_cost_valid), ', acc: %.4f]' % accuracy_valid
            , " %.2f seconds" % (time.time() - start_time_epoch))

        #if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
        if accuracy_valid > max_valid_acc:
            max_valid_acc = accuracy_valid
            checkpoint_path = os.path.join('./model/', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch + 1)
            print('Save checkpoint: %s' %(checkpoint_path))

    print('Learning Finished!')
    print("--- %.2f seconds ---" %(time.time() - start_time))


def eval(model, eval_dataset):
    """evaluation 수행

    args:
        eval_dataset: 평가용 데이터. data.Dataset_image
    """
    start_time = time.time()
    avg_cost = 0
    accuracy = 0
    eval_count = eval_dataset.countData()
    total_batch_eval = eval_dataset.countBatch()
    print('eval_count: ', eval_count, ', total_batch_eval: ', total_batch_eval)

    eval_dataset.init_iterator()
    for i in range(total_batch_eval):
        batch_x_image, batch_y = eval_dataset.next_batch()
        a, c = model.eval(batch_x_image, batch_y)
        accuracy += a / total_batch_eval
        avg_cost += c / total_batch_eval

    print('Evaluation Finished!')
    print("cost: ", "{:.9f}".format(avg_cost), ", accuracy: %.4f" % accuracy)
    print("--- %.2f seconds ---" %(time.time() - start_time))

    return avg_cost, accuracy


def predict(model, test_dataset):
    """predict 수행

    args:
        test_dataset: 테스트용 데이터. data.Dataset_image
    """
    start_time = time.time()

    total_batch = test_dataset.countBatch()

    test_dataset.init_iterator()
    predict_list = []
    probability_list = []
    for i in range(total_batch):
        batch_x_image = test_dataset.next_batch()
        batch_predict, batch_prob = model.predict(batch_x_image)
        predict_list.extend(batch_predict)
        probability_list.extend(batch_prob)

    print("--- %.2f seconds ---" %(time.time() - start_time))
    return predict_list, probability_list


if __name__ == "__main__":

    """
    train, eval 예제
    """
    import os
    from sklearn.model_selection import train_test_split
    import data
    import model
    import numpy as np

    # 학습데이터 준비
    DATA_DIR = "../data/"
    TRAIN_DIR = DATA_DIR + "train_resize/"

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
    train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

    print("train_dogs: ", len(train_dogs))
    print("train_cats: ", len(train_cats))

    train_dogs = train_dogs[:20]
    train_cats = train_cats[:20]

    train_dog_cat = train_dogs + train_cats

    label_dog = [0 for i in range(len(train_dogs))]
    label_cat = [1 for i in range(len(train_cats))]
    label = label_dog + label_cat
    label_one_hot = np.eye(2)[label]

    # 학습, 평가 데이터 분리
    x_train, x_valid, y_train, y_valid = train_test_split (
        train_dog_cat, label_one_hot, test_size=0.3, random_state=42)

    print("train: ", len(x_train))
    print("valid: ", len(x_valid))

    learning_rate = 0.0001
    epochs = 2
    batch_size = 10

    # dataset 생성
    train_dataset = data.Dataset_image([x_train, y_train], batch_size = batch_size)
    valid_dataset = data.Dataset_image([x_valid, y_valid], batch_size = batch_size)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))
    model = model.Vgg(sess, "model")
    model.build_net(image_shape=[150, 150, 3], class_count=2)

    # 학습 및 평가 수행
    train (model, train_dataset, valid_dataset, learning_rate, epochs)
    eval (model, valid_dataset)

    """
    predict 예제
    """
    import pandas as pd
    import re

    TEST_DIR = DATA_DIR + "test_resize/"
    test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
    test_images.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    test_images = test_images[:10]

    # dataset 생성 및 predict
    test_dataset = data.Dataset_image([test_images], batch_size = batch_size)
    predict_list, probability_list = predict(model, test_dataset)

    # 결과 저장
    hp = np.array(probability_list)
    df = pd.DataFrame({"id": range(1, len(probability_list) + 1), "label": hp[:, 0], "class": predict_list})
    df.to_csv('predict.csv', index=False)

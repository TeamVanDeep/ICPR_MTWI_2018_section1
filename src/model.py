import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
import keras.backend as K
import word_dict
from keras.utils.vis_utils import plot_model

import numpy as np
import icpr_data as td
from PIL import Image


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]

    '''
    y_pred = y_pred[:, 2:, :]
    那么在 Keras 里面，CTC Loss 已经内置了，我们直接定义这样一个函数，即可实现 CTC Loss，
    由于我们使用的是循环神经网络，所以默认丢掉前面两个输出，因为它们通常无意义，且会影响模型的输出。    
    '''
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


image_height = 32

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# shape = (batch, image_height, image_width, channels)


def cnn_rnn_ctc_model(num_classes):
    '''
    base 用来预测，model用来训练，base包含于model中
    model部分用来训练，因为tf对ctc_loss的实现，需要单独为了ctc_loss对输入和输出扩展，输出是[y_pred, y_true]，base的输出才是[y_true]
    另外Keras的ctc_decode的实现有点问题，需要单独分出来以提高性能
    :return:basemodel, base
    '''
    '''
    stride为1的时候，当kernel为 3 padding为1或者kernel为5 padding为2 一看就是卷积前后尺寸不变。,选‘same’
    '''
    input_image = Input(shape=(32, None, 1))
    # base.add(InputLayer(shape=(image_height, None, 1)))
    m = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(input_image)
    m = MaxPool2D(pool_size=2, strides=2)(m)
    m = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(m)
    m = MaxPool2D(pool_size=2, strides=(2, 2))(m)  # 延长seq length
    m = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(m)
    m = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(m)
    m = MaxPool2D(pool_size=(1, 2), strides=(2, 1))(m)
    m = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(m)
    m = BatchNormalization()(m)
    m = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(m)
    m = BatchNormalization()(m)
    m = MaxPool2D(pool_size=(1, 2), strides=(2, 1))(m)
    m = Conv2D(filters=512, kernel_size=2, strides=1, padding='valid')(m)
    # m = ZeroPadding2D(padding=(0, 1))(m)

    # rnn part
    m = Permute((2, 1, 3))(m)  # change order, now shape should be (1, ?, 512)
    m = TimeDistributed(Flatten())(m)  # 这里把只有一个元素的维度去掉，shape(?, 512) 之后的每个层都按时间步展开
    m = Bidirectional(GRU(256, return_sequences=True))(m)
    m = Dense(256, name='blstm1_out', activation='linear')(m)
    m = Bidirectional(GRU(256, return_sequences=True))(m)

    y_pred = Dense(num_classes + 1, activation='softmax')(m)
    basemodel = Model(inputs=input_image, outputs=y_pred)
    # shape = (batch, image_width, num_classes)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')  # 预测序列长度，以图片宽来代替
    label_length = Input(name='label_length', shape=[1], dtype='int64')  # 真值序列长度，放在sentence.csv里了

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])
    # ctc输出元素取值在0-1之间的向量
    model = Model(inputs=[input_image, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    # model.summary()
    plot_model(model, show_layer_names=True, show_shapes=True, to_file="model.png")

    return model, basemodel


def get_model(height, nclass):
    from keras.optimizers import SGD
    rnnunit = 256
    input = Input(shape=(32, None, 1), name='the_input')
    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), name='pool1')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), name='pool2')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
    m = BatchNormalization()(m)
    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
    m = BatchNormalization()(m)
    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)

    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)

    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm1')(m)
    m = Dense(rnnunit, name='blstm1_out', activation='linear')(m)
    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm2')(m)
    y_pred = Dense(nclass, name='blstm2_out', activation='softmax')(m)

    basemodel = Model(inputs=input, outputs=y_pred)

    labels = Input(name='the_labels', shape=[None, ], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    # model.summary()
    return model, basemodel


def captcha_model(n_class):
    rnn_size = 128
    height = 80
    input_tensor = Input((height, 170, 1))
    x = input_tensor
    for i in range(3):
        x = Convolution2D(32, 3, 3, activation='relu')(x)
        x = Convolution2D(32, 3, 3, activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    x = Dense(32, activation='relu')(x)

    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 init='he_normal', name='gru1_b')(x)
    gru1_merged = Add()([gru_1, gru_1b])

    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 init='he_normal', name='gru2_b')(gru1_merged)
    x = Add()([gru_2, gru_2b])
    x = Dropout(0.25)(x)
    x = Dense(n_class+1, init='he_normal', activation='softmax')(x)
    base_model = Model(input=input_tensor, output=x)

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([x, labels, input_length, label_length])

    model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    return model, base_model

def train():
    from keras.callbacks import TensorBoard
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=True, write_images=True)
    mc = ModelCheckpoint(
        './logs/weight.hdf5',
        monitor='loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
    )

    # model, basemodel = cnn_rnn_ctc_model()

    num_classes = 10  # contains blank class
    model, basemodel = cnn_rnn_ctc_model(num_classes)  # from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    model, basemodel = captcha_model(num_classes)
    # model, basemodel = get_model(32, num_classes+1)

    import os
    #
    # # 加载之前训练的模型
    # if os.path.exists('./models/keras.hdf5'):
    #     basemodel.load_weights('./models/keras.hdf5')
    #
    # ##注意此处保存的是model的权重
    # checkpointer = ModelCheckpoint(filepath="./models/model{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss',
    #                                verbose=0, save_weights_only=False, save_best_only=True)
    # rlu = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto', min_delta=0.0001,
    #                         cooldown=0, min_lr=0)
    # if os.path.exists('./logs/weight.hdf5'):
    #     model.load_weights('./logs/weight.hdf5')
    # print(train_data[0][0][0].shape)
    # [[train_data[0][0][0], train_data[0][0][1]]
    # model.fit([[np.ones((32,137,1)),np.ones((32,136,1))], np.ones((2,5530)), np.ones((2,1)), np.ones((2,1))], [np.ones((2,5530))])
    # train_data = td.get_train_data()
    # model.fit(train_data[0], train_data[1])
    model.fit_generator(td.gen(True), epochs=10000, validation_steps=1000, steps_per_epoch=20,
                        callbacks=[tensorboard, mc])
    # model.fit_generator(td.gen(),
    #                     steps_per_epoch=1024,
    #                     epochs=10000,
    #                     validation_steps=1024)
    # [img_array_list, one_hot_matrix_list, input_length_list, label_length_list] = train_data[0]

    # model.fit(x = [img_array_list[np.newaxis, :, : ,:,: ], one_hot_matrix_list[0][np.newaxis,:,:], input_length_list[np.newaxis,:,:], label_length_list[np.newaxis,:, :]], y = label_length_list[np.newaxis, :, :],epochs=300,validation_split=0.05, verbose=2)
    # print(one_hot_matrix_list[0][np.newaxis,:,:].shape)
    # model.fit(x = [img_array_list, one_hot_matrix_list, input_length_list, label_length_list], y = one_hot_matrix_list,epochs=300,validation_split=0.05, verbose=2)

    # model.save_weights('./models/final_model_weights.h5')
    # model.save('./models/final_model.h5')


def test(data):
    import os
    num_classes = 10  # contains blank class
    model, basemodel = cnn_rnn_ctc_model(num_classes)
    if os.path.exists('./logs/weight.hdf5'):
        model.load_weights('./logs/weight.hdf5')
    # basemodel.predict()
    # ctc-beam-search


if __name__ == '__main__':
    train()

import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np
import word_dict
import pandas as pd
import glob
from captcha.image import ImageCaptcha
import random

img_target_height = 32
img_target_width = 160
n_len = 6     # 预测最大长度


def get_raw_data():
    '''
    [image_list, sentence_list]
    '''
    images_path = "image_output/"
    images = glob.glob(images_path + "*.jpg")
    sentences = pd.read_csv("sentences.csv", encoding='utf-8', iterator=True)
    image_num = len(images)
    for i in range(image_num):
        # print(i)
        yield [Image.open(images[i]),
               str(list(sentences.get_chunk(1)["content"])[0])]


def sentence2onehot(sentence: str):
    label = np.zeros(n_len)
    for i, char in zip(range(n_len), sentence):
        index = word_dict.out_charset.find(char)
        if index == -1:
            index = word_dict.out_charset.find(u' ')
        label[i] = index
    return label
    # tmp = [1 if x == y else 0 for x in word_dict.out_charset for y in sentence] # one hot matrix
    # tmp = np.array(tmp)
    # # print(tmp)
    # return tmp


def resize_image(img: Image):
    return img.resize((img_target_width, img_target_height))


def make_one_sample(img: Image, sentence: str):
    # [[图像宽，图像高，通道数]，文本独热矩阵，预测序列长度，真值序列长度]
    sample = []
    one_hot_matrix = []
    input_length = 0  # 预测序列长度
    label_length = 0  # 真实序列长度
    '''
        n_len 输出标签长度，必须大于最大时间步
    '''
    img_width = img.size[0]
    img_height = img.size[1]

    if img_width > img_height:  # 横向
        img = img.resize((int(img_width / img_height * 32), 32)).convert('L')  # 灰度图
        img_width = img.size[0]
        img_height = img.size[1]
    else:  # 竖向
        img = img.resize((32, int(img_height / img_width * 32))).convert('L')
        img_width = img.size[0]
        img_height = img.size[1]
        tmp_img = Image.new('L', (((int(img_height / 32) + 1) * 32), 32), 'white')
        # print(tmp_img.size)
        for x in range(int(img_height / 32) + 1):
            if img_height - img_width * (x + 1) >= 0:
                crop_box = (0, img_width * x, img_width, img_width * (x + 1))
            else:
                crop_box = (0, img_width * x, img_width, img_height)
            a = img.crop(crop_box).resize((img_width, img_width))
            #             tmp_img.paste(a, (32*(x+1), 32*(x+1), 32*x, 32*x))
            #             print((crop_box[2], crop_box[3], crop_box[0], crop_box[1]))
            # print(x)
            tmp_img.paste(a, (32 * x, 0, 32 * (x + 1), 32))
        img = tmp_img.copy()
    img = resize_image(img)
    img_array = np.array(img, dtype=np.float32)  # [图像宽，图像高]
    img_array = img_array / 255.0  # 归一化
    # img_array = img_array[:, :, np.newaxis]  # [图像宽，图像高，通道数]
    # img_array = img_array[:, :,:, np.newaxis]  # [图像宽，图像高，通道数]
    # img_array = img_array[np.newaxis, :, :, :]
    img_array = img_array.reshape((img.height, img.width, 1))
    one_hot_matrix = sentence2onehot(sentence)
    # input_length = int(img_array.shape[1]/32) # 怎么算靠猜 img_array.shape[1]是图片宽
    input_length = n_len
    label_length = min(len(sentence), n_len - 1)  # 超过n_len的句子将被截断
    # 合成数据
    sample_data = [img_array, one_hot_matrix, input_length, label_length]
    # sample_data = [np.array(x) for x in sample_data]
    return sample_data


# def get_train_data():
#     data_generator = get_raw_data()
#     img_array_list = []
#     one_hot_matrix_list = []
#     input_length_list = []
#     label_length_list = []
#     # for raw_data in data_generator:
#     #     sample = make_one_sample(raw_data[0], raw_data[1])
#     #     yield (sample, sample[1])
#     # for x, raw_data in zip(range(1), data_generator):
#     #     pass
#     for x, raw_data in zip(range(20), data_generator):
#         sample = make_one_sample(raw_data[0], raw_data[1])
#         img_array, one_hot_matrix, input_length, label_length = sample
#         img_array_list.append(img_array)
#         one_hot_matrix_list.append(one_hot_matrix)
#         input_length_list.append(input_length)
#         label_length_list.append(label_length)
#     input_length_list = np.array(input_length_list)
#     label_length_list = np.array(label_length_list)
#     one_hot_matrix_list = np.array(one_hot_matrix_list)
#     return ([img_array_list, one_hot_matrix_list, input_length_list, label_length_list],
#             one_hot_matrix_list)

def get_train_data():
    data_generator = get_raw_data()
    img_array_list = []
    one_hot_matrix_list = []
    input_length_list = []
    label_length_list = []
    samples = []
    # for raw_data in data_generator:
    #     sample = make_one_sample(raw_data[0], raw_data[1])
    #     yield (sample, sample[1])
    # for x, raw_data in zip(range(1), data_generator):
    #     pass
    for x, raw_data in zip(range(5), data_generator):
        sample = make_one_sample(raw_data[0], raw_data[1])
        img_array, one_hot_matrix, input_length, label_length = sample
        one_hot_matrix_list.append(one_hot_matrix)
        input_length_list.append(input_length)
        # label_length_list.append(label_length)
        label_length_list.append(7)
        img_array_list.append(img_array)
    # print(img_array_list[0].shape)
    one_hot_matrix_list = np.array(one_hot_matrix_list)
    return ([img_array_list, one_hot_matrix_list, input_length_list, label_length_list],
            [one_hot_matrix_list])


def gen(test=False):
    if test is False:
        loader = get_raw_data()
        for raw_image, sentence in loader:
            sample = make_one_sample(raw_image, sentence)
            img_array, one_hot_matrix, input_length, label_length = sample
            img_array = img_array[np.newaxis, :, :, :]
            one_hot_matrix = one_hot_matrix[np.newaxis, :]
            # print(one_hot_matrix)
            yield ([img_array, one_hot_matrix, np.ones(1) * input_length, np.ones(1) * label_length],
                   [one_hot_matrix])
    else:
        batch_size = 32
        width = 170
        height = 80
        characters = "1234567890"
        X = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
        y = np.zeros((batch_size, n_len), dtype=np.uint8)
        # random_str = ''.join([random.choice(characters) for j in range(4)])
        # generator = ImageCaptcha(width=width, height=height)
        # print(np.array(generator.generate_image(random_str).convert('L')).shape)
        while True:
            generator = ImageCaptcha()
            for i in range(batch_size):
                random_str = ''.join([random.choice(characters) for j in range(4)])
                X[i] = np.array(generator.generate_image(random_str).convert('L').resize((170,80)))[:,:,np.newaxis]
                X[i] = X[i] / 255.0
                y[i] = [characters.find(x) for x in random_str] + [0] * (n_len-4)
            yield [X, y, np.ones(batch_size) * n_len,
                   np.ones(batch_size) * n_len], np.ones(batch_size)
#

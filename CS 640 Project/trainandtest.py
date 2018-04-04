# Cs640 porject2 : Shifeng Li , Mingrui Yang and Anrui Wang
import tensorflow as tf
import vggnetwork as vgg19
import xlrd
import numpy as np
from skimage import io, transform, color
import glob
import os
import pandas as pd

import time
import csv
#import cv2 as cv
from keras.preprocessing import image
import xml.etree.ElementTree as ET

def get_v_lists(dirpath, ID):#get the top point
    
    xmllist = os.listdir(dirpath)
    for xml in xmllist:
        if xml[-4:] != '.xml':
            continue
        tree = ET.parse(dirpath + "/" + xml)
    root = tree.getroot()
    lists = []
    arr = np.zeros(shape=(0, 0))

    found = False
    for elem in tree.iter():
        if elem.tag == "name" and elem.text == ID:
            # print(elem.tag, elem.text)
            found = True
        if elem.tag == "polygon" and found == True:
            found = False
            # print(elem.tag, elem.text)
            for v in elem:
                for coord in v:
                    # print(coord.tag, coord.text)
                    arr = np.append(arr, int(coord.text))
            if len(arr) > 0:
                arr = np.reshape(arr, (int(len(arr) / 2), 2))
            lists.append(arr)
            # clear arr
            arr = np.zeros(shape=(0, 0))

    return lists


def build_bb(poly_v_lists):# build the bounding box
    pass
    x_max = 0
    x_min = 10000
    y_max = 0
    y_min = 10000
    for poly_v in poly_v_lists:
        for v in poly_v:
            cur_x = v[1]
            cur_y = v[0]
            x_max = cur_x if cur_x > x_max else x_max
            x_min = cur_x if cur_x < x_min else x_min
            y_max = cur_y if cur_y > y_max else y_max
            y_min = cur_y if cur_y < y_min else y_min
    res = np.asarray([np.ceil(x_min), np.ceil(y_min), np.floor(x_max - x_min), np.floor(y_max - y_min)], np.int32)
    return res


def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis] #read the width and heigh of the image
    tx = hshift * h #set the shift value of the heigh
    ty = wshift * w #set the shift value of the wide
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w) #keep the centroid and make it cannot be change
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        img = color.rgb2hsv(image)
        h, s ,v = img[:,:,0],img[:,:,1],img[:,:,2]
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        
        h = h + hue_shift
        
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = s + sat_shift
        
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = v + val_shift
        
        img[:,:,0],img[:,:,1],img[:,:,2] = h, s ,v
        
        image = color.hsv2rgb(img)
    
    return image

def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

# read the image
def read_img_train(path):
    w = 224
    h = 224
    labels = []
    imgs = []
    imgs1 = []
    imgs2 = []
    imgs3 = []
    imgs4 = []
    imgs5 = []
    imgs6 = []
    imgs7 = []
    dirlist = os.listdir(path)
    
    for dirs in dirlist:
        if os.path.isfile(path + dirs):
            continue
        for dir1 in os.listdir(path + dirs):
            if os.path.isfile(path + dirs + '/' + dir1):
                continue
            xls_file = xlrd.open_workbook(path + dirs + '/' + dir1 + '/' + dir1 + '.xls')
            xls_sheet = xls_file.sheets()[0]
            samples = np.array(xls_sheet.col_values(0))
            
            label = np.array(xls_sheet.col_values(3))
            
            imlist = os.listdir(path + dirs + '/' + dir1)
            for im in imlist:
                if im[-4:] != '.JPG':
                    continue
                
                print('reading train image:%s' % (path + dirs + '/' + dir1 + '/' + im))
                img = io.imread(path + dirs + '/' + dir1 + '/' + im)
                img = transform.resize(img, (w, h), 1, 'constant')
                img1 = img[::-1,:,:]
                img2 = img[:,::-1,:]
                
            
                imgs.append(img)
                labels.append(label[samples == im[:-4]])
#                imgs.append(img1)
#                imgs.append(img2)
#                labels.append(label[samples == im[:-4]])
#                labels.append(label[samples == im[:-4]])

                for i in range(3):
                    rotate_limit=(-20, 20)
                    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
                    img_rot = rotate(img, theta)
                    imgs.append(img_rot)
                    labels.append(label[samples == im[:-4]])

                    w_limit=(-20, 20)
                    h_limit=(-20, 20)
                    wshift = np.random.uniform(w_limit[0], w_limit[1])
                    hshift = np.random.uniform(h_limit[0], h_limit[1])
                    img_shift = shift(img, wshift, hshift)
                    imgs.append(img_shift)
                    labels.append(label[samples == im[:-4]])

                    zoom_range=(0.6, 1)
                    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
                    img_zoom = zoom(img, zx, zy)
                    imgs.append(img_zoom)
                    labels.append(label[samples == im[:-4]])

                    intensity = 0.5
                    sh = np.random.uniform(-intensity, intensity)
                    img_shear = shear(img, sh)
                    imgs.append(img_shear)
                    labels.append(label[samples == im[:-4]])

                    contrast_img = randomHueSaturationValue(img)
                    imgs.append(contrast_img)
                    labels.append(label[samples == im[:-4]])

#                    img_chsh = random_channel_shift(img, intensity = 0.05)
#                    imgs.append(img_chsh)
#                    labels.append(label[samples == im[:-4]])

    unique_label = np.unique(labels)
    L = []
    for i in range(len(unique_label)):
        L.append(i)
    L = np.array(L)
    label = []
    n = 0
    for l in labels:
        tmp2 = L[unique_label == l]
        label.append(tmp2)

    return np.asarray(imgs, np.float32), np.asarray(label,np.int32), len(unique_label), np.array(unique_label), np.array(L)

def getTestInfo(path, unique_label, L):
    w = 224
    h = 224
    imgs = []
    labels = []
    LABEL = []
    names = []
    NAME = []
    dirlist = os.listdir(path)
    
    for dirs in dirlist:
        if os.path.isfile(path + dirs):
            continue
        for dir1 in os.listdir(path + dirs):
            if os.path.isfile(path + dirs + '/' + dir1):
                continue
            xls_file = xlrd.open_workbook(path + dirs + '/' + dir1 + '/' + dir1 + '.xls')
            xls_sheet = xls_file.sheets()[0]
            IDs = np.array(xls_sheet.col_values(0))
            name = np.array(xls_sheet.col_values(2))
            label = np.array(xls_sheet.col_values(3))
            
            imlist = os.listdir(path + dirs + '/' + dir1)
            for im in imlist:
                if im[-4:] != '.jpg':
                    continue
                
                LABEL.append(label)
                NAME.append(name)
                print('reading test image:%s' % (path + dirs + '/' + dir1 + '/' + im))
                img = io.imread(path + dirs + '/' + dir1 + '/' + im)
                for ID in IDs:
                    v_list = get_v_lists(path + dirs + '/' + dir1, ID)
                    bbox = build_bb(v_list)

                    if bbox[0] < 1:
                        bbox[0] = 1
                    if bbox[1] < 1:
                        bbox[1] = 1
                    if bbox[2] > 0 and bbox[3] > 0 and bbox[0] > 0 and bbox[1] > 0:
                        imgp = img[bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3]]
                        imgp = transform.resize(imgp, (w, h), 1, 'constant')
                        labels.append(L[unique_label == label[IDs == ID]])
                        imgs.append(imgp)
                
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

path = '../CS640 Project dataset/'
data_train, label_train, num_class, unique_label, L = read_img_train(path)
data_test, label_test = getTestInfo(path, unique_label, L)


y_train = np.zeros((np.shape(data_train)[0], num_class))
for i in range(np.shape(data_train)[0]):
    y_train[i,label_train[i]] = 1

print(y_train.shape)
y_test = np.zeros((np.shape(data_test)[0], num_class))
for i in range(np.shape(data_test)[0]):
    y_test[i,label_test[i]] = 1

print(y_test.shape)

# keep the order
num_example = data_train.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data_train = data_train[arr]
y_train = y_train[arr]

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False): #get the minimize of the batch
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    data = []
    label = []
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        data.append(inputs[excerpt])
        label.append(targets[excerpt])
    return np.array(data), np.array((label))

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, num_class])
train_mode = tf.placeholder(tf.bool)
vgg = vgg19.Vggnetwork('./test-save_final.npy', num_class = num_class)
vgg.build(images, train_mode)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc8, labels=true_out)
#cost = tf.reduce_sum((vgg.prob - true_out)**2)
cost = tf.reduce_mean(cost)
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=100,decay_rate=0.9)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
add_global = global_step.assign_add(1)
#train = tf.train.(0.0001).minimize(cost)
correct_prediction = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 1)
correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #     correct_prediction = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 10)
    #     correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    #     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    n_epoch = 100
    batch_size = 64
    tf.global_variables_initializer().run()
    
    VALIDATION_SIZE = 100
    EARLY_STOP_PATIENCE = 10
    best_validation_loss = 100000
    current_epoch = 100
    X_valid, y_valid = data_test[:VALIDATION_SIZE], y_test[:VALIDATION_SIZE]
#    data_train, y_train = data_train[VALIDATION_SIZE:], y_train[VALIDATION_SIZE:]

    #     # test classification
    #     prob = sess.run(vgg.prob, feed_dict={images: x_train, true_out: y_train, train_mode: True})
    #     utils.print_prob(prob[0], './synset.txt')

    print("Number of Sample: %d, Total class: %d, Intial loss: %g"%(data_train.shape[0],num_class, sess.run(cost,feed_dict={images: data_train[:64], true_out: y_train[:64], train_mode: False})))
    for epoch in range(n_epoch):
        start_time = time.time()

        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        data, label = minibatches(data_train, y_train, batch_size, True)
        for i in range(data.shape[0]):
            x_train_a = data[i]
            y_train_a = label[i]
            _, err, ac, y, crr = sess.run([train, cost, acc, vgg.prob, correct_prediction],
                                  feed_dict={images: x_train_a, true_out: y_train_a, train_mode: True})
            print(np.where(y_train_a[0] == np.max(y_train_a[0]))[0][0])
            print(np.where(y[0] == np.max(y[0]))[0][0])
#            print(y[0])
#            print(y_train_a[0])
#            print(crr)
            train_loss = train_loss + err
            train_acc = train_acc + ac
            n_batch = n_batch + 1
            print("it: %d, numbatch: %d, loss: %g,acc: %g" % (epoch, n_batch, err, ac))

        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        validation_loss, val_acc = sess.run([cost, acc],
                                            feed_dict={images: X_valid, true_out: y_valid, train_mode: False})

        print('epoch %d done! validation loss: %g, accuracy: %g'%(epoch, validation_loss, val_acc))
        if (validation_loss < best_validation_loss) and (validation_loss < 6):
            best_validation_loss = validation_loss
            current_epoch = epoch
            vgg.save_npy(sess, './test-save.npy')
        elif (epoch - current_epoch) >= EARLY_STOP_PATIENCE:
            print('early stopping')
            break

    # utils.print_prob(prob[0], './synset.txt')
    # test
    total_corr = 0
    # test classification
    #    prob, acc = sess.run([vgg.prob, correct_num], feed_dict={images: data_test,true_out: y_test, train_mode: False})
    for i in range(data_test.shape[0]):
        prob, num = sess.run([vgg.prob, correct_num], feed_dict={images: [data_test[i, :, : ,:]],true_out: [y_test[i]], train_mode: False})
        prob = np.array(prob[0])
        total_corr = total_corr + num
        idx = np.where(prob == np.max(prob))[0][0]
        print(label_test[i][0])
        print(idx)
    #        print('for %d-th image, true label is'%(i), names_test[label_test[i][0]])
    #        print('Most possible test label is', names_test[idx], 'which is ', label_test[i] == idx)
    print('Accuracy is %g'%(total_corr / data_test.shape[0]))
    # test save
#    vgg.save_npy(sess, './test-save.npy')


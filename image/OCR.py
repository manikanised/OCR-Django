# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 18:25:14 2021

@author: Manik
"""

def conv(url, path):
    import cv2
    import numpy as np
    from tensorflow.keras import layers
    from tensorflow.keras import Model
    from tensorflow.keras import backend as tf_keras_backend

    tf_keras_backend.set_image_data_format('channels_last')


    img=cv2.imread(url)
    img=cv2.resize(img,(720,480))
    #imshow(img)


    def fixskew(image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
          angle = -(90 + angle)
        else:
          angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(image):
        return cv2.medianBlur(image,5)

    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    img=get_grayscale(img)
    #imshow(img)

    img=thresholding(img)
    #imshow(img)


    horizontal_hist = img.shape[1]-np.sum(img,axis=1,keepdims=True)/255
    #plt.plot(horizontal_hist)
    #plt.show()

    pos=1
    imges=[]
    j=0
    thresh=35
    for i in range(len(horizontal_hist)):
      if pos==1 and horizontal_hist[i]<thresh:
        if j-8>0:
          up=j-8
        else:
          up=j
        if i+8<len(img):
          dn=i+8
        else:
          dn=i
        imges.append(img[up:dn])
        pos=0
        j=i
        continue
      if pos==1 and horizontal_hist[i]>=thresh:
        continue
      if pos==0 and horizontal_hist[i]<thresh:
        continue
      if pos==0 and horizontal_hist[i]>=thresh:
        j=i
        pos=1
        continue
    print(len(imges))

    count=0
    for i in imges:
      print(count)
      count+=1
      #imshow(i)

    vh=imges[1].shape[0] - np.sum(imges[1],axis=0,keepdims=True)/255
    print(vh.shape,imges[1].shape)



    pos=0
    thresh=10
    t_v=5
    doc=[]
    for k in range(len(imges)):
      words=[]
      vh=imges[k].shape[0] - np.sum(imges[k],axis=0,keepdims=True)/255
      vh=vh[0]
      #plt.plot(vh)
      #plt.show()
      i=0
      while i<720:
        if pos==0 and vh[i]>=t_v:
          pos=1
          j=i
          i+=1
          continue
        if pos==0 and vh[i]<t_v:
          i+=1
          continue
        if pos==1 and vh[i]>=t_v:
          i+=1
          continue
        if pos==1 and vh[i]<t_v:
          sm=sum(vh[i:i+thresh])
          if sm<t_v:
            test=imges[k]
            test=test[:,j-2:i]
            words.append(test)
            i+=thresh
            pos=0
            j=i
          i+=1
      doc.append(words)


    def add_padding(img, old_w, old_h, new_w, new_h):
        h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
        w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
        #img_pad = np.zeros([new_h, new_w, 3]) 
        img_pad = np.ones([new_h, new_w, 3]) * 255
        img_pad[h1:h2, w1:w2, :] = img.reshape(img.shape[0],img.shape[1],1)
        return img_pad

    def fix_size(img, target_w, target_h):
        h, w = img.shape[:2]
        if w < target_w and h < target_h:
            img = add_padding(img, w, h, target_w, target_h)
        elif w >= target_w and h < target_h:
            new_w = target_w
            new_h = int(h * new_w / w)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = add_padding(new_img, new_w, new_h, target_w, target_h)
        elif w < target_w and h >= target_h:
            new_h = target_h
            new_w = int(w * new_h / h)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = add_padding(new_img, new_w, new_h, target_w, target_h)
        else:
            """w>=target_w and h>=target_h """
            ratio = max(w / target_w, h / target_h)
            new_w = max(min(target_w, int(w / ratio)), 1)
            new_h = max(min(target_h, int(h / ratio)), 1)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = add_padding(new_img, new_w, new_h, target_w, target_h)
        return img


    def preprocess(img, img_w, img_h):
        """ Pre-processing image for predicting """
        img=img.reshape(img.shape[0], img.shape[1], 1)
        img = fix_size(img, img_w, img_h)

        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)
        img /= 255
        return img


    input_data = layers.Input(name='the_input', shape=(128,64,1), dtype='float32')  # (None, 128, 64, 1)

    # Convolution layer (VGG)
    iam_layers = layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Activation('relu')(iam_layers)
    iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(iam_layers)  # (None,64, 32, 64)

    iam_layers = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Activation('relu')(iam_layers)
    iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)

    iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Activation('relu')(iam_layers)
    iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Activation('relu')(iam_layers)
    iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(iam_layers)  # (None, 32, 8, 256)

    iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Activation('relu')(iam_layers)
    iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv6')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Activation('relu')(iam_layers)
    iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)

    iam_layers = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Activation('relu')(iam_layers)

    # CNN to RNN
    iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)
    iam_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)

    # RNN layer
    # layer ten
    iam_layers = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(iam_layers)
    # layer nine
    iam_layers = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)

    # transforms RNN output to character activations:
    iam_layers = layers.Dense(80, kernel_initializer='he_normal', name='dense2')(iam_layers)
    iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

    labels = layers.Input(name='the_labels', shape=[16], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')


    iam_model_pred = None
    iam_model_pred = Model(inputs=input_data, outputs=iam_outputs)
    iam_model_pred.load_weights(filepath=path)

    letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    num_classes = len(letters) + 1
    print(num_classes)
    def numbered_array_to_text(numbered_array):
        numbered_array = numbered_array[numbered_array != -1]
        return "".join(letters[j] for j in numbered_array)

    stri=''
    for i in doc:
      for j in i:
        j=j.reshape(j.shape[0],j.shape[1],1)
        test_image_processed = preprocess(j, img_w=128, img_h=64)
        test_image_processed=test_image_processed.T

        test_image_processed=[test_image_processed]
        test_image_processed = np.array(test_image_processed)

        n=test_image_processed.shape[0]

        test_image_processed = test_image_processed.reshape(n, 128, 64, 1)
        test_image_processed.shape

        test_predictions_encoded = iam_model_pred.predict(x=test_image_processed)
        test_predictions_encoded.shape

        test_predictions_decoded = tf_keras_backend.get_value(tf_keras_backend.ctc_decode(test_predictions_encoded,
                                                                                      input_length = np.ones(test_predictions_encoded.shape[0])*test_predictions_encoded.shape[1],
                                                                                      greedy=True)[0][0])
        test_predictions_decoded.shape
        stri+=' '+numbered_array_to_text(test_predictions_decoded[0])[1:]

    return stri

    print('done')




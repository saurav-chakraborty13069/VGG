#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class intel_Image:
    def __init__(self,filename):
        self.filename =filename


    def predict_image(self):
        # load model
        model = load_model('lenet2.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        # if result[0][0] == 1:
        #     prediction = 'dog'
        #     return [{ "image" : prediction}]
        # else:
        #     prediction = 'cat'
        #     return [{ "image" : prediction}]

        for i in result[0]:
            if i == 1.0:
                index = np.where(result[0] == 1.0)[0][0]

        pred_class = {0: 'buldings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
        prediction = pred_class[index]
        return [{ "image" : prediction}]



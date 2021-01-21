import numpy as np
from keras.preprocessing import image
from keras.models import load_model
model = load_model('alexnet1.h5')
path = '/home/saurav/Documents/Saurav/MLDL/practice/cnn/archived_data/intelImage/archive/seg_pred/seg_pred'
test_image = image.load_img(path+'/73.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
for i in result[0]:
    if i == 1.0:
        index = np.where(result[0]==1.0)[0][0]


pred_class = {0:'buldings', 1:'forest', 2:'glacier', 3: 'mountain', 4:'sea', 5:'street'}
prediction = pred_class[index]
print(prediction)
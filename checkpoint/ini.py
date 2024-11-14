# import keras as K

import tensorflow.compat.v1 as tf
# import tensorflow as tf


from tensorflow.keras.models import load_model
import pywt
import numpy as np
import cv2
import os

# from keras.backend.tensorflow_backend import set_session
import shutil
os.environ['KERAS_BACKEND'] = 'tensorflow'

tf.disable_v2_behavior()  # 禁用 Eager Execution



# os.environ["CUDA_VISIBLE_DEVICES"]=""
# print(os.listdir("/media/liem/hai/haihh/dataset/classify/classify/train"))
#config = tf.compat.v1.ConfigProto() # 兼容
config = tf.ConfigProto() # 兼容 上面已经兼容
#config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)  #chay server thi comment
#sess = tf.compat.v1.Session(config=config)# 兼容

sess = tf.Session(config=config)# 兼容
#set_session(sess)  # set this TensorFlow session as the default
#tf.keras.backend.tensorflow_backend.set_session(config=config)
#tf.compat.v1.keras.backend.set_session(sess)
tf.keras.backend.set_session(sess)

def scale(x, mode=0, axis=None):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    if mode != 0:
        x = (x - 0.5) * 2
    return x

def DWT(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2     #500x402x4
    norm_LL = scale(LL, 0, 2)
    norm_LH = scale(LH, -1, 2)
    norm_HL = scale(HL, -1, 2)
    return norm_LL, norm_LH, norm_HL
def preproces_data(img_path, size):
    img = cv2.imread(img_path, 0)
    if img is None:
        return img
    w, h = img.shape[1], img.shape[0]
    #random crop

    #resize image
    img = cv2.resize(img, size)
    return img

model = load_model("weights-01-0.9875.hdf5")
PATH = "test_recapture"
GT = 1
batch_size = 16
list_file = os.listdir(PATH)
# print(list_file)
step = len(list_file) // batch_size
print(step)
count = 0
for iteration in range(step + 1):
    batch_ll = []
    batch_lh = []
    batch_hl = []
    files = []
    for item in range(batch_size):
        idx = item + iteration * batch_size
        if idx >= len(list_file):
            continue
        img_path = os.path.join(PATH, list_file[idx])
        img = cv2.imread(img_path, 0)
        #img = cv2.resize(img, (1280, 482))
        ll0, lh0, hl0 = DWT(preproces_data(img_path, (1280, 960)))

        ll1 = np.expand_dims(ll1, -1)
        lh1 = np.expand_dims(lh1, -1)
        hl1 = np.expand_dims(hl1, -1)
        l1 = [0.0, 1.0]
        ll_arr = np.array([ll0, ll1])
        lh_arr = np.array([lh0, lh1])
        hl_arr = np.array([hl0, hl1])
        outp = np.array([l0, l1])
        #yield ([ll_arr, lh_arr, hl_arr], outp)


        batch_ll.append(ll0)
        batch_lh.append(lh0)
        batch_hl.append(hl0)
        
#         img = cv2.resize(img, (224, 224))
#         batch.append(img)
        files.append(list_file[idx])
    if(len(batch_ll) == 0):
        continue
    batch_ll = np.asarray(batch_ll)
    batch_lh = np.asarray(batch_lh)
    batch_hl = np.asarray(batch_hl)
    
    res = model.predict([batch_ll, batch_lh, batch_hl],)
    print(res)
    class_res = np.argmax(res, axis=1)
    print(class_res)
    for iii in range(len(class_res)):
        if(class_res[iii] != GT):
            print("WRONG: {} ---- {}".format(files[iii], class_res[iii]))
            shutil.copy(os.path.join(PATH, files[iii]), os.path.join("wrong", files[iii]))
            count += 1
print("COUNT: ", count)
print("RESULT: {}".format(count/len(list_file)))
cv2.destroyAllWindows()
# print(res)
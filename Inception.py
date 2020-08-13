# from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import pylab
import tensorflow as tf
from tensorflow import keras
# base_model = VGG16(weights='imagenet')
import os
# from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix,accuracy_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def feature_extraction(x):
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)



    # x = preprocess_input(x)
    x = (2*(x/255.0) - 1.0).astype(np.float32)
    fc2 = model.predict(x)
    return fc2

def get_VGG_input(Vgg_input_image, Ultrasound_batch_size):
    u_image = np.resize(Vgg_input_image, (Ultrasound_batch_size, 299, 299, 1))
    #         u_zeros = np.zeros(u_image_denoise.shape)
    U_batch = np.concatenate((u_image, u_image, u_image), 3)
    #        print(U_batch_denoise.shape)
    #         print('aaaaaaaaaaaaaaa',U_batch.shape)
    # U_batch = (U_batch - np.min(U_batch)) / (np.max(U_batch) - np.min(U_batch))

    return U_batch
def read_image():
    cwd = 'train_crop/'
    class_path = ['class0', 'class1', 'class2', 'class3', 'class4']

    datavector = []
    label = []
    num = 1
    for index, name in enumerate(class_path):
        class_path1 = cwd + name + '/'

        for img_name in os.listdir(class_path1):
            img_path = class_path1 + img_name
            img = Image.open(img_path).convert('L')
            vgg_img = get_VGG_input(img, 1)
            print(vgg_img.shape)



            vector_data = feature_extraction(vgg_img)
            print(vector_data.shape)

            if num == 1:

                datavector = vector_data
                labela = np.array([index])
                label = labela.tolist()
                num = 2
            else:

                datavector = np.concatenate((datavector, vector_data), axis=0)
                #            datavector = datavector.append(vector_data)
                label.append(index)
            print(index)

    # datavector_matrix = np.reshape(datavector, (-1, 4096))
    #    datavector_matrix = np.reshape(datavector, (-1,4096))
    np.savetxt("dataset_train.csv", datavector, fmt="%f", delimiter=",")
    np.savetxt("label_train.csv", label, fmt="%d", delimiter=",")


if __name__ == '__main__':
    base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet',include_top=True)
    # print(base_model.summary())
    print("Model has been onload !")
    # rootdir = 'F:/shiyan/TensorFlow/retrain/data/train'
    # save_path = "dataset_incetion.csv"
    read_image()
    print("work has been done !")
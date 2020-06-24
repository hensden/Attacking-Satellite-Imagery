from __future__ import absolute_import, division, print_function, unicode_literals

import sys 
import os
import random 
import warnings 
import numpy as np 
from tqdm import tqdm
from PIL import Image
import tensorflow as tf 
from matplotlib import pyplot as plt 
from art.attacks import AdversarialPatch 
from art.classifiers import KerasClassifier 
from art.classifiers import TensorFlowClassifier 
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.densenet import DenseNet121 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense 
from armory.data.utils import download_file_from_s3
from armory.data.utils import maybe_download_weights_from_s3 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions 
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
sess = tf.compat.v1.InteractiveSession()
plt.rcParams['figure.figsize'] = [10, 10] 

SCALE = 0.35
num_classes = 45
label_name = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']


def mean_std():
    resisc_mean = np.array(
        [0.36386173189316956, 0.38118692953271804, 0.33867067558870334,]
    )

    resisc_std = np.array([0.20350874, 0.18531173, 0.18472934])

    return resisc_mean, resisc_std


def preprocess_input_densenet121_resisc(img):
    # Model was trained with Caffe preprocessing on the images
    # load the mean and std of the [0,1] normalized dataset
    # Normalize images: divide by 255 for [0,1] range
    mean, std = mean_std()
    img_norm = img / 255.0
    # Standardize the dataset on a per-channel basis
    output_img = (img_norm - mean) / std
    return output_img


def preprocessing_fn(x: np.ndarray) -> np.ndarray:
    shape = (224, 224)  # Expected input shape of model
    output = []
    for i in range(x.shape[0]):
        im_raw = image.array_to_img(x[i])
        im = image.img_to_array(im_raw.resize(shape))
        output.append(im)
    output = preprocess_input_densenet121_resisc(np.array(output))
    return output


def make_densenet121_resisc_model(**model_kwargs) -> tf.keras.Model:
    # Load ImageNet pre-trained DenseNet
    model_notop = DenseNet121(
        include_top=False, weights=None, input_shape=(224, 224, 3)
    )

    # Add new layers
    x = GlobalAveragePooling2D()(model_notop.output)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create graph of new model and freeze pre-trained layers
    new_model = Model(inputs=model_notop.input, outputs=predictions)

    for layer in new_model.layers[:-1]:
        layer.trainable = False
        if "bn" == layer.name[-2:]:  # allow batchnorm layers to be trainable
            layer.trainable = True

    # compile the model
    new_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return new_model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    model = make_densenet121_resisc_model(**model_kwargs)
    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        model.load_weights(filepath)

    mean, std = mean_std()
    wrapped_model = KerasClassifier(
        model, clip_values=((0.0 - mean) / std, (1.0 - mean) / std), **wrapper_kwargs
    )
    return wrapped_model


def from_keras(x):
    x = np.copy(x)
    x[:, :, 2] += 123.68
    x[:, :, 1] += 116.779
    x[:, :, 0] += 103.939
    return x[:, :, [2, 1, 0]].astype(np.uint8)


image_shape = (224, 224, 3)
batch_size = 45
scale_min = 0.3
scale_max = 1.0
rotation_max = 22.5
learning_rate = 20.0
max_iter = 1000

_image_input = tf.keras.Input(shape=image_shape)
_target_ys = tf.placeholder(tf.float32, shape=(None, 45))



model_kwargs = {} 
wrapper_kwargs = {} 
w_file = 'densenet121_resisc45_v1.h5' 
model = get_art_model(model_kwargs, wrapper_kwargs, w_file)

images_list = list()
dir = 'resisc45/'
NUM = int(sys.argv[1])

for image_path in tqdm(sorted(os.listdir('resisc45/'))):
    im = image.load_img(dir+image_path, target_size=(224, 224))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    im = preprocessing_fn(im)
    images_list.append(im)

print(len(images_list))
images = images_list[:batch_size]
images = np.concatenate(images, axis=0)
mean, std = mean_std()

ap = AdversarialPatch(
    classifier=model,
    target=NUM,
    scale_min=scale_min,
    scale_max=scale_max,
    learning_rate=learning_rate,
    max_iter=max_iter,
    batch_size=batch_size, 
    clip_patch=[(-mean[0]/std[0], (1.0 - mean[0])/std[0]), 
                (-mean[1]/std[1], (1.0 - mean[1])/std[1]),
                (-mean[2]/std[2], (1.0 - mean[2])/std[2])]
)

label = NUM
y_one_hot = np.zeros(45)
y_one_hot[NUM] = 1.0
y_target = np.tile(y_one_hot, (images.shape[0], 1))
patch, patch_mask = ap.generate(x=images, y=y_target)
PATCH = (patch * patch_mask).astype(np.uint8)
IMAGE = Image.fromarray(PATCH)
IMAGE.save('patch.png')
print("Patch Generated and saved")
patched_images = ap.apply_patch(images, scale=SCALE)
print("Patch Applied")
PRED_NO_PATCH = []
PRED = []

def predict_model(model, image0, ind, og0):
    image = np.copy(image0)
    og = np.copy(og0)
    
    image = np.expand_dims(image, axis=0)
    og = np.expand_dims(og, axis=0)
    
    image = preprocessing_fn(image)
    og = preprocessing_fn(og)
    
    prediction = model.predict(image)
    prediction_og = model.predict(og)
    
    top = 5
    
    a = np.asarray(prediction[0])
    top_idx = np.argsort(a)[-top:]
    top_v = [a[i] for i in top_idx]
    idx = [label_name[i] for i in top_idx]
    mean, std = mean_std()
    image0 *= std
    image0 += mean
    image0*=255.0

    img = 'results_g/im_'+str(ind)+'.png'
    img2 = 'results_g/og_'+str(ind)+'.png'
    og0 *= std
    og0 += mean
    og0*=255.0
    image2 = image0.astype(np.uint8)
    image3 = og0.astype(np.uint8)
    PRED.append(dict(zip(idx, top_v)))
    I = Image.fromarray(image2)
    I2 = Image.fromarray(image3)
    I.save(img)
    I2.save(img2)
    
    oga = np.asarray(prediction_og[0])
    ogtop_idx = np.argsort(oga)[-top:]
    ogtop_v = [oga[i] for i in ogtop_idx]
    ogidx = [label_name[i] for i in ogtop_idx]
    PRED_NO_PATCH.append(dict(zip(ogidx, ogtop_v)))

for i in tqdm(range(len(patched_images))):
    predict_model(model, patched_images[i,:,:,:], i, images[i,:,:,:])


print("Inference completed")


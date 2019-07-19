import os
import functools
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_eager_execution()
print("tf.__version__: ", tf.__version__)

# from tensorflow.keras.applications import resnet50
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from icecream import ic
from random import randint, gauss
from time import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

IDX_MINIVAN = 656
IDX_IND_ELEPHANT = 385
IDX_MANTIS = 315
IDX_AIRLINER = 404

SOURCE_IMG_PATH = "elephant.jpg"
SOURCE_IMG_PATH = "minivan.jpg"
SOURCE = IDX_MINIVAN
TARGET = IDX_AIRLINER

N_POPULATION = 100
N_GENERATIONS = 500

NET = 'ResNet50'
NET = 'MobileNetV2'

if NET == 'ResNet50':
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    model = ResNet50(weights="imagenet", include_top=True)
elif NET == 'MobileNetV2':
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    model = MobileNetV2(weights="imagenet", include_top=True)

# TODO: cleanup
# input_tensor = tf.keras.layers.Input(shape=(32, 32, 3))
# image = resnet50.preprocess_input(image)

# dataset = tfds.load(name='cifar10', split=tfds.Split.TRAIN)
# dataset = dataset.batch(1)

# for features in dataset.take(1):
#     image, label = features['image'], features['label']
#     ic(image.shape)


def show_image(x):
    plt.figure()
    plt.imshow(x)  # , vmin=0, vmax=255)
    plt.show()


def fitness_factory(image, target_class_idx):
    # TODO: memoization or batch_size > 1
    def fitnessfun(candidate):
        perturbation = np.zeros((224, 224, 3))
        x, y = int(candidate[0]), int(candidate[1])
        perturbation[x, y, 0] = candidate[2]
        perturbation[x, y, 1] = candidate[3]
        perturbation[x, y, 2] = candidate[4]
        adversarial = np.clip(image + perturbation, 0, 255)
        # show_image(adversarial)
        # save_img("adversarial.jpg", adversarial)
        adversarial = np.expand_dims(adversarial, axis=0)
        adversarial = preprocess_input(adversarial)
        prediction = model.predict(adversarial)
        return prediction[0, target_class_idx]
    return fitnessfun


def initialize_population(pop_size):
    xy = np.random.rand(pop_size, 2) * 223
    rgb = np.clip(np.random.randn(pop_size, 3) * 127 + 128, 0, 255)
    population = np.hstack((xy, rgb))
    return np.rint(population)


def make_children(parents, pop_size):
    # TODO: vectorize, for speed
    children = np.rint(
        np.array([
            parents[randint(0, pop_size - 1)] + 0.5 *
            (parents[randint(0, pop_size - 1)] -
             parents[randint(0, pop_size - 1)]) for _ in range(pop_size)
        ]))
    np.clip(children[:, :2], 0, 223, out=children[:, :2])
    np.clip(children[:, 2:], 0, 255, out=children[:, 2:])
    return children


def make_new_generation(parents, children):
    all = np.vstack((parents, children))
    unique = np.unique(all, axis=0)
    ic(len(unique))
    if len(unique) < N_POPULATION:
        new_random = initialize_population(N_POPULATION - len(unique))
        all = np.vstack((unique, new_random))
    else:
        all = unique
    scores = np.array([fitness(c) for c in all])
    idxs = np.flip(np.argsort(scores))[:N_POPULATION]
    return all[idxs], scores[idxs]


img = load_img(SOURCE_IMG_PATH, target_size=(224, 224))
img = img_to_array(img)
fitness = fitness_factory(img, TARGET)

x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
y_hat = model.predict(x)
ic(decode_predictions(y_hat))
ic(y_hat[0, SOURCE])
ic(y_hat[0, TARGET])

generation = initialize_population(N_POPULATION)
evolution = []
for i in range(N_GENERATIONS):
    t0 = time()
    children = make_children(generation, N_POPULATION)
    generation, scores = make_new_generation(generation, children)
    dt = round(time() - t0, 1)
    ic((i, dt))
    debug = [(s, x) for s, x in zip(scores, generation[:10, :])]
    ic(debug)
    evolution.append(scores)
np.save('evolution.npy', np.array(evolution))

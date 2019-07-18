import tensorflow as tf

# from tensorflow.keras.applications import resnet50
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from icecream import ic
from random import randint, gauss
from time import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)

tf.enable_eager_execution()

# dataset = tfds.load(name='cifar10', split=tfds.Split.TRAIN)
# dataset = dataset.batch(1)

# for features in dataset.take(1):
#     image, label = features['image'], features['label']
#     ic(image.shape)


def show_image(x):
    plt.figure()
    plt.imshow(x)  # , vmin=0, vmax=255)
    plt.show()


source = 385
target = 329

img_path = "elephant.jpg"
img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img)

# input_tensor = tf.keras.layers.Input(shape=(32, 32, 3))
# image = resnet50.preprocess_input(image)
model = ResNet50(weights="imagenet", include_top=True)


def fitness_factory(image, target_class_idx):
    def fitnessfun(candidate):
        perturbation = np.zeros((224, 224, 3))
        x, y = int(candidate[0]), int(candidate[1])
        perturbation[x, y, 0] = candidate[2]
        perturbation[x, y, 1] = candidate[3]
        perturbation[x, y, 2] = candidate[4]
        adversarial = np.clip(image + perturbation, 0.0, 255.0)
        # show_image(adversarial)
        # save_img("adversarial.jpg", adversarial)
        adversarial = np.expand_dims(adversarial, axis=0)
        adversarial = preprocess_input(adversarial)
        prediction = model.predict(adversarial)
        return prediction[0, target_class_idx]

    return fitnessfun


fitness = fitness_factory(img, target)

x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
y_hat = model.predict(x)
ic(y_hat[0, source])
ic(y_hat[0, target])

# y_hat = decode_predictions(y_hat)
# candidate = np.array([1, 1, 100, 100, 100], np.int)
# fitness(candidate)

N_POPULATION = 100
N_GENERATIONS = 500


def initialize_population(pop_size):
    return np.array([
        np.array([
            randint(0, 31),
            randint(0, 31),
            int(gauss(128, 127)),
            int(gauss(128, 127)),
            int(gauss(128, 127)),
        ]) for _ in range(pop_size)
    ])


def make_children(parents, pop_size):
    return np.rint(
        np.clip(
            np.array([
                parents[randint(0, pop_size - 1)] + 0.5 *
                (parents[randint(0, pop_size - 1)] -
                 parents[randint(0, pop_size - 1)]) for _ in range(pop_size)
            ]), 0.0, 255.0))


def make_new_generation(g0, c1):
    parent_scores = [fitness(g) for g in g0]
    children_scores = [fitness(c) for c in c1]
    gp = np.vstack((g0, c1))
    fit = parent_scores + children_scores
    idxs = np.flip(np.argsort(fit))[:N_POPULATION]
    return gp[idxs]


generation = initialize_population(N_POPULATION)

for i in range(N_GENERATIONS):
    t0 = time()
    children = make_children(generation, N_POPULATION)
    generation = make_new_generation(generation, children)
    dt = round(time() - t0, 1)
    ic((i, fitness(generation[0, :]), generation[0, :], dt))

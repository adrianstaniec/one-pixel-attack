import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from icecream import ic
from random import randint, gauss

tf.enable_eager_execution()

dataset = tfds.load(name='cifar10', split=tfds.Split.TRAIN)
dataset = dataset.shuffle(1024).batch(1)

for features in dataset.take(1):
    image, label = features['image'], features['label']
    ic(image.shape)

POP_SIZE = 10


def initialize_population(pop_size):
    return np.array([
        np.array([
            randint(0, 31),
            randint(0, 31),
            int(gauss(128, 127)),
            int(gauss(128, 127)),
            int(gauss(128, 127))
        ]) for _ in range(pop_size)
    ])


g0 = initialize_population(POP_SIZE)
ic(g0)


def make_children(parents, pop_size):
    return np.rint(
        np.array([
            parents[randint(0, pop_size - 1)] + 0.5 *
            (parents[randint(0, pop_size - 1)] -
             parents[randint(0, pop_size - 1)]) for _ in range(pop_size)
        ]))


c1 = make_children(g0, POP_SIZE)
ic(c1)


def fitness(candidate):
    # TODO: implement
    return 0


def make_new_generation(g0, c1):
    parent_scores = [fitness(g) for g in g0]
    children_scores = [fitness(c) for c in c1]
    # TODO: concat
    # TODO: argsort
    # TODO: vstack -> gather
    g1 = np.vstack((g0[:POP_SIZE], c1[POP_SIZE:]))
    return g1


g1 = make_new_generation(g0, c1)
ic(g1)

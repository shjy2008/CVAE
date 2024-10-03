import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pandas as pd
import os

# ----------- Modify this -------------
is_training = True # Set to True if want to train
generate_vowel = True # True(or 1): Vowel, False(or 0): Consonant
#--------------------------------------

MODEL_SAVE_PATH = "myModel_task1.keras"

NUM_CLASSES = 2

all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # 26 letters
letters = "ABCDEFGHIKLMNOPQRSTUVWXY" # only 24 letters, except J and Z
vowels = "AEIOU"

NUM_CLASSES_LETTERS = len(letters)

def index_to_letter(index):
    return letters[index]

def letter_to_index(letter):
    return letters.index(letter)

# label: the label in the csv table, 0-24 but doesn't have 9
def label_to_index(label):
    return letter_to_index(all_letters[label])

def is_label_a_vowel(label):
    return all_letters[label] in vowels

# Load the CSV file
train_data_csv = pd.read_csv('sign_mnist_train.csv')
y_train = train_data_csv.iloc[:, 0].values
x_train = train_data_csv.iloc[:, 1:].values
x_train = x_train.reshape(-1, 28, 28)
x_train = x_train.astype('float32') / 255.
x_train = x_train[..., tf.newaxis]

test_data_csv = pd.read_csv('sign_mnist_test.csv')
y_test = test_data_csv.iloc[:, 0].values
x_test = test_data_csv.iloc[:, 1:].values
x_test = x_test.reshape(-1, 28, 28)
x_test = x_test.astype('float32') / 255.
x_test = x_test[..., tf.newaxis]

# Convert label to 0(consonant) or 1(vowel)
y_train_is_vowel = np.array([1 if is_label_a_vowel(label) else 0 for label in y_train])
y_test_is_vowel = np.array([1 if is_label_a_vowel(label) else 0 for label in y_test])

# Convert label to index
y_train_letter_indexes = np.array([label_to_index(label) for label in y_train])

def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    Compute the log probability of a Gaussian distribution 
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.keras.utils.register_keras_serializable()
class CVAE(tf.keras.Model):
    """Conditional Convolutional variational autoencoder."""

    def __init__(self, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.latent_dim = 512
        self.num_classes = NUM_CLASSES
        add_dimension = self.num_classes if self.num_classes > 2 else 1 # If only 2 classes, only need to add one dimension
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(28, 28, 1 + add_dimension + NUM_CLASSES_LETTERS)),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(self.latent_dim + add_dimension + NUM_CLASSES_LETTERS,)),
                tf.keras.layers.Dense(units=7*7*self.latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, self.latent_dim)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                # tf.keras.layers.Conv2DTranspose(
                #     filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    def sample(self, is_vowel, letter_index = None):
        eps = tf.random.normal((1, self.latent_dim))
        if is_vowel != None:
            is_vowel = tf.constant([is_vowel], dtype=tf.int32)
        if letter_index != None:
            letter_index = tf.constant([letter_index], dtype=tf.int32)
        return self.decode(eps, is_vowel, letter_index, apply_sigmoid=True)

    def encode(self, x, y, y_letter_index = None):
        if self.num_classes > 2:
            y_one_hot = tf.one_hot(y, depth=self.num_classes)
            y_one_hot = tf.reshape(y_one_hot, [-1, 1, 1, self.num_classes])
            y_one_hot = tf.tile(y_one_hot, [1, 28, 28, 1])
            x_cond = tf.concat([x, y_one_hot], axis=-1)
        else:
            # Don't need one-hot encoding, concatenate directly
            y = tf.cast(y, dtype=tf.float32)
            y = tf.reshape(y, [-1, 1, 1, 1])
            y = tf.tile(y, [1, 28, 28, 1])
            x_cond = tf.concat([x, y], axis=-1)
        
        if y_letter_index != None:
            y_letter_one_hot = tf.one_hot(y_letter_index, depth=NUM_CLASSES_LETTERS)
            y_letter_one_hot = tf.reshape(y_letter_one_hot, [-1, 1, 1, NUM_CLASSES_LETTERS])
            y_letter_one_hot = tf.tile(y_letter_one_hot, [1, 28, 28, 1])
            x_cond = tf.concat([x_cond, y_letter_one_hot], axis=-1)
        else:
            # If y_letter_index is None, pad with zeros for NUM_CLASSES_LETTERS
            padding = tf.zeros([tf.shape(x)[0], 28, 28, NUM_CLASSES_LETTERS])
            x_cond = tf.concat([x_cond, padding], axis=-1)

        mean, logvar = tf.split(self.encoder(x_cond), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, y, y_letter_index = None, apply_sigmoid=False):
        if y != None:
            if self.num_classes > 2:
                y_one_hot = tf.one_hot(y, depth=self.num_classes)
                z_cond = tf.concat([z, y_one_hot], axis=1)
            else:
                y = tf.cast(y, dtype=tf.float32)
                y = tf.reshape(y, [-1, 1])
                z_cond = tf.concat([z, y], axis=1)
        else:
            # If y is None, check the number of classes
            if self.num_classes > 2:
                padding = tf.zeros([tf.shape(z)[0], self.num_classes])
                z_cond = tf.concat([z, padding], axis=-1)
            else:
                # Pad with a single zero if there are only 2 classes (binary)
                padding = tf.zeros([tf.shape(z)[0], 1])
                z_cond = tf.concat([z, padding], axis=-1)

        if y_letter_index != None:
            y_letter_one_hot = tf.one_hot(y_letter_index, depth=NUM_CLASSES_LETTERS)
            z_cond = tf.concat([z_cond, y_letter_one_hot], axis=-1)
        else:
            # If y_letter_index is None, pad with zeros for NUM_CLASSES_LETTERS
            padding = tf.zeros([tf.shape(z)[0], NUM_CLASSES_LETTERS])
            z_cond = tf.concat([z_cond, padding], axis=-1)

        logits = self.decoder(z_cond)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits



class CVAE_trainer():

    def __init__(self, train_images, train_labels, train_letter_indexes, test_images, test_labels, batch_size=256):
        self.cvae = CVAE()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
        # self.num_examples_to_generate = 16
        # self.seed = tf.random.normal([self.num_examples_to_generate, self.latent_dim])
        self.batch_size = batch_size
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_letter_indexes)).shuffle(train_images.shape[0]).batch(self.batch_size)
        self.test_dataset = (tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(test_images.shape[0]).batch(self.batch_size))
        self.num_batches = train_images.shape[0] // self.batch_size
    

    def compute_loss(self, x, y, train_letter_indexes = None):
        mean, logvar = self.cvae.encode(x, y, train_letter_indexes)
        z = self.cvae.reparameterize(mean, logvar)
        x_logit = self.cvae.decode(z, y, train_letter_indexes)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logpz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logpz_x)


    def train_step(self, x, y, train_letter_indexes):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y, train_letter_indexes)
            gradients = tape.gradient(loss, self.cvae.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.cvae.trainable_variables))


    def train(self, epochs):
        # self.generate_random_and_save(0)
        if os.path.exists(MODEL_SAVE_PATH):
            self.load_model()
            print ("Load previous model success, continue to train based on the previous model")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x, train_y, train_letter_indexes in self.train_dataset:
                self.train_step(train_x, train_y, train_letter_indexes)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x, test_y in self.test_dataset:
                loss(self.compute_loss(test_x, test_y))
            elbo = -loss.result()
            # display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
            # self.generate_random_and_save(epoch)
            self.generate_and_save_images(epoch)

            self.save_model()

    def generate_and_save_images(self, epoch):
        num_examples = 24
        generated_images = []
        for i in range(num_examples):
            generated_image = self.cvae.sample(i % NUM_CLASSES)
            generated_images.append(generated_image[0])

        fig = plt.figure(figsize=(13, 10))
        for i in range(num_examples):
            plt.subplot(5, 6, i + 1)
            plt.imshow(generated_images[i][:, :, 0], cmap='gray')
            title = "Vowel" if i % NUM_CLASSES > 0 else "Consonant"
            plt.title(f'{title}')
            plt.axis('off')

        plt.savefig(f'generated_images_epoch_{epoch:04d}.png')
        plt.close()

    def save_model(self):
        self.cvae.save(MODEL_SAVE_PATH)
    
    def load_model(self):
        self.cvae = tf.keras.models.load_model(MODEL_SAVE_PATH)


def generate_image_vowel_or_consonant(is_vowel):
    myModel = tf.keras.models.load_model(MODEL_SAVE_PATH)
    y = 1 if is_vowel else 0

    num_examples = 24
    generated_images = []
    for i in range(num_examples):
        generated_image = myModel.sample(y)
        generated_images.append(generated_image[0])

    fig = plt.figure(figsize=(13, 10))
    for i in range(num_examples):
        plt.subplot(5, 6, i + 1)
        plt.imshow(generated_images[i][:, :, 0], cmap='gray')
        title = "Vowel" if is_vowel else "Consonant"
        plt.title(title)
        plt.axis('off')

    plt.savefig(f'generated_{title}.png')
    # plt.close()
    plt.show()

if is_training:
    trainer = CVAE_trainer(x_train, y_train_is_vowel, y_train_letter_indexes, x_test, y_test_is_vowel)
    trainer.train(epochs=50)
else:
    generate_image_vowel_or_consonant(generate_vowel)


def generate_image_with_letter(letter):
    letter = letter.upper()
    if letter in letters:
        myModel = tf.keras.models.load_model(MODEL_SAVE_PATH)
        index = letter_to_index(letter)

        num_examples = 24
        generated_images = []
        for i in range(num_examples):
            generated_image = myModel.sample(None, index)
            generated_images.append(generated_image[0])

        fig = plt.figure(figsize=(13, 10))
        for i in range(num_examples):
            plt.subplot(5, 6, i + 1)
            plt.imshow(generated_images[i][:, :, 0], cmap='gray')
            plt.title(f'Letter: {letter}')
            plt.axis('off')

        plt.savefig(f'generated_{letter}.png')
        # plt.close()
        plt.show()
    else:
        print (f"Input value invalid, please try again. Valid input: {letters}")

# generate_image_vowel_or_consonant(True)
# generate_image_with_letter("b")
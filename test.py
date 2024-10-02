import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pandas as pd
import os

MODEL_SAVE_PATH = "myModel.keras"

# Load the CSV file
train_data_csv = pd.read_csv('sign_mnist_train.csv')
y_train = train_data_csv.iloc[:, 0].values#[:100]
x_train = train_data_csv.iloc[:, 1:].values#[:100]
x_train = x_train.reshape(-1, 28, 28)
x_train = x_train.astype('float32') / 255.
x_train = x_train[..., tf.newaxis]

test_data_csv = pd.read_csv('sign_mnist_test.csv')
y_test = test_data_csv.iloc[:, 0].values
x_test = test_data_csv.iloc[:, 1:].values
x_test = x_test.reshape(-1, 28, 28)
x_test = x_test.astype('float32') / 255.
x_test = x_test[..., tf.newaxis]

vowels = [0, 4, 8, 14, 20]

def create_rotated_data(x, y):
    rotated_x = []
    rotated_y = []
    for i in range(len(x)):
        image = x[i]
        rotated_x.append(image)
        rotated_y.append(0)
        for rotate_90_num in range(1, 4): # 1: 90, 2: 180, 3: 270
            rotated_image = tf.image.rot90(image, k = rotate_90_num)
            rotated_x.append(rotated_image)
            rotated_y.append(rotate_90_num)

    rotated_x = np.array(rotated_x)
    rotated_y = np.array(rotated_y)
    return (rotated_x, rotated_y)

rotated_train_data_path = 'rotated_train_data.npz'
rotated_test_data_path = 'rotated_test_data.npz'

if os.path.exists(rotated_train_data_path) and os.path.exists(rotated_test_data_path):
    print("Local rotated data exists, do not need to create")
    rotated_train_data = np.load(rotated_train_data_path)
    rotated_x_train = rotated_train_data['x']
    rotated_y_train = rotated_train_data['y']

    rotated_test_data = np.load(rotated_test_data_path)
    rotated_x_test = rotated_test_data['x']
    rotated_y_test = rotated_test_data['y']
else:
    print("Creating local rotated data...")
    rotated_x_train, rotated_y_train = create_rotated_data(x_train, y_train)
    rotated_x_test, rotated_y_test = create_rotated_data(x_test, y_test)

    np.savez('rotated_train_data.npz', x=rotated_x_train, y=rotated_y_train)
    np.savez('rotated_test_data.npz', x=rotated_x_test, y=rotated_y_test)

    print("Finish creating local rotated data")


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
        self.num_classes = 26
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(28, 28, 1 + self.num_classes)),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(self.latent_dim + self.num_classes,)),
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

    def sample(self, label):
        eps = tf.random.normal((1, self.latent_dim))
        labels = tf.constant([label], dtype=tf.int32)
        return self.decode(eps, labels, apply_sigmoid=True)

    def encode(self, x, y):
        y_one_hot = tf.one_hot(y, depth=self.num_classes)
        y_one_hot = tf.reshape(y_one_hot, [-1, 1, 1, self.num_classes])
        y_one_hot = tf.tile(y_one_hot, [1, 28, 28, 1])
        x_cond = tf.concat([x, y_one_hot], axis=-1)
        mean, logvar = tf.split(self.encoder(x_cond), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, y, apply_sigmoid=False):
        y_one_hot = tf.one_hot(y, depth=self.num_classes)
        z_cond = tf.concat([z, y_one_hot], axis=1)
        logits = self.decoder(z_cond)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits



class CVAE_trainer():

    def __init__(self, train_images, train_labels, test_images, test_labels, batch_size=256):
        self.cvae = CVAE()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        # self.num_examples_to_generate = 16
        # self.seed = tf.random.normal([self.num_examples_to_generate, self.latent_dim])
        self.batch_size = batch_size
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(train_images.shape[0]).batch(self.batch_size)
        self.test_dataset = (tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(test_images.shape[0]).batch(self.batch_size))
        self.num_batches = train_images.shape[0] // self.batch_size
    

    def compute_loss(self, x, y):
        mean, logvar = self.cvae.encode(x, y)
        z = self.cvae.reparameterize(mean, logvar)
        x_logit = self.cvae.decode(z, y)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logpz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logpz_x)


    def train_step(self, x, y):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y)
            gradients = tape.gradient(loss, self.cvae.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.cvae.trainable_variables))


    def train(self, epochs):
        # self.generate_random_and_save(0)
        if os.path.exists(MODEL_SAVE_PATH):
            self.load_model()
            print ("Load previous model success")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x, train_y in self.train_dataset:
                self.train_step(train_x, train_y)
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
        num_examples = 26
        generated_images = []
        for label in range(num_examples):
            generated_image = self.cvae.sample(label)
            generated_images.append(generated_image[0])

        fig = plt.figure(figsize=(13, 10))
        for i in range(num_examples):
            plt.subplot(5, 6, i + 1)
            plt.imshow(generated_images[i][:, :, 0], cmap='gray')
            plt.title(f'Label: {i}')
            plt.axis('off')

        plt.savefig(f'generated_images_epoch_{epoch:04d}.png')
        plt.close()

    def save_model(self):
        self.cvae.save(MODEL_SAVE_PATH)
    
    def load_model(self):
        self.cvae = tf.keras.models.load_model(MODEL_SAVE_PATH)


# Train rotated data
# trainer = CVAE_trainer(rotated_x_train, rotated_y_train, rotated_x_test, rotated_y_test)
# trainer.train(epochs=50)


trainer = CVAE_trainer(x_train, y_train, x_test, y_test)
trainer.train(epochs=50)

# with tf.keras.utils.custom_object_scope({'CVAE': CVAE}):
#     trainer.load_model()
#     trainer.generate_and_save_images(1)

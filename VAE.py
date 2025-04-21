import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses, Model
from tensorflow.keras.datasets import mnist
from scipy.ndimage import gaussian_filter

# Load and preprocess data
(mn_x_train, _), (mn_x_test, _) = mnist.load_data()
mn_x_train = mn_x_train.astype('float32') / 255.
mn_x_test = mn_x_test.astype('float32') / 255.

mn_x_train = mn_x_train[..., tf.newaxis]
mn_x_test = mn_x_test[..., tf.newaxis]

# Blur the images
blur_x_train = np.array([gaussian_filter(img, sigma=4) for img in mn_x_train])
blur_x_test = np.array([gaussian_filter(img, sigma=4) for img in mn_x_test])

class VAE(Model):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten()
        ])
        
        self.mean_layer = layers.Dense(64)
        self.log_var_layer = layers.Dense(64)
        
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(7 * 7 * 8, activation='relu'),
            layers.Reshape((7, 7, 8)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * epsilon

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var

    def compute_loss(self, x, recon_x, mean, log_var):
        recon_loss = losses.binary_crossentropy(tf.keras.backend.flatten(x), tf.keras.backend.flatten(recon_x))
        recon_loss *= 28 * 28  # Rescale loss
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        return tf.reduce_mean(recon_loss + kl_loss)

# Training the VAE
vae = VAE()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        recon_x, mean, log_var = vae(x)
        loss = vae.compute_loss(x, recon_x, mean, log_var)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

# Train the VAE
epochs = 1
for epoch in range(epochs):
    for i in range(len(blur_x_train) // 32):
        x_batch = blur_x_train[i * 32:(i + 1) * 32]
        loss = train_step(x_batch)
    print(f'Epoch: {epoch + 1}, Loss: {loss.numpy()}')

# Testing the VAE
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original blurred image
    ax = plt.subplot(2, n, i + 1)
    plt.title("Original")
    plt.imshow(tf.squeeze(blur_x_test[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstruct image
    recon_x, _, _ = vae(blur_x_test[i:i+1])
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("Reconstructed")
    plt.imshow(tf.squeeze(recon_x))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import tensorflow as tf
print('TF Version: ', tf.__version__)
from platform import python_version
print('Python Version: ', python_version())

import os
import shutil


# In[2]:


GPU = 1 # define the GPU to use
# Set the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


# In[3]:


"""
## Setup
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
autotune = tf.data.AUTOTUNE


"""
## Prepare the dataset

"""

# Load the horse-zebra dataset using tensorflow-datasets.
#dataset, _ = tfds.load("cycle_gan/horse2zebra", with_info=True, as_supervised=True)
noisy_tr = np.load('hn_train_set.npy')
clean_tr = np.load('ln_train_set.npy')
noisy_va = np.load('hn_valid_set.npy')
clean_va = np.load('ln_valid_set.npy')
noisy_te = np.load('hn_test_set.npy')
clean_te = np.load('ln_test_set.npy')
#train_noisy, train_clean = np.expand_dims(noisy_tr[:200],axis=0), np.expand_dims(clean_tr[:200],axis=0)
#test_noisy, test_clean = np.expand_dims(noisy_tr[:1],axis=0), np.expand_dims(clean_tr[:1],axis=0)
train_noisy, train_clean = noisy_tr, clean_tr
valid_noisy, valid_clean = noisy_va, clean_va
test_noisy, test_clean = noisy_te, clean_te





# Define the standard image size.
orig_img_size = (500)
# Size of the random crops to be used during training.
input_img_size = (500,1)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = 256
batch_size = 5


"""
## Building blocks used in the CycleGAN generators and discriminators
"""


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3),
    strides=(1),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    #x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv1D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(input_tensor)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    #x = ReflectionPadding2D()(x)
    x = layers.Conv1D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3),
    strides=(2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3),
    strides=(2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv1DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


"""
## Build the generators

The generator consists of downsampling blocks: nine residual blocks
and upsampling blocks.
"""


def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    #x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv1D(filters, (7), kernel_initializer=kernel_init, use_bias=False,padding="same")(
        img_input
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    #x = ReflectionPadding2D(padding=(3))(x)
    x = layers.Conv1D(1, (7), padding="same")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


"""
## Build the discriminators

The discriminators implement the following architecture:
`C64->C128->C256->C512`
"""


def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv1D(
        filters,
        (4),
        strides=(2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4),
                strides=(2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4),
                strides=(1),
            )

    x = layers.Conv1D(
        1, (4), strides=(1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")


"""
## Build the CycleGAN model

We will override the `train_step()` method of the `Model` class
for training via `fit()`.
"""


class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def call(self, inputs):
        return (
            self.disc_X(inputs),
            self.disc_Y(inputs),
            self.gen_G(inputs),
            self.gen_F(inputs),
        )

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is noisy and y is clean
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }


"""
## Create a callback that periodically saves generated images
"""


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""
    
    def on_epoch_end(self, epoch, logs=None):
        #manually batch test set (200), and evaluate them
        #save noisy network inputs, denoised spectra
        #and ground truths
        spectra = test_noisy
        prediction = np.zeros(np.shape(spectra))
        GTS = np.zeros(np.shape(spectra))
        inputs = np.zeros(np.shape(spectra))
        counter = 0
        for i in range(200, np.shape(spectra)[0], 200):
            prediction[i-200:i,:] = np.squeeze(self.model.gen_G(spectra[i-200:i]))
            GTS[i-200:i,:] = test_clean[i-200:i]
            inputs[i-200:i,:] = spectra[i-200:i]
            counter = counter+1
        # get remaining bit of last batch
        prediction[(200*counter):] = np.squeeze(self.model.gen_G(spectra[(200*counter):]))
        GTS[(200*counter):] = test_clean[(200*counter):]
        inputs[(200*counter):] = spectra[(200*counter):]
        
        prediction = np.reshape(prediction,(-1,500))
        GTS = np.reshape(GTS,(-1,500))
        inputs = np.reshape(inputs,(-1,500))
        path = './epoch_' + str(epoch)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        np.save(path + '/network_denoised', prediction)
        np.save(path + '/network_denoised_GT', GTS)
        np.save(path + '/network_input', inputs)
        
        
        
        # compute supervised validation loss
        spectra_valid = valid_noisy
        prediction_valid = []
        GTS_valid = []
        inputs_valid = []
        counter = 0
        for i in range(200, np.shape(spectra_valid)[0], 200):
            prediction_valid.append(self.model.gen_G(spectra_valid[i-200:i]))
            GTS_valid.append(valid_clean[i-200:i])
            inputs_valid.append(spectra_valid[i-200:i])
            counter = counter+1
        # get remaining bit of last batch
        prediction_valid.append(self.model.gen_G(img[(200*counter):]))
        GTS_valid.append(valid_clean[(200*counter):])
        inputs_valid.append(spectra_valid[(200*counter):])
        
        prediction_valid = np.reshape(prediction,(-1,500))
        GTS_valid = np.reshape(GTS,(-1,500))
        inputs_valid = np.reshape(inputs_valid,(-1,500))
        
        valid_loss = np.mean(np.mean((np.squeeze(prediction_valid) - np.squeeze(GTS_valid))**2,axis=1))
        np.save('valid_loss_' + str(epoch), valid_loss)
        
        
        '''
        valid_loss = np.mean(np.mean((np.squeeze(prediction) - np.squeeze(test_clean))**2,axis=1))
        print('VALIDATION LOSS: ')
        print(valid_loss)
        ax[0].plot(np.squeeze(img)[50])
        ax[0].plot(np.squeeze(prediction)[50])
        ax[1].plot(np.squeeze(test_clean)[50])
        ax[1].plot(np.squeeze(prediction)[50])
        
        ax[2].plot(np.squeeze(img)[90])
        ax[2].plot(np.squeeze(prediction)[90])
        ax[3].plot(np.squeeze(test_clean)[90])
        ax[3].plot(np.squeeze(prediction)[90])
        
        ax[4].plot(np.squeeze(img)[20])
        ax[4].plot(np.squeeze(prediction)[20])
        ax[5].plot(np.squeeze(test_clean)[20])
        ax[5].plot(np.squeeze(prediction)[20])
        '''

"""
## Train the end-to-end model
"""


# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# Define the loss function for the generators


def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)
# Callbacks
plotter = GANMonitor()
checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True
)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


train_noisy = tf.data.Dataset.from_tensor_slices((train_noisy))
train_clean = tf.data.Dataset.from_tensor_slices((train_clean))
#test_horses = tf.data.Dataset.from_tensor_slices((test_horses))
#test_zebras = tf.data.Dataset.from_tensor_slices((test_zebras))
train_noisy = train_noisy.batch(batch_size)
train_clean = train_clean.batch(batch_size)



history = cycle_gan_model.fit(
    tf.data.Dataset.zip((train_noisy, train_clean)),
    epochs=2,
    callbacks=[plotter, model_checkpoint_callback],
)

np.save('my_history.npy',history.history)


# In[ ]:


#history=np.load('my_history.npy',allow_pickle='TRUE').item()


# In[ ]:





# 
# # Load the checkpoints
# weight_file = "./saved_checkpoints/cyclegan_checkpoints.001"
# cycle_gan_model.load_weights(weight_file).expect_partial()
# print("Weights loaded successfully")
# 
# 

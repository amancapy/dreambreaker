import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras import layers, models, Model, activations, losses, optimizers, regularizers
from keras.utils import plot_model
import gc


env = gym.make("ALE/Skiing", full_action_space=False)
obs = env.reset()


class NormalScalar(layers.Layer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x * tf.random.normal(1, 0., 1.)


def vae_loss(y_true, y_pred):
    print(y_pred.shape, y_true.shape)
    mu = y_pred[1]
    logs2 = y_pred[2]

    print(mu, logs2, y_pred)

    recon = losses.mse(y_true, y_pred)
    kld = -0.5 * tf.reduce_sum(1 + logs2 - tf.square(mu) - tf.exp(logs2))

    return recon + kld


class VAE(models.Model):
    def encoder(self):
        inputs = layers.Input((64, 64, 3))

        x = layers.Conv2D(32, 4, 2, activation="relu")(inputs)
        x = layers.Conv2D(64, 4, 2, activation="relu")(x)
        x = layers.Conv2D(128, 4, 2, activation="relu")(x)
        x = layers.Conv2D(256, 4, 2, activation="relu")(x)
        x = layers.Flatten()(x)

        mu = layers.Dense(128, activation="tanh")(x)
        logs2 = layers.Dense(128, activation="relu")(x)
        z = mu + tf.exp(logs2 / 2) * tf.random.normal((1, ), 0., 1.)

        return Model(inputs=inputs, outputs=[mu, logs2, z])
    
    def decoder(self):
        inputs = layers.Input((128, 1))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Reshape((1, 1, 128))(x)
        x = layers.Conv2DTranspose(256, 5, 2, activation="relu")(x)
        x = layers.Conv2DTranspose(128, 5, 2, activation="relu")(x)
        x = layers.Conv2DTranspose(64, 6, 2, activation="relu")(x)

        outputs = layers.Conv2DTranspose(3, 6, 2, activation="tanh")(x)

        return Model(inputs=inputs, outputs=outputs)
    
    def __init__(self):
        super().__init__()

load_saved_model = False
if not load_saved_model:
    opt = optimizers.Adam()
    loss = losses.MeanSquaredError()

    batchsize = 64
    engine = get_vae()
    
    for i in range(10):
        stream = []
        env.reset()

        while len(stream) < batchsize * 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs = env.reset()[0]
            obs = tf.convert_to_tensor(obs, dtype=tf.dtypes.float32)
            obs = tf.pad(obs, [[23, 23], [48, 48], [0, 0]])
            obs = tf.image.resize(obs, (64, 64)) / 255.
            stream.append(obs)

        stream = tf.convert_to_tensor(stream, dtype=tf.dtypes.float32)
        engine.fit(stream, stream, epochs=20, shuffle=True, batch_size=batchsize)
        gc.collect()

                    
            # with tf.GradientTape() as tape:
            #     stm = stream[-batchsize:]
            #     stm = tf.convert_to_tensor(stm, dtype=tf.dtypes.float32) / 255.
            #     w = engine(stm)
            #     l = loss(w, stm)

            #     grads = tape.gradient(l, engine.trainable_variables)
            #     opt.apply_gradients(zip(grads, engine.trainable_variables))

            #     stream = []
            #     print(i, f"{float(l): .6f}")
            #     i += 1
        
    engine.save(f"save")

else:
    engine = models.load_model(f"save")

env.reset()
vidstream = []
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()

    obs = tf.convert_to_tensor(obs, dtype=tf.dtypes.float32) / 255.
    obs = tf.pad(obs, [[23, 23], [48, 48], [0, 0]])
    obs = tf.image.resize(obs, (64, 64))
    vidstream.append(obs)
    # obs = tf.expand_dims(obs, axis=0)
    # pred = engine(obs)
    
    # if i % 100 == 0:
        # gc.collect()

vidstream = tf.convert_to_tensor(vidstream, dtype=tf.dtypes.float32)

reconstream = engine(vidstream)

vidstream = vidstream.numpy()
reconstream = reconstream.numpy()

reconstream[reconstream < 0.] = 0.
reconstream[reconstream > 1.] = 1.

_, ax = plt.subplots(1, 2)
for i in range(len(reconstream)):
    ax[0].imshow(vidstream[i])
    ax[1].imshow(reconstream[i])

    ax[1].set_xlabel([i])
    plt.pause(0.001)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras import layers, models, Model, activations, losses, optimizers, regularizers, metrics
from keras.utils import plot_model
import gc


env = gym.make("ALE/Breakout-v5", full_action_space=False)
obs = env.reset()


class NormalScalar(layers.Layer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x * 0.2 * tf.random.normal(1, 0, 1)


# adapted from https://keras.io/examples/generative/vae/
class VAE(models.Model):
    def __init__(self):
        super().__init__()
        self.total_loss_tracker = metrics.Mean("total_loss")
        self.recon_loss_tracker = metrics.Mean("recon_loss")
        self.kld_loss_tracker = metrics.Mean("kld_loss")

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.aside = Model(inputs=self.encoder.inputs, outputs=self.decoder(self.encoder.outputs[2]))
        self.aside.build((64, 64, 3))

    def get_built_shadow(self):
        return self.aside
    
    def get_encoder(self):
        inputs = layers.Input((64, 64, 3))

        x = layers.Conv2D(32, 4, 2, activation="relu")(inputs)
        x = layers.Conv2D(64, 4, 2, activation="relu")(x)
        x = layers.Conv2D(128, 4, 2, activation="relu")(x)
        x = layers.Conv2D(256, 4, 2, activation="relu")(x)
        x = layers.Flatten()(x)

        mu = layers.Dense(128)(x)
        logs2 = layers.Dense(128)(x)
        z = mu + NormalScalar()(tf.exp(logs2 / 2))

        return Model(inputs=inputs, outputs=[mu, logs2, z])
    
    def get_decoder(self):
        inputs = layers.Input((128, ))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Reshape((1, 1, 128))(x)
        x = layers.Conv2DTranspose(256, 5, 2, activation="relu")(x)
        x = layers.Conv2DTranspose(128, 5, 2, activation="relu")(x)
        x = layers.Conv2DTranspose(64, 6, 2, activation="relu")(x)

        outputs = layers.Conv2DTranspose(3, 6, 2, activation="tanh")(x)

        return Model(inputs=inputs, outputs=outputs)
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kld_loss_tracker]
    
    def __call__(self, x):
        _, _, z= self.encoder(x)
        return self.decoder(z)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, logs2, z = self.encoder(data)
            recon = self.decoder(z)

            recon_loss = losses.mse(data, recon)
            kld_loss = -.05 * tf.reduce_sum((1 + logs2 - tf.square(mu) - tf.exp(logs2)), axis=-1)

            total_loss = recon_loss + kld_loss

            grads = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.kld_loss_tracker.update_state(kld_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kld_loss": self.kld_loss_tracker.result()
            }


load_saved_model = False
if not load_saved_model:
    opt = optimizers.Adam()
    loss = losses.MeanSquaredError()

    batchsize = 64
    vae = VAE()
    vae.compile(optimizer=optimizers.Adam())

    for i in range(1):
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
        vae.fit(stream, epochs=1, shuffle=True, batch_size=batchsize)
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
    
    built_shadow = vae.get_built_shadow()
    built_shadow.save(f"save")

else:
    vae = models.load_model(f"save")

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

reconstream = vae(vidstream)

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

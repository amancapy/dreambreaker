import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym
import tensorflow as tf
from keras import layers, models, Model, activations, losses, optimizers, regularizers, metrics
from keras.utils import plot_model
import gc
import pygame

env = gym.make("ALE/Breakout-v5", full_action_space=False)
obs = env.reset()


class NormalScalar(layers.Layer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x * tf.random.normal(1, 0, 1)


class VAE(models.Model):
    def __init__(self, latent_size):
        super().__init__(latent_size)
        self.total_loss_tracker = metrics.Mean("total_loss")
        self.recon_loss_tracker = metrics.Mean("recon_loss")
        self.kld_loss_tracker = metrics.Mean("kld_loss")

        self.optimizer = optimizers.Adam()
        self.latent_size = latent_size    
        self.era = 0

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()


    def get_built_shadow(self):
        shadow = Model(inputs=self.encoder.inputs, outputs=self.decoder(self.encoder.outputs[2]))
        shadow.build((64, 64, 3))
        return shadow
    
    def get_encoder(self):
        inputs = layers.Input((64, 64, 3))

        x = layers.Conv2D(32, 4, 2, activation="relu")(inputs)
        x = layers.Conv2D(64, 4, 2, activation="relu")(x)
        x = layers.Conv2D(128, 4, 2, activation="relu")(x)
        x = layers.Conv2D(256, 4, 2, activation="relu")(x)
        x = layers.Flatten()(x)

        mu = layers.Dense(self.latent_size)(x)
        logs2 = layers.Dense(self.latent_size)(x)
        z = mu + NormalScalar()(tf.exp(logs2 / 2))

        return Model(inputs=inputs, outputs=[mu, logs2, z])
    
    def get_decoder(self):
        inputs = layers.Input((self.latent_size,))

        x = layers.Reshape((1, 1, self.latent_size))(inputs)
        x = layers.Dense(self.latent_size, activation="relu")(x)
        x = layers.Conv2DTranspose(128, 5, 2, activation="relu")(x)
        x = layers.Conv2DTranspose(64, 5, 2, activation="relu")(x)
        x = layers.Conv2DTranspose(32, 6, 2, activation="relu")(x)

        x = layers.Conv2DTranspose(3, 6, 2, activation="tanh")(x)

        return Model(inputs=inputs, outputs=x)
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kld_loss_tracker]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, logs2, z = self.encoder(data)
            recon = self.decoder(z)

            recon_loss = losses.mse(data, recon)
            kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum((1 + logs2 - tf.square(mu) - tf.exp(logs2)), axis=1))

            total_loss = recon_loss + 0. * kld_loss

            grads = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.kld_loss_tracker.update_state(kld_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "recon": self.recon_loss_tracker.result(),
                "kld": self.kld_loss_tracker.result()
            }


load_saved_model = True
save_name = "save"

if not load_saved_model:
    opt = optimizers.Adam()
    loss = losses.MeanSquaredError()

    batchsize = 256
    vae = VAE(64)
    vae.compile(optimizer=optimizers.Adam(learning_rate=0.0005, amsgrad=True))

    for i in range(20):
        stream = []
        env.reset()

        while len(stream) < batchsize * 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                a = env.reset()

            obs = tf.convert_to_tensor(obs, dtype=tf.dtypes.float32) / 255.
            obs = tf.pad(obs, [[23, 23], [48, 48], [0, 0]])
            obs = tf.image.resize(obs, (64, 64), "nearest")
            stream.append(obs)

        stream = tf.convert_to_tensor(stream, dtype=tf.dtypes.float32)
        vae.fit(stream, epochs=10, shuffle=True, batch_size=batchsize, verbose=1)
        vae.era += 1
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
    
    built_shadow = vae.get_built_shadow()
    built_shadow.save(save_name)


vae = models.load_model(save_name)

env.reset()
vidstream = []
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()

    obs = tf.convert_to_tensor(obs, dtype=tf.dtypes.float32) / 255.
    obs = tf.pad(obs, [[23, 23], [48, 48], [0, 0]])
    obs = tf.image.resize(obs, (64, 64), "nearest")
    vidstream.append(obs)


stream = []
env.reset()

while len(stream) < 1000:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        a = env.reset()

    obs = tf.convert_to_tensor(obs, dtype=tf.dtypes.float32) / 255.
    obs = tf.pad(obs, [[23, 23], [48, 48], [0, 0]])
    obs = tf.image.resize(obs, (64, 64), "nearest")
    stream.append(obs)

stream = tf.convert_to_tensor(stream, dtype=tf.dtypes.float32)
vidstream = tf.convert_to_tensor(stream, dtype=tf.dtypes.float32)
reconstream = vae(stream)

vidstream = vidstream.numpy()
reconstream = reconstream.numpy()

reconstream[reconstream < 0.] = 0.
reconstream[reconstream > 1.] = 1.


pygame.init()
display = pygame.display.set_mode((512, 256))


for x, y in zip(vidstream, reconstream):
    x = pygame.surfarray.make_surface(x * 255)
    x = pygame.transform.scale(x, (256, 256))
    x = pygame.transform.rotate(x, 270)
    x = pygame.transform.flip(x, True, False)

    y = pygame.surfarray.make_surface(y * 255)
    y = pygame.transform.scale(y, (256, 256))
    y = pygame.transform.rotate(y, 270)
    y = pygame.transform.flip(y, True, False)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    display.blit(x, (0, 0))
    display.blit(y, (256, 0))

    pygame.time.wait(96)
    pygame.display.update()

pygame.quit()
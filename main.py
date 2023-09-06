import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, Model, activations, losses, optimizers, regularizers
from keras.utils import plot_model
import gc

env = gym.make("ALE/Breakout-v5", full_action_space=False)
obs = env.reset()

class NormalScalar(layers.Layer):
    def __init__(self):
        super().__init__()
        pass

    def forward(x):
        return x * tf.random.normal(1, 0., 1.)

def get_vae():
    inputs = layers.Input((128, 128, 3))

    x = layers.Conv2D(16, 4, 2, activation="relu")(inputs)
    x = layers.Conv2D(32, 4, 2, activation="relu")(x)
    x = layers.Conv2D(64, 4, 2, activation="relu")(x)
    x = layers.Conv2D(128, 4, 2, activation="relu")(x)
    x = layers.Conv2D(256, 4, 2, activation="relu")(x)

    x = layers.Flatten()(x)

    mu = layers.Dense(1024, activation="tanh")(x)
    sigma = layers.Dense(1024, activation="relu")(x)

    z = layers.Add()([mu, NormalScalar()(sigma)])

    w = layers.Dense(1024, activation="relu")(z)
    w = layers.Reshape((1, 1, 1024))(w)
    w = layers.Conv2DTranspose(256, 5, 2, activation="relu")(w)
    w = layers.Conv2DTranspose(128, 5, 2, activation="relu")(w)
    w = layers.Conv2DTranspose(64, 5, 2, activation="relu")(w)
    w = layers.Conv2DTranspose(32, 6, 2, activation="relu")(w)
    w = layers.Conv2DTranspose(3, 6, 2, activation="tanh", use_bias=False)(w)

    vae = Model(inputs=inputs, outputs = w)
    vae.summary()
    vae.compile("adam", "mse")
    # plot_model(vae, show_shapes=True)

    return vae


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
            if terminated:
                env.reset()
            obs = tf.convert_to_tensor(obs, dtype=tf.dtypes.float32)
            obs = tf.pad(obs, [[23, 23], [48, 48], [0, 0]])
            obs = tf.image.resize(obs, (128, 128)) / 255.
            stream.append(obs)

            if terminated or truncated:
                env.reset()

        stream = tf.convert_to_tensor(stream, dtype=tf.dtypes.float32)

        engine.fit(stream, stream, epochs=10, shuffle=True, batch_size=batchsize)
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

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()

obs = tf.convert_to_tensor(obs, dtype=tf.dtypes.float32) / 255.
obs = tf.pad(obs, [[23, 23], [48, 48], [0, 0]])
obs = tf.image.resize(obs, (128, 128))
obs = tf.expand_dims(obs, axis=0)
pred = engine(obs)

_, ax = plt.subplots(1, 2)

ax[0].imshow(obs[0])
ax[1].imshow(pred[0])

plt.show()
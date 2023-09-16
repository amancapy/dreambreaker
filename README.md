# dreambreaker

Adaptation of Ha & Schmidhuber's [World Models](https://arxiv.org/abs/1803.10122) to the more challenging task of learning evolving/expanding game-state domains.

The essence of the "world model" is that it understands the dynamics of the world i.e. environment. Instead of predicting the entire image of $S_(t+1)$ given $(S_t, A_t)$, first an autoencoder is trained on the pure visual representation of the environment under agent exploration. Once the latent space is learned, the world model, a sequence model (in the paper, an LSTM) predicts $encoder(S_(t+1))$ given $(encoder(S_t), A_t)$. This comes with two advantages. One is that the latent represents the "essence" of the state, and the other is that since it is a small vector, agents that learn to act on this compressed input can be more effectively trained/selected.

---------------------------------

Progress so far: the VAE has trained fairly well. An architectural issue is the use of BCE in the reconstruction loss, since pure MSE seems to be completely unable to locate the ball. This needs to be done away with. What comes next is to train an LSTM to predict $S_(t+1)$ given $(S_t, A_t)$. Once both are well trained, the choice remains open for method of arriving upon good agents. The authors use evolutionary methods, but from what I know evo. methods do not scale all that well with network size given the sheer dimensionality of the search space.

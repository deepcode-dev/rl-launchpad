# PPO learning notes

PPO is an on-policy actor-critic method. The clipped objective compares new and rollout policy log probabilities, then caps the probability ratio so a minibatch cannot reward arbitrarily large policy changes. Generalized advantage estimation reduces variance by mixing temporal-difference residuals over time; in a vector environment, that recurrence must remain separate for every environment column.

For bounded continuous control, this project samples a Gaussian latent action and applies `tanh`. PPO must evaluate the executed bounded action under the same transformed distribution, including the log-Jacobian correction. Observation running moments are frozen throughout each rollout/update and refreshed only afterward so rollout and update log probabilities use one normalization contract.

# PPO tests

- [x] lam: 0.97, alpha 3e-4 - not good
- [x] rollout buffer size 2000 - not good
- [x] more vf iters
- [x] adjust adam lr
- [x] adjust steps_per_epoch
- [x] add n_epochs
- [x] adjust return to use td(lambda)
- [x] 26 combine optimizers
- [x] 27 normalize adv in mini batch
- [x] 28 add vf_coef
- [x] 29 use sb3 formula to calc approx_kl
- [x] 30 remove kl max
- [x] 31 add kl mean
- [x] 32 add kl max warning
- [x] 33 wrap with tf.tidy
- [x] 34 fix memeory leak
- [x] 35 add back kl max
- [x] 36 remove kl max
- [x] 37 random shuffle
- [x] 38 add back kl max
- [x] 39 adjust dense layers to 64
- [x] 40 remove unused loss calcuations
- [x] 41 add max norm constraint and gain to layers
- [x] 42 ajust max kl to 0.03
- [x] 43 remove bias terms
- [] fix vf loss high
- [] fix unstable results

# block env implementation
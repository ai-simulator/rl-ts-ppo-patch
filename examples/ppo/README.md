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
- [] 30 remove kl max
- [] fix vf loss high
- [] fix unstable results

# block env implementation
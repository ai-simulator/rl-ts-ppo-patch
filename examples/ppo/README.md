# block env implementation

- [x] 1 init block env
- [x] 2 working with simple game config size 4
- [x] 3 follow sb3 log_std, size 6, 2000 iters
- [x] 4 add conv option
- [x] 5 size 6, categorical, 2000 iters
- [x] 6 size 9, 1000 iters
- [x] 7 size 7, 1000 iters
- [x] 8 size 8, 1000 iters
- [x] 9 size 9, 1000 iters
- [x] 10 size 9, expert set
- [x] 11 size 9, expert set, more conv filters and larger conv kernel, relu -> not good
- [x] 12 size 9, expert set, 3000 iters
- [x] 13 negative 1 penalty
- [x] 14 negative -0.1 penalty, 4000 iters
- [x] 15 invalid action masking
- [x] 16 record fps and best ever reward
- [x] 17 split rollout and train stats
- [x] 18 reproduce results
- [x] 19 split rollout and train procedure
- [x] 20 split further
- [x] 21 mobile procedure
- [x] 22 remove conv
- [x] 23 no conv, 4000 iterations
- [x] 24 mobile setting, 4000 iterations
- [] 25 mobile setting, 4000 iterations, fix fps

# PPO pendulum tests

- [x] 2 adjust adam epsilon
- [x] 3 clip norm
- [x] 4 add timeout info

# PPO cartpole tests

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
- [x] 44 save and load model
- [x] 45 add max norm clipping and reduce max kl
- [x] 46 increase max kl
- [x] 47 remove norm limit in init
- [x] 48 GAE formula
- [x] 49 fix vf loss high by bootstraping timeout reward
- [x] 50 adjust max kl
- [x] 52 buggy categorical
- [x] 55 categorical, visual output

# regex

^(?!pen)

# python tfboard

```
python3.9 -m venv venv
source venv/bin/activate
pip install tensorboard==2.12.0
tensorboard --logdir logs
```
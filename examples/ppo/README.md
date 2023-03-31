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
- [x] 25 mobile setting, 5000 iterations, fix fps
- [x] 26 mobile setting, 8000 iterations
- [x] 27 reward scaling
- [x] 28 reward scaling divide 8
- [x] 29 higher n_epochs 10
- [x] 30 add back conv
- [x] 31 batch size 64
- [x] 32 batch size 64, steps per iteration 2048 GOOD
- [x] 33 batch size 64, steps per iteration 2048 n_epochs: 5
- [x] 34 batch size 64, steps per iteration 2048 no convs
- [x] 35 batch size 64, steps per iteration 2048 no convs re-run


```
__________________________________________________________________________________________
Layer (type)                Input Shape               Output shape              Param #   
==========================================================================================
input1 (InputLayer)         [[null,9,9,6]]            [null,9,9,6]              0         
__________________________________________________________________________________________
flatten_Flatten1 (Flatten)  [[null,9,9,6]]            [null,486]                0         
__________________________________________________________________________________________
dense_Dense1 (Dense)        [[null,486]]              [null,64]                 31168     
__________________________________________________________________________________________
dense_Dense2 (Dense)        [[null,64]]               [null,64]                 4160      
__________________________________________________________________________________________
dense_Dense3 (Dense)        [[null,64]]               [null,81]                 5265      
==========================================================================================
Total params: 40593
Trainable params: 40593
Non-trainable params: 0
__________________________________________________________________________________________
__________________________________________________________________________________________
Layer (type)                Input Shape               Output shape              Param #   
==========================================================================================
input2 (InputLayer)         [[null,9,9,6]]            [null,9,9,6]              0         
__________________________________________________________________________________________
flatten_Flatten2 (Flatten)  [[null,9,9,6]]            [null,486]                0         
__________________________________________________________________________________________
dense_Dense4 (Dense)        [[null,486]]              [null,64]                 31168     
__________________________________________________________________________________________
dense_Dense5 (Dense)        [[null,64]]               [null,64]                 4160      
__________________________________________________________________________________________
dense_Dense6 (Dense)        [[null,64]]               [null,1]                  65        
==========================================================================================
Total params: 35393
Trainable params: 35393
Non-trainable params: 0
__________________________________________________________________________________________
merged configs: {
  optimizer: AdamOptimizer {
    learningRate: 0.0003,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    accumulatedFirstMoment: [],
    accumulatedSecondMoment: [],
    accBeta1: Variable {
      kept: false,
      isDisposedInternal: false,
      shape: [],
      dtype: 'float32',
      size: 1,
      strides: [],
      dataId: {},
      id: 33384,
      rankType: '0',
      trainable: true,
      name: '0'
    },
    accBeta2: Variable {
      kept: false,
      isDisposedInternal: false,
      shape: [],
      dtype: 'float32',
      size: 1,
      strides: [],
      dataId: {},
      id: 33386,
      rankType: '0',
      trainable: true,
      name: '1'
    }
  },
  vf_coef: 0.5,
  ckptFreq: 1000,
  steps_per_iteration: 2048,
  n_epochs: 10,
  max_ep_len: 1000,
  batch_size: 64,
  iterations: 50,
  train_v_iters: 80,
  gamma: 0.99,
  lam: 0.95,
  clip_ratio: 0.2,
  seed: 0,
  train_pi_iters: 1,
  target_kl: 0.02,
  verbosity: 'info',
  name: 'PPO_Train',
  stepCallback: [Function: stepCallback],
  iterationCallback: [Function: iterationCallback],
  savePath: './models/block-35-mobile-2048-64-n-epochs-5-noconv'
}
iterations: 0
PPO_Train | Trajectory cut off by epoch at 14 steps
PPO_Train | Early stopping at batch 31/32 of optimizing policy due to reaching max kl 0.031649019569158554 / 0.03
PPO_Train | Iteration 0 metrics:  {
  kl: 0.031649019569158554,
  entropy: 2.4626107215881348,
  clip_frac: 0.34375,
  continueTraining: false,
  trained_epoches: 6,
  loss_pi: -0.01959216594696045,
  loss_vf: 3.6290812492370605,
  ep_rets: {
    min: 80,
    max: 290,
    mean: 127.46853146853147,
    bestEver: 290,
    std: 40.65061851872987
  },
  ep_rewards: { min: 8, max: 44, mean: 8.9638671875 },
  t: 0,
  fps: 367.61802189912044,
  duration_rollout: 0.057,
  duration_train: 0.014,
  fps_rollout: 1122.8070175438595,
  fps_train: 4571.428571428572
}
iterations: 1
```

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
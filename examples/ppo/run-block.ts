import * as RL from '../../src';
import { Block } from '../../src/Environments/examples/Block';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';
import { Game } from '../../src/Environments/examples/Block/model/game';
import { DEFAULT_CLEAR_LINE_GAME_CONFIG, SIMPLE_CONFIG } from '../../src/Environments/examples/Block/model/gameConfig';
import { expertSet } from '../../src/Environments/examples/Block/model/shape';
import { TrainMetrics } from '../../src/Algos/ppo';

const run_number = 36;
const steps_per_iteration = 2048;
const iterations = 8000;
const batch_size = 64;
const n_epochs = 10;
const conv = false;
const RUN = `block-${run_number}-s${steps_per_iteration}-i${iterations}-b${batch_size}-n${n_epochs}-conv-${conv}`;
const tfBoardPath = `./logs/${RUN}-${Date.now()}`;
const summaryWriter = tf.node.summaryFileWriter(tfBoardPath);

const modelPath = `./models/${RUN}`;
const savePath = modelPath;

const game = new Game({
  ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
  ...{
    width: 9,
    height: 9,
    ...expertSet,
  },
});

const main = async () => {
  random.seed(0);
  const makeEnv = () => {
    return new Block({ game });
  };
  const env = makeEnv();
  const ac = new RL.Models.MLPActorCritic(
    env.observationSpace,
    env.actionSpace,
    [64, 64],
    'tanh',
    conv,
    undefined,
    undefined
  );
  ac.print();
  const ppo = new RL.Algos.PPO(makeEnv, ac, {
    actionToTensor: (action: tf.Tensor) => {
      // console.log('TCL ~ action:', action);
      return action.squeeze();
    },
  });

  const config = {
    optimizer: tf.train.adam(3e-4, 0.9, 0.999, 1e-8),
    lam: 0.95,
    steps_per_iteration,
    n_epochs,
    batch_size,
    vf_coef: 0.5,
    target_kl: 0.02,
    savePath,
    iterationCallback(epochData) {
      summaryWriter.scalar('step_r', epochData.ep_rewards.mean, epochData.t);
      summaryWriter.scalar('reward', epochData.ep_rets.mean, epochData.t);
      summaryWriter.scalar('max', epochData.ep_rets.max, epochData.t);
      summaryWriter.scalar('best_ever', epochData.ep_rets.bestEver, epochData.t);
      // summaryWriter.scalar('delta_pi_loss', epochData.delta_pi_loss, epochData.t);
      // summaryWriter.scalar('delta_vf_loss', epochData.delta_vf_loss, epochData.t);
      summaryWriter.scalar('loss_vf', epochData.loss_vf, epochData.t);
      summaryWriter.scalar('loss_pi', epochData.loss_pi, epochData.t);
      summaryWriter.scalar('kl', epochData.kl, epochData.t);
      summaryWriter.scalar('entropy', epochData.entropy, epochData.t);
      summaryWriter.scalar('fps', epochData.fps, epochData.t);
      summaryWriter.scalar('fps_rollout', epochData.fps_rollout, epochData.t);
      summaryWriter.scalar('fps_train', epochData.fps_train, epochData.t);
      summaryWriter.scalar('duration_rollout', epochData.duration_rollout, epochData.t);
      summaryWriter.scalar('duration_train', epochData.duration_train, epochData.t);
    },
  };
  ppo.setupTrain(config);
  for (let i = 0; i < iterations; i++) {
    console.log('iterations:', i);
    const maxBatch = ppo.getMaxBatch();
    const startTime = Date.now();
    let collectBatch = 0;
    while (collectBatch < maxBatch) {
      ppo.collectRollout(collectBatch);
      collectBatch++;
    }
    // update actor critic
    ppo.prepareMiniBatch();
    let metrics: TrainMetrics = {
      kl: 0,
      entropy: 0,
      clip_frac: 0,
      trained_epoches: 0,
      continueTraining: false,
      loss_pi: 0,
      loss_vf: 0,
    };
    let continueTraining = true;
    let j = 0;
    for (; j < ppo.trainConfigs.n_epochs; j++) {
      if (!continueTraining) {
        break;
      }
      let batch = 0;
      while (batch < maxBatch && continueTraining) {
        metrics = ppo.update(batch);
        continueTraining = metrics.continueTraining;
        batch++;
      }
    }
    // collect metrics
    metrics.trained_epoches = j;
    ppo.collectMetrics(startTime, i, metrics);
  }
};

main();

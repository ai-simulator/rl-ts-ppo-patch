import * as RL from '../../src';
import { Block } from '../../src/Environments/examples/Block';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';
import { Game } from '../../src/Environments/examples/Block/model/game';
import { DEFAULT_CLEAR_LINE_GAME_CONFIG, SIMPLE_CONFIG } from '../../src/Environments/examples/Block/model/gameConfig';
import { expertSet } from '../../src/Environments/examples/Block/model/shape';

const RUN = `block-23-size9-split-mobile`;
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
  const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [64, 64], 'tanh', false);
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
    steps_per_iteration: 1024,
    n_epochs: 5,
    train_pi_iters: 10,
    train_v_iters: 10,
    batch_size: 64,
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
  const iterations = 4000;
  for (let i = 0; i < iterations; i++) {
    console.log('iterations:', i);
    const startTime = Date.now();
    ppo.collectRollout();
    // update actor critic
    const metrics = ppo.update();
    // collect metrics
    ppo.collectMetrics(startTime, i, metrics);
  }
};

main();

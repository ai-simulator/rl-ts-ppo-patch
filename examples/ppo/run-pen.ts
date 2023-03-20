import * as RL from '../../src';
import { Pendulum } from '../../src/Environments/examples/Pendulum';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';

const RUN = `pen-4-box`;
const tfBoardPath = `./logs/${RUN}-${Date.now()}`;
const summaryWriter = tf.node.summaryFileWriter(tfBoardPath);

const modelPath = `./models/${RUN}`;
const savePath = modelPath;

const main = async () => {
  random.seed(0);
  const makeEnv = () => {
    return new Pendulum({ discretizeActionSpace: true });
  };
  const env = makeEnv();
  const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [64, 64], 'tanh', false);
  const ppo = new RL.Algos.PPO(makeEnv, ac, {
    actionToTensor: (action: tf.Tensor) => {
      return action;
    },
  });
  await ppo.train({
    optimizer: tf.train.adam(3e-4, 0.9, 0.999, 1e-5),
    lam: 0.95,
    steps_per_iteration: 2048,
    iterations: 300,
    n_epochs: 10,
    train_pi_iters: 10,
    train_v_iters: 10,
    batch_size: 64,
    vf_coef: 0.5,
    target_kl: 0.03,
    savePath,
    iterationCallback(epochData) {
      summaryWriter.scalar('reward', epochData.ep_rets.mean, epochData.t);
      // summaryWriter.scalar('delta_pi_loss', epochData.delta_pi_loss, epochData.t);
      // summaryWriter.scalar('delta_vf_loss', epochData.delta_vf_loss, epochData.t);
      summaryWriter.scalar('loss_vf', epochData.loss_vf, epochData.t);
      summaryWriter.scalar('loss_pi', epochData.loss_pi, epochData.t);
      summaryWriter.scalar('kl', epochData.kl, epochData.t);
      summaryWriter.scalar('entropy', epochData.entropy, epochData.t);
    },
  });
};

main();

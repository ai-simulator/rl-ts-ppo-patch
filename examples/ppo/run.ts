import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';

const RUN = `15-iter-10|50-a-3e4-lam-0.95-step-1000`;
const tfBoardPath = `./logs/${RUN}-${Date.now()}`;
const summaryWriter = tf.node.summaryFileWriter(tfBoardPath);

const main = async () => {
  random.seed(0);
  const makeEnv = () => {
    return new CartPole();
  };
  const env = makeEnv();
  const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [24, 48]);
  const ppo = new RL.Algos.PPO(makeEnv, ac, {
    actionToTensor: (action: tf.Tensor) => {
      return action.argMax(1);
    },
  });
  await ppo.train({
    vf_optimizer: tf.train.adam(3e-4),
    pi_optimizer: tf.train.adam(3e-4),
    lam: 0.95,
    steps_per_epoch: 1000,
    epochs: 500,
    train_pi_iters: 10,
    train_v_iters: 50,
    epochCallback(epochData) {
      summaryWriter.scalar('reward', epochData.ep_rets.mean, epochData.t);
      summaryWriter.scalar('delta_pi_loss', epochData.delta_pi_loss, epochData.t);
      summaryWriter.scalar('delta_vf_loss', epochData.delta_vf_loss, epochData.t);
      summaryWriter.scalar('loss_vf', epochData.loss_vf, epochData.t);
      summaryWriter.scalar('loss_pi', epochData.loss_pi, epochData.t);
      summaryWriter.scalar('kl', epochData.kl, epochData.t);
      summaryWriter.scalar('entropy', epochData.entropy, epochData.t);
    },
  });
};

main();
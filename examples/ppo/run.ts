import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';

const RUN = `34-batch-64-a-3e4|3e4-lam-0.95-step-2048-epoch-250-n-10-ret-td-lambda-combine-optimizer`;
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
    optimizer: tf.train.adam(3e-4),
    lam: 0.95,
    steps_per_iteration: 2048,
    iterations: 250,
    n_epochs: 10,
    train_pi_iters: 10,
    train_v_iters: 10,
    batch_size: 64,
    vf_coef: 0.5,
    iterationCallback(epochData) {
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

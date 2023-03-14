import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';

const RUN = 2;
const tfBoardPath = `./logs/${RUN}-${Date.now()}`;
const summaryWriter = tf.node.summaryFileWriter(tfBoardPath);

describe('Test PPO', () => {
  it('should run', async () => {
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
      steps_per_epoch: 1000,
      epochs: 200,
      train_pi_iters: 10,
      train_v_iters: 80,
      epochCallback(epochData) {
        summaryWriter.scalar('reward', epochData.ep_rets.mean, epochData.t);
        summaryWriter.scalar('delta_pi_loss', epochData.delta_pi_loss, epochData.t);
        summaryWriter.scalar('delta_vf_loss', epochData.delta_vf_loss, epochData.t);
      },
    });
  }).slow(20000);
});

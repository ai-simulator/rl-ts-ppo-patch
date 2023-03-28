import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';

describe('Test PPO', () => {
  it('should run', async () => {
    random.seed(0);
    const makeEnv = () => {
      return new CartPole();
    };
    const env = makeEnv();
    const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [24, 48], 'tanh', false);
    const ppo = new RL.Algos.PPO(makeEnv, ac, {
      actionToTensor: (action: tf.Tensor) => {
        return action.squeeze();
      },
    });
    ppo.setupTrain({
      steps_per_iteration: 1000,
      iterations: 5,
      iterationCallback(epochData) {
        console.log(epochData);
      },
    });
    const iterations = 7;
    for (let i = 0; i < iterations; i++) {
      console.log('iterations:', i);
      const startTime = Date.now();
      ppo.collectRollout();
      // update actor critic
      const metrics = ppo.update();
      // collect metrics
      ppo.collectMetrics(startTime, i, metrics);
    }
  }).slow(20000);
});

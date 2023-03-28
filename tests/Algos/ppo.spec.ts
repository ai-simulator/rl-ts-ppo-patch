import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';
import { TrainMetrics } from '../../src/Algos/ppo';

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
    let i = 0;
    for (; i < iterations; i++) {
      const startTime = Date.now();
      let start = 0;
      while (start < ppo.trainConfigs.steps_per_iteration) {
        ppo.collectRollout(start, start + ppo.trainConfigs.batch_size);
        start += ppo.trainConfigs.batch_size;
      }
      // update actor critic
      ppo.prepareMiniBatch();
      const maxBatch = ppo.getMaxBatch();
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
  }).slow(20000);
});

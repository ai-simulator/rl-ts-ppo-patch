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
    const ac = new RL.Models.MLPActorCritic(
      env.observationSpace,
      env.actionSpace,
      [24, 48],
      'tanh',
      false,
      undefined,
      undefined
    );
    const ppo = new RL.Algos.PPO(makeEnv, ac, {
      actionToTensor: (action: tf.Tensor) => {
        return action.squeeze();
      },
    });
    ppo.setupTrain({
      steps_per_iteration: 1024,
      iterations: 5,
      iterationCallback(epochData) {
        console.log(epochData);
      },
    });
    const iterations = 7;
    let i = 0;
    for (; i < iterations; i++) {
      tf.tidy(() => {
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
      });

      console.log(tf.memory());
    }
    const action = ppo.act(env.state, env.invalidActionMask());
    console.log('action:', action);
  }).slow(20000);
});

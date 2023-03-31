import * as RL from '../../src';
import { Block } from '../../src/Environments/examples/Block';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';
import { Game } from '../../src/Environments/examples/Block/model/game';
import { DEFAULT_CLEAR_LINE_GAME_CONFIG, SIMPLE_CONFIG } from '../../src/Environments/examples/Block/model/gameConfig';
import { expertSet } from '../../src/Environments/examples/Block/model/shape';
import { TrainMetrics } from '../../src/Algos/ppo';

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
  const ac = new RL.Models.MLPActorCritic(env.observationSpace, env.actionSpace, [32], 'tanh', false);
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
    steps_per_iteration: 512,
    n_epochs: 3,
    batch_size: 32,
    vf_coef: 0.5,
    target_kl: 0.02,
    iterationCallback(epochData) {
      console.log('TCL ~ epochData:', epochData);
    },
  };
  ppo.setupTrain(config);
  const iterations = 20;
  for (let i = 0; i < iterations; i++) {
    tf.tidy(() => {
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
    });

    console.log(tf.memory());
  }
};

main();

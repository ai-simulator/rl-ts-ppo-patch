import * as RL from '../../src';
import { CartPole } from '../../src/Environments/examples/Cartpole';
import * as tf from '@tensorflow/tfjs-node';
import * as random from '../../src/utils/random';

const RUN = `cart-55-batch-64`;
const tfBoardPath = `./logs/${RUN}-${Date.now()}`;
const summaryWriter = tf.node.summaryFileWriter(tfBoardPath);

const modelPath = `./models/${RUN}`;
const savePath = modelPath;

const main = async () => {
  random.seed(0);
  const makeEnv = () => {
    return new CartPole();
  };
  const env = makeEnv();
  const ac = new RL.Models.MLPActorCritic(
    env.observationSpace,
    env.actionSpace,
    [64, 64],
    'tanh',
    false,
    undefined,
    undefined
  );
  ac.print();
  const ppo = new RL.Algos.PPO(makeEnv, ac, {
    actionToTensor: (action: tf.Tensor) => {
      return action.squeeze();
    },
  });
  await ppo.train({
    optimizer: tf.train.adam(3e-4, 0.9, 0.999, 1e-8),
    lam: 0.95,
    steps_per_iteration: 2048,
    iterations: 200,
    n_epochs: 10,
    train_pi_iters: 10,
    train_v_iters: 10,
    batch_size: 64,
    vf_coef: 0.5,
    target_kl: 0.02,
    savePath,
    iterationCallback: async (epochData) => {
      let obs = env.reset();
      let rewards = 0;
      while (true) {
        const action = ppo.act(obs, env.invalidActionMask());
        const stepInfo = env.step(action);
        rewards += stepInfo.reward;
        if (epochData.iteration > 10) {
          // after 10 epochs, start rendering the evaluation onto a web viewer
          await env.render('web', { fps: 60, episode: epochData.iteration });
        }
        obs = stepInfo.observation;
        if (stepInfo.done) break;
      }
      console.log(`Episode ${epochData.iteration} - Eval Rewards: ${rewards}`);
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

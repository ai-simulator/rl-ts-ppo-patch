import { Agent } from 'rl-ts/lib/Agent';
import { Environment } from 'rl-ts/lib/Environments';
import { Discrete, Space } from 'rl-ts/lib/Spaces';
import * as random from 'rl-ts/lib/utils/random';
import * as tf from '@tensorflow/tfjs';
import { PPOBuffer, PPOBufferComputations } from 'rl-ts/lib/Algos/ppo/buffer';
import { DeepPartial } from 'rl-ts/lib/utils/types';
import { deepMerge } from 'rl-ts/lib/utils/deep';
import * as np from 'rl-ts/lib/utils/np';
import nj, { NdArray } from 'numjs';
import { ActorCritic } from 'rl-ts/lib/Models/ac';
import pino from 'pino';
import { ct } from '../..';
import { readFileSync } from 'fs';
import { actionToMove } from 'rl-ts/lib/Environments/examples/Block';
const log = pino({
  prettyPrint: {
    colorize: true,
  },
});

export interface PPOConfigs<Observation, Action> {
  /** Converts observations to batchable tensors of shape [1, ...observation shape] */
  obsToTensor: (state: Observation) => tf.Tensor;
  /** Converts actor critic output tensor to tensor that works with environment. Necessary if in discrete action space! */
  actionToTensor: (action: tf.Tensor) => TensorLike;
  /** Optional act function to replace the default act */
  act?: (obs: Observation) => Action;
}
export interface PPOTrainConfigs {
  stepCallback(stepData: {
    step: number;
    episodeDurations: number[];
    episodeRewards: number[];
    episodeIteration: number;
    info: any;
    loss: number | null;
  }): any;
  iterationCallback(epochData: {
    iteration: number;
    kl: number;
    entropy: number;
    // delta_pi_loss: number;
    // delta_vf_loss: number;
    loss_pi: number;
    loss_vf: number;
    ep_rets: {
      min: number;
      max: number;
      mean: number;
      std: number;
    };
    ep_rewards: {
      min: number;
      max: number;
      mean: number;
    };
    t: number;
  }): any;
  optimizer: tf.Optimizer;
  vf_coef: number;
  /** How frequently in terms of total steps to save the model. This is not used if saveDirectory is not provided */
  ckptFreq: number;
  /** path to store saved models in. */
  savePath?: string;
  saveLocation?: TFSaveLocations;
  iterations: number;
  verbosity: string;
  gamma: number;
  lam: number;
  target_kl: number;
  clip_ratio: number;
  train_v_iters: number;
  train_pi_iters: number;
  steps_per_iteration: number;
  n_epochs: number;
  /** maximum length of each trajectory collected */
  max_ep_len: number;
  batch_size: number;
  seed: number;
  name: string;
}
type Action = NdArray | number;
export class PPO<
  Observation,
  ObservationSpace extends Space<Observation>,
  ActionSpace extends Space<Action>
> extends Agent<Observation, Action> {
  public configs: PPOConfigs<Observation, Action> = {
    obsToTensor: (obs: Observation) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if observation space is dict. if observation space is dict, user needs to override this.
      const tensor = np.tensorLikeToTensor(obs);
      return tensor.reshape([1, ...tensor.shape]);
    },
    actionToTensor: (action: tf.Tensor) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if action space is dict. if action space is dict, user needs to override this.
      return action;
    },
  };

  public env: Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>;

  /** Converts observations to batchable tensors of shape [1, ...observation shape] */
  private obsToTensor: (obs: Observation) => tf.Tensor;
  private actionToTensor: (action: tf.Tensor) => TensorLike;

  constructor(
    /** function that creates environment for interaction */
    public makeEnv: () => Environment<ObservationSpace, ActionSpace, Observation, any, Action, number>,
    /** The actor crtic model */
    public ac: ActorCritic<tf.Tensor>,
    /** configs for the PPO model */
    configs: DeepPartial<PPOConfigs<Observation, Action>> = {}
  ) {
    super();
    this.configs = deepMerge(this.configs, configs);

    this.env = makeEnv();
    this.obsToTensor = this.configs.obsToTensor;
    this.actionToTensor = this.configs.actionToTensor;
  }

  /**
   * Select action using the current policy network. By default selects the action by feeding the observation through the network
   * then return the argmax of the outputs
   * @param observation - observation to select action off of
   * @returns action
   */
  public act(observation: Observation): Action {
    return np.tensorLikeToNdArray(this.actionToTensor(this.ac.act(this.obsToTensor(observation))));
  }

  public async train(trainConfigs: Partial<PPOTrainConfigs>) {
    let configs: PPOTrainConfigs = {
      optimizer: tf.train.adam(3e-4, 0.9, 0.999, 1e-8),
      vf_coef: 0.5,
      ckptFreq: 1000,
      steps_per_iteration: 10000,
      n_epochs: 10,
      max_ep_len: 1000,
      batch_size: 64,
      iterations: 50,
      train_v_iters: 80,
      gamma: 0.99,
      lam: 0.97,
      clip_ratio: 0.2,
      seed: 0,
      train_pi_iters: 1,
      target_kl: 0.01,
      verbosity: 'info',
      name: 'PPO_Train',
      stepCallback: () => {},
      iterationCallback: () => {},
    };
    configs = deepMerge(configs, trainConfigs);
    log.level = configs.verbosity;

    // try to load model
    if (configs.savePath) {
      try {
        await this.ac.load(configs.savePath);
      } catch (error) {
        // console.log(error);
        console.log(`failed to load model from ${configs.savePath}`);
      }
    }

    const { clip_ratio, optimizer, target_kl } = configs;

    // TODO do some seeding things
    random.seed(configs.seed);
    // TODO: seed tensorflow if possible

    const env = this.env;
    const obs_dim = env.observationSpace.shape;

    let buffer_action_dim = env.actionSpace.shape;
    if (env.actionSpace instanceof Discrete) {
      buffer_action_dim = [1];
    }

    let local_steps_per_iteration = configs.steps_per_iteration;

    const buffer = new PPOBuffer({
      gamma: configs.gamma,
      lam: configs.lam,
      actDim: buffer_action_dim,
      obsDim: obs_dim,
      size: local_steps_per_iteration,
    });

    type pi_info = {
      approx_kl: number;
      entropy: number;
      clip_frac: any;
    };
    const compute_loss_pi = (data: PPOBufferComputations, epoch: number): { loss_pi: tf.Tensor; pi_info: pi_info } => {
      let { obs, act, adv } = data;
      return tf.tidy(() => {
        const logp_old = data.logp.expandDims(-1);
        const adv_e = adv.expandDims(-1);
        // console.log('TCL ~ act:', act);
        const { pi, logp_a } = this.ac.pi.apply(obs, act);
        // console.log('TCL ~ logp_a:', logp_a);
        // console.log('TCL ~ logp_old:', logp_old);

        const ratio = logp_a!.sub(logp_old).exp();
        // console.log('TCL ~ ratio:', ratio);

        const clip_adv = ratio.clipByValue(1 - clip_ratio, 1 + clip_ratio).mul(adv_e);
        // console.log('TCL ~ clip_adv:', clip_adv);

        const adv_ratio = ratio.mul(adv_e);
        // console.log('TCL ~ adv_ratio:', adv_ratio);

        const ratio_and_clip_adv = tf.stack([adv_ratio, clip_adv]);

        const loss_pi = ratio_and_clip_adv.min(0).mean().mul(-1);
        // const loss_pi = tf.tensor(0);
        // console.log('TCL ~ loss_pi:', loss_pi);

        // from stablebaseline3
        const log_ratio = logp_a!.sub(logp_old);
        const approx_kl = log_ratio.exp().sub(1).sub(log_ratio).mean().arraySync() as number;

        if (ratio.size === 64) {
          // console.log('TCL ~ logp_old:', logp_old.arraySync());
          // console.log('TCL ~ logp_a:', logp_a!.arraySync());
          // console.log('TCL ~ log_ratio:', log_ratio.arraySync());
          // console.log(epoch, 'TCL ~ approx_kl:', approx_kl);
          // console.log("TCL ~ adv:", adv.arraySync())
          // console.log('TCL ~ ratio:', ratio.arraySync());
        }
        const entropy = pi.entropy().mean().arraySync() as number;
        const clipped = ratio
          .greater(1 + clip_ratio)
          .logicalOr(ratio.less(1 - clip_ratio))
          .mean()
          .arraySync() as number;

        return {
          loss_pi,
          pi_info: {
            approx_kl,
            entropy,
            clip_frac: clipped,
          },
        };
      });
    };
    const compute_loss_vf = (data: PPOBufferComputations) => {
      const { obs, ret } = data;
      return tf.tidy(() => {
        const predict = this.ac.v.apply(obs).flatten();
        // if (data.ret.size === 64) {
        //   console.log('TCL ~ ret:', ret);
        //   console.log('TCL ~ obs:', obs.arraySync());
        //   console.log('TCL ~ ret:', ret.arraySync());
        //   console.log('TCL ~ predict:', predict.arraySync());
        //   console.log('TCL ~ predict.sub(ret).pow(2):', predict.sub(ret).pow(2));
        //   console.log('TCL ~ predict.sub(ret).pow(2):', predict.sub(ret).pow(2).arraySync());
        // }
        return predict.sub(ret).pow(2).mean();
      });
    };

    const update = async () => {
      return tf.tidy(() => {
        const data = buffer.get();
        const totalSize = configs.steps_per_iteration;
        const batchSize = configs.batch_size;

        let kls: number[] = [];
        let entropy = 0;
        let clip_frac = 0;
        let trained_pi_iters = 0;

        // let loss_pi_old = compute_loss_pi(data, 0).loss_pi.arraySync() as number;
        // let loss_vf_old = compute_loss_vf(data).arraySync() as number;

        let loss_pi_ = 0;
        let loss_vf_ = 0;

        let continueTraining = true;

        for (let epoch = 0; epoch < configs.n_epochs; epoch++) {
          let batchStartIndex = 0;
          let batch = 0;
          let maxBatch = Math.floor(totalSize / batchSize);
          const indices = tf.tensor1d(Array.from(tf.util.createShuffledIndices(totalSize)), 'int32');
          while (batch < maxBatch) {
            const batchData = {
              obs: data.obs.gather(indices.slice(batchStartIndex, batchSize)),
              act: data.act.gather(indices.slice(batchStartIndex, batchSize)),
              adv: data.adv.gather(indices.slice(batchStartIndex, batchSize)),
              ret: data.ret.gather(indices.slice(batchStartIndex, batchSize)),
              logp: data.logp.gather(indices.slice(batchStartIndex, batchSize)),
            };

            // normalization adv
            const stats = {
              mean: batchData.adv.mean(),
              std: nj.std(batchData.adv.arraySync()),
            };
            batchData.adv = batchData.adv.sub(stats.mean).div(stats.std + 1e-8);
            // console.log('TCL ~ batchData:', batchData);

            batchStartIndex += batchSize;

            const grads = optimizer.computeGradients(() => {
              const { loss_pi, pi_info } = compute_loss_pi(batchData, epoch);
              // console.log('TCL ~ loss_pi:', loss_pi.arraySync());
              kls.push(pi_info.approx_kl);
              entropy = pi_info.entropy;
              clip_frac = pi_info.clip_frac;

              const loss_v = compute_loss_vf(batchData);
              loss_pi_ = loss_pi.arraySync() as number;
              loss_vf_ = loss_v.arraySync() as number;
              return loss_pi.add(loss_v.mul(configs.vf_coef)) as tf.Scalar;
              // return loss_v.mul(configs.vf_coef) as tf.Scalar;
            });
            // console.log('TCL ~ grads:', grads);
            if (kls[kls.length - 1] > 1.5 * target_kl) {
              // log.warn(
              //   `${configs.name} | Epoch ${epoch} | Reaching max kl ${kls[kls.length - 1]} / ${1.5 * target_kl}`
              // );
              log.warn(
                `${configs.name} | Early stopping at epoch ${epoch} batch ${batch}/${Math.floor(
                  totalSize / batchSize
                )} of optimizing policy due to reaching max kl ${kls[kls.length - 1]} / ${1.5 * target_kl}`
              );
              continueTraining = false;
              break;
            }

            // function clipGradNorm(grads: tf.Tensor, maxNorm: number) {
            //   const gradNorm = tf.norm(grads);
            //   const clipNorm = tf.maximum(gradNorm, maxNorm);
            //   const clippedGrads = grads.mul(clipNorm.div(gradNorm));
            //   return clippedGrads;
            // }

            const maxNorm = 0.5;
            const clippedGrads: tf.NamedTensorMap = {};
            // const norm = tf.norm(Object.values(grads.grads));
            const totalNorm = tf.norm(tf.stack(Object.values(grads.grads).map((grad) => tf.norm(grad))));
            // console.log('TCL ~ totalNorm:', totalNorm.arraySync());
            const clipCoeff = tf.minimum(tf.scalar(1.0), tf.scalar(maxNorm).div(totalNorm.add(1e-6)));
            // console.log('TCL ~ clipCoeff:', clipCoeff.arraySync());
            // const clippedGrads = tf.mul(grads.grads, scale);
            Object.keys(grads.grads).forEach((name) => {
              // clippedGrads[name] = clipGradNorm(grads[name], 0.5);
              clippedGrads[name] = tf.mul(grads.grads[name], clipCoeff);
            });

            // console.log(
            //   'TCL ~ grads:',
            //   Object.entries(grads.grads).map(([key, value]) => ({ key, value: value.arraySync() }))
            // );

            // for (const key in grads.grads) {
            //   if (Object.prototype.hasOwnProperty.call(grads.grads, key)) {
            //     const element = grads.grads[key];
            //     const clippedGrads = element.clipByValue(-0.01, 0.01);
            //     grads.grads[key] = clippedGrads;
            //     // console.log('TCL ~ element:', element);
            //   }
            // }

            optimizer.applyGradients(clippedGrads);
            batch++;
          }
          trained_pi_iters++;
          if (!continueTraining) {
            break;
          }
        }

        // let loss_pi = compute_loss_pi(data, 0).loss_pi.arraySync() as number;
        // let loss_vf = compute_loss_vf(data).arraySync() as number;

        const metrics = {
          kl: nj.mean(nj.array(kls)),
          entropy,
          clip_frac,
          trained_pi_iters,
          loss_pi: loss_pi_,
          loss_vf: loss_vf_,
          // delta_pi_loss: loss_pi - loss_pi_old,
          // delta_vf_loss: loss_vf - loss_vf_old,
        };

        return metrics;
      });
    };

    // const start_time = process.hrtime()[0] * 1e6 + process.hrtime()[1];
    let o = env.reset();
    let ep_ret = 0;
    let ep_len = 0;
    let ep_rets: number[] = [];
    let ep_rewards: number[] = [];
    let same_return_count = 0;
    for (let iteration = 0; iteration < configs.iterations; iteration++) {
      tf.tidy(() => {
        for (let t = 0; t < local_steps_per_iteration; t++) {
          let { a, v, logp_a } = this.ac.step(this.obsToTensor(o));
          // // @ts-ignore
          // console.log('TCL ~ o:', o.selection.data);
          // console.log('TCL ~ this.obsToTensor(o):', this.obsToTensor(o).arraySync());
          // console.log('TCL ~ a:', a.arraySync());
          // console.log('TCL ~ v:', v.arraySync());
          const action = this.actionToTensor(a) as number;
          // console.log('TCL ~ action:', action);
          // console.log('TCL ~ move:', actionToMove(action.selection.data[0], 9));
          const stepInfo = env.step(action);
          // // @ts-ignore
          // console.log('TCL ~ stepInfo:', stepInfo.observation.selection.data);
          // console.log('TCL ~ stepInfo.reward:', stepInfo.reward);
          // console.log('TCL ~ stepInfo.done:', stepInfo.done);
          const next_o = stepInfo.observation;

          let r = stepInfo.reward;
          ep_rewards.push(r);
          const d = stepInfo.done;
          ep_ret += r;
          ep_len += 1;

          if (d && stepInfo.info && stepInfo.info['TimeLimit.truncated'] && stepInfo.info['terminal_observation']) {
            const terminalObs = this.obsToTensor(stepInfo.info['terminal_observation']);
            const terminalValue = (this.ac.step(terminalObs).v.arraySync() as number[][])[0][0];
            r += configs.gamma * terminalValue;
          }

          if (env.actionSpace.meta.discrete) {
            a = a.reshape([-1, 1]);
          }

          buffer.store(
            np.tensorLikeToNdArray(this.obsToTensor(o)),
            np.tensorLikeToNdArray(a),
            r,
            np.tensorLikeToNdArray(v).get(0, 0),
            np.tensorLikeToNdArray(logp_a!).get(0, 0)
          );

          o = next_o;

          const timeout = ep_len === configs.max_ep_len;
          const terminal = d || timeout;
          const epoch_ended = t === local_steps_per_iteration - 1;
          if (terminal || epoch_ended) {
            if (epoch_ended && !terminal) {
              log.warn(`${configs.name} | Trajectory cut off by epoch at ${ep_len} steps`);
            }
            let v = 0;
            if (timeout || epoch_ended) {
              v = (this.ac.step(this.obsToTensor(o)).v.arraySync() as number[][])[0][0];
            }
            buffer.finishPath(v, d);
            if (terminal) {
              // store ep ret and eplen stuff
              ep_rets.push(ep_ret);
            }
            o = env.reset();
            ep_ret = 0;
            ep_len = 0;
          }
        }
      });

      // save model
      if (ep_rets.every((ret) => ret === ep_rets[0])) {
        same_return_count++;
        log.warn(`${configs.name} | Episode returns are the same for ${same_return_count} times`);
        if (same_return_count > 40) {
          if (configs.savePath) {
            log.warn(
              `${configs.name} | Early stopping at iteration ${iteration} due to all episode returns are the same`
            );
            console.log('saving model to', configs.savePath);
            await this.ac.save(configs.savePath);
          }
          break;
        }
      } else {
        same_return_count = 0;
      }

      // update actor critic
      const metrics = await update();
      const ep_rets_metrics = {
        min: nj.min(ep_rets),
        max: nj.max(ep_rets),
        mean: nj.mean(ep_rets),
        std: nj.std(ep_rets),
      };
      const ep_rewards_metrics = {
        min: nj.min(ep_rewards),
        max: nj.max(ep_rewards),
        mean: nj.mean(ep_rewards),
      };

      const msg = `${configs.name} | Iteration ${iteration} metrics: `;
      log.info(
        {
          ...metrics,
          ep_rets: ep_rets_metrics,
          ep_rewards: ep_rewards_metrics,
        },
        msg
      );
      // console.log('numTensors', tf.memory().numTensors);
      await configs.iterationCallback({
        iteration,
        ...metrics,
        ep_rets: ep_rets_metrics,
        ep_rewards: ep_rewards_metrics,
        t: iteration * local_steps_per_iteration,
      });

      ep_rets = [];
      ep_rewards = [];
    }
  }
}

import { Agent } from 'rl-ts/lib/Agent';
import { Environment } from '../../Environments';
import { Discrete, Space } from 'rl-ts/lib/Spaces';
import * as random from 'rl-ts/lib/utils/random';
import * as tf from '@tensorflow/tfjs';
import { PPOBuffer, PPOBufferComputations } from 'rl-ts/lib/Algos/ppo/buffer';
import { DeepPartial } from 'rl-ts/lib/utils/types';
import { deepMerge } from 'rl-ts/lib/utils/deep';
import * as np from 'rl-ts/lib/utils/np';
import nj, { NdArray, NjArray } from 'numjs';
import { ActorCritic } from 'rl-ts/lib/Models/ac';

type pi_info = {
  approx_kl: number;
  entropy: number;
  clip_frac: any;
};

export type TrainMetrics = {
  kl: number;
  entropy: number;
  clip_frac: number;
  trained_epoches: number;
  continueTraining: boolean;
  loss_pi: number;
  loss_vf: number;
};

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
    loss_pi: number;
    loss_vf: number;
    ep_rets: {
      bestEver: number;
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
    fps: number;
    fps_rollout: number;
    fps_train: number;
    duration_rollout: number;
    duration_train: number;
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

  private buffer: PPOBuffer;
  public trainConfigs: PPOTrainConfigs;

  // minibatch
  private miniBatchData: PPOBufferComputations;
  private miniBatchIndices: tf.Tensor1D;

  // stats
  private rollOutDuration = 0;
  private trainDuration = 0;
  private ep_ret: number;
  private ep_len: number;
  private ep_rets: number[];
  private ep_rewards: number[];
  private bestEver: number;

  // debug
  private debug: boolean;

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
    this.debug = false;
  }

  /**
   * Select action using the current policy network. By default selects the action by feeding the observation through the network
   * then return the argmax of the outputs
   * @param observation - observation to select action off of
   * @returns action
   */
  public act(observation: Observation, acitonMask: nj.NdArray<number>): Action {
    return np.tensorLikeToNdArray(
      this.actionToTensor(this.ac.act(this.obsToTensor(observation), np.toTensor(acitonMask)))
    );
  }

  public setupTrain(trainConfigs: Partial<PPOTrainConfigs>, debug: boolean = true) {
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
    this.trainConfigs = configs;

    this.debug = debug;
    if (this.debug) {
      console.log('merged configs:', this.trainConfigs);
    }

    // TODO do some seeding things
    random.seed(configs.seed);
    // TODO: seed tensorflow if possible

    const obs_dim = this.env.observationSpace.shape;

    let buffer_action_dim = this.env.actionSpace.shape;
    let mask_dim = this.env.actionSpace.shape;
    if (this.env.actionSpace instanceof Discrete) {
      buffer_action_dim = [1];
    }

    this.buffer = new PPOBuffer({
      gamma: configs.gamma,
      lam: configs.lam,
      actDim: buffer_action_dim,
      maskDim: mask_dim,
      obsDim: obs_dim,
      size: configs.steps_per_iteration,
    });

    this.ep_ret = 0;
    this.ep_len = 0;
    this.ep_rets = [];
    this.ep_rewards = [];
    this.bestEver = 0;
    this.env.reset();
  }

  public train(trainConfigs: Partial<PPOTrainConfigs>) {
    this.setupTrain(trainConfigs);
    for (let iteration = 0; iteration < this.trainConfigs.iterations; iteration++) {
      const startTime = Date.now();
      let start = 0;
      while (start < this.trainConfigs.steps_per_iteration) {
        this.collectRollout(start, start + this.trainConfigs.batch_size);
        start += this.trainConfigs.batch_size;
      }
      // update actor critic
      this.prepareMiniBatch();
      const maxBatch = this.getMaxBatch();
      let metrics: TrainMetrics;
      let continueTraining = true;
      let i = 0;
      for (; i < this.trainConfigs.n_epochs; i++) {
        if (!continueTraining) {
          break;
        }
        let batch = 0;
        while (batch < maxBatch && continueTraining) {
          console.log('Epoch :', i, 'Batch:', batch);
          metrics = this.update(batch);
          continueTraining = metrics.continueTraining;
          batch++;
        }
      }
      // collect metrics
      metrics.trained_epoches = i;
      this.collectMetrics(startTime, iteration, metrics);
    }
  }

  public collectMetrics(startTime: number, iteration: number, metrics: TrainMetrics) {
    const configs = this.trainConfigs;
    const totalDuration = (Date.now() - startTime) / 1000;
    const ep_max = nj.max(this.ep_rets);
    if (ep_max > this.bestEver) {
      this.bestEver = ep_max;
    }

    const perfMetrics = {
      t: iteration * configs.steps_per_iteration,
      fps: configs.steps_per_iteration / totalDuration,
      duration_rollout: this.rollOutDuration,
      duration_train: this.trainDuration,
      fps_rollout: configs.steps_per_iteration / this.rollOutDuration,
      fps_train: (configs.steps_per_iteration * metrics.trained_epoches) / this.trainDuration,
    };

    const ep_rets_metrics = {
      min: nj.min(this.ep_rets),
      max: ep_max,
      mean: nj.mean(this.ep_rets),
      bestEver: this.bestEver,
      std: nj.std(this.ep_rets),
    };
    const ep_rewards_metrics = {
      min: nj.min(this.ep_rewards),
      max: nj.max(this.ep_rewards),
      mean: nj.mean(this.ep_rewards),
    };

    const msg = `${configs.name} | Iteration ${iteration} metrics: `;
    if (this.debug) {
      console.log(msg, {
        ...metrics,
        ep_rets: ep_rets_metrics,
        ep_rewards: ep_rewards_metrics,
        ...perfMetrics,
      });
    }
    // console.log('numTensors', tf.memory().numTensors);
    configs.iterationCallback({
      iteration,
      ...metrics,
      ep_rets: ep_rets_metrics,
      ep_rewards: ep_rewards_metrics,
      ...perfMetrics,
    });

    this.ep_rets = [];
    this.ep_rewards = [];
  }

  public collectRollout(start: number, end: number) {
    const rolloutStartTime = Date.now();
    tf.tidy(() => {
      const env = this.env;
      const configs = this.trainConfigs;
      let invalidMask = env.invalidActionMask();
      let o = env.state as Observation;
      for (let t = start; t < configs.steps_per_iteration && t < end; t++) {
        let { a, v, logp_a } = this.ac.step(this.obsToTensor(o), np.toTensor(invalidMask));

        const action = this.actionToTensor(a) as number;
        const stepInfo = env.step(action);
        const next_o = stepInfo.observation;

        let r = stepInfo.reward;
        this.ep_rewards.push(r);
        const d = stepInfo.done;
        this.ep_ret += r;
        this.ep_len += 1;

        if (d && stepInfo.info && stepInfo.info['TimeLimit.truncated'] && stepInfo.info['terminal_observation']) {
          const terminalObs = this.obsToTensor(stepInfo.info['terminal_observation']);
          const terminalValue = (this.ac.step(terminalObs, np.toTensor(invalidMask)).v.arraySync() as number[][])[0][0];
          r += configs.gamma * terminalValue;
        }

        if (env.actionSpace.meta.discrete) {
          a = a.reshape([-1, 1]);
        }

        this.buffer.store(
          np.tensorLikeToNdArray(this.obsToTensor(o)),
          np.tensorLikeToNdArray(a),
          r,
          np.tensorLikeToNdArray(v).get(0, 0),
          np.tensorLikeToNdArray(logp_a!).get(0, 0),
          invalidMask.reshape(1, -1)
        );

        o = next_o;
        invalidMask = env.invalidActionMask();

        const timeout = this.ep_len === configs.max_ep_len;
        const terminal = d || timeout;
        const epoch_ended = t === configs.steps_per_iteration - 1;
        if (terminal || epoch_ended) {
          if (epoch_ended && !terminal) {
            if (this.debug) {
              console.log(`${configs.name} | Trajectory cut off by epoch at ${this.ep_len} steps`);
            }
          }
          let v = 0;
          if (timeout || epoch_ended) {
            v = (this.ac.step(this.obsToTensor(o), np.toTensor(invalidMask)).v.arraySync() as number[][])[0][0];
          }
          this.buffer.finishPath(v, d);
          if (terminal) {
            // store ep ret and eplen stuff
            this.ep_rets.push(this.ep_ret);
          }
          o = env.reset();
          invalidMask = env.invalidActionMask();
          this.ep_ret = 0;
          this.ep_len = 0;
        }
      }
    });

    this.rollOutDuration = (Date.now() - rolloutStartTime) / 1000;
  }

  public prepareMiniBatch() {
    const configs = this.trainConfigs;
    const totalSize = configs.steps_per_iteration;
    this.miniBatchData = this.buffer.get();
    this.miniBatchIndices = tf.tensor1d(Array.from(tf.util.createShuffledIndices(totalSize)), 'int32');
  }

  public getMaxBatch() {
    const configs = this.trainConfigs;
    return Math.floor(configs.steps_per_iteration / configs.batch_size);
  }

  public update(batch: number) {
    const configs = this.trainConfigs;
    const updateStartTime = Date.now();
    const { clip_ratio, optimizer, target_kl } = configs;

    const compute_loss_pi = (data: PPOBufferComputations): { loss_pi: tf.Tensor; pi_info: pi_info } => {
      let { obs, act, adv, mask } = data;
      return tf.tidy(() => {
        const logp_old = data.logp.expandDims(-1);
        const adv_e = adv.expandDims(-1);
        const { pi, logp_a } = this.ac.pi.apply(obs, act, mask);

        const ratio = logp_a!.sub(logp_old).exp();

        const clip_adv = ratio.clipByValue(1 - clip_ratio, 1 + clip_ratio).mul(adv_e);

        const adv_ratio = ratio.mul(adv_e);

        const ratio_and_clip_adv = tf.stack([adv_ratio, clip_adv]);

        const loss_pi = ratio_and_clip_adv.min(0).mean().mul(-1);

        // from stablebaseline3
        const log_ratio = logp_a!.sub(logp_old);
        const approx_kl = log_ratio.exp().sub(1).sub(log_ratio).mean().arraySync() as number;

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
        return predict.sub(ret).pow(2).mean();
      });
    };

    return tf.tidy(() => {
      const totalSize = configs.steps_per_iteration;
      const batchSize = configs.batch_size;

      const data = this.miniBatchData;
      const indices = this.miniBatchIndices;
      if (!data || !indices) {
        throw new Error('Mini batch data or indices not found');
      }

      let kls: number[] = [];
      let entropy = 0;
      let clip_frac = 0;
      let loss_pi_ = 0;
      let loss_vf_ = 0;

      let continueTraining = true;

      let batchStartIndex = batch * batchSize;
      let maxBatch = this.getMaxBatch();
      if (batch < maxBatch) {
        const batchData = {
          obs: data.obs.gather(indices.slice(batchStartIndex, batchSize)),
          act: data.act.gather(indices.slice(batchStartIndex, batchSize)),
          adv: data.adv.gather(indices.slice(batchStartIndex, batchSize)),
          ret: data.ret.gather(indices.slice(batchStartIndex, batchSize)),
          logp: data.logp.gather(indices.slice(batchStartIndex, batchSize)),
          mask: data.mask.gather(indices.slice(batchStartIndex, batchSize)),
        };

        // normalization adv
        const stats = {
          mean: batchData.adv.mean(),
          std: nj.std(batchData.adv.arraySync()),
        };
        batchData.adv = batchData.adv.sub(stats.mean).div(stats.std + 1e-8);

        const grads = optimizer.computeGradients(() => {
          const { loss_pi, pi_info } = compute_loss_pi(batchData);
          kls.push(pi_info.approx_kl);
          entropy = pi_info.entropy;
          clip_frac = pi_info.clip_frac;

          const loss_v = compute_loss_vf(batchData);
          loss_pi_ = loss_pi.arraySync() as number;
          loss_vf_ = loss_v.arraySync() as number;
          return loss_pi.add(loss_v.mul(configs.vf_coef)) as tf.Scalar;
        });
        if (kls[kls.length - 1] > 1.5 * target_kl) {
          if (this.debug) {
            console.log(
              `${configs.name} | Early stopping at batch ${batch}/${Math.floor(
                totalSize / batchSize
              )} of optimizing policy due to reaching max kl ${kls[kls.length - 1]} / ${1.5 * target_kl}`
            );
          }
          continueTraining = false;
        }

        const maxNorm = 0.5;
        const clippedGrads: tf.NamedTensorMap = {};
        const totalNorm = tf.norm(tf.stack(Object.values(grads.grads).map((grad) => tf.norm(grad))));
        const clipCoeff = tf.minimum(tf.scalar(1.0), tf.scalar(maxNorm).div(totalNorm.add(1e-6)));
        Object.keys(grads.grads).forEach((name) => {
          clippedGrads[name] = tf.mul(grads.grads[name], clipCoeff);
        });

        optimizer.applyGradients(clippedGrads);
        batch++;
      } else {
        throw new Error(`batch ${batch} is out of range ${maxBatch}`);
      }

      this.trainDuration = (Date.now() - updateStartTime) / 1000;

      const metrics = {
        kl: nj.mean(nj.array(kls)),
        entropy,
        clip_frac,
        continueTraining,
        trained_epoches: 1,
        loss_pi: loss_pi_,
        loss_vf: loss_vf_,
      };

      return metrics;
    });
  }
}

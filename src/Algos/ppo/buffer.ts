import * as tf from '@tensorflow/tfjs';
import nj, { NdArray } from 'numjs';
import * as np from 'rl-ts/lib/utils/np';
import * as core from 'rl-ts/lib/Algos/utils/core';
import { Tensor1D } from '@tensorflow/tfjs';
export interface PPOBufferConfigs {
  obsDim: number[];
  actDim: number[];
  maskDim: number[];
  /** Buffer Size */
  size: number;
  gamma?: number;
  lam?: number;
}
export interface PPOBufferComputations {
  obs: tf.Tensor;
  mask: tf.Tensor;
  act: tf.Tensor1D;
  ret: tf.Tensor1D;
  adv: tf.Tensor1D;
  logp: tf.Tensor1D;
}

/**
 * Buffer for PPO for storing trajectories experienced by a PPO agent.
 * Uses Generalized Advantage Estimation (GAE-Lambda) to calculate advantages of state-action pairs
 */
export class PPOBuffer {
  /** Observations buffer */
  public obsBuf: NdArray;
  /** Actions buffer */
  public actBuf: NdArray;
  /** Masks buffer */
  public maskBuf: NdArray;
  /** Advantage estimates buffer */
  public advBuf: NdArray;
  /** Rewards buffer */
  public rewBuf: NdArray;
  /** Returns buffer */
  public retBuf: NdArray;
  /** Values buffer */
  public valBuf: NdArray;
  /** Log probabilities buffer */
  public logpBuf: NdArray;

  private ptr = 0;
  private pathStartIdx = 0;
  private maxSize = -1;

  public gamma = 0.99;
  public lam = 0.97;

  constructor(public configs: PPOBufferConfigs) {
    if (configs.gamma !== undefined) {
      this.gamma = configs.gamma;
    }
    if (configs.lam !== undefined) {
      this.lam = configs.lam;
    }

    this.obsBuf = nj.zeros([configs.size, ...configs.obsDim], 'float32');
    this.maskBuf = nj.zeros([configs.size, ...configs.maskDim], 'float32');
    this.actBuf = nj.zeros([configs.size, ...configs.actDim], 'float32');
    this.advBuf = nj.zeros([configs.size], 'float32');
    this.rewBuf = nj.zeros([configs.size], 'float32');
    this.retBuf = nj.zeros([configs.size], 'float32');
    this.valBuf = nj.zeros([configs.size], 'float32');
    this.logpBuf = nj.zeros([configs.size], 'float32');
    this.maxSize = configs.size;
  }
  public store(obs: NdArray, act: NdArray, rew: number, val: number, logp: number, mask: NdArray) {
    if (this.ptr >= this.maxSize) throw new Error('Experience Buffer has no room');
    const slice = [this.ptr, this.ptr + 1];
    this.obsBuf.slice(slice).assign(obs, false);
    // console.log('TCL ~ mask:', mask);
    // console.log('TCL ~ this.maskBuf.slice(slice):', this.maskBuf.slice(slice));
    this.maskBuf.slice(slice).assign(mask, false);
    // console.log('TCL ~ act:', act);
    // console.log('TCL ~ this.actBuf.slice(slice).shape:', this.actBuf.slice(slice).shape);
    this.actBuf.slice(slice).assign(act, false);
    this.rewBuf.set(this.ptr, rew);
    this.valBuf.set(this.ptr, val);
    this.logpBuf.set(this.ptr, logp);
    this.ptr += 1;
  }

  public finishPath(lastVal = 0, done: boolean) {
    // const path_slice = [this.pathStartIdx, this.ptr];
    // const rews = np.push(this.rewBuf.slice(path_slice), lastVal);
    // const vals = np.push(this.valBuf.slice(path_slice), lastVal);
    // // GAE Lambda Advantage = sum (gamma lambda)^h delta_{t+h, 0}
    // // compute delta_{t+h, 0}
    // // replicates deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    // const deltas = rews
    //   .slice([0, -1])
    //   .add(vals.slice(1).multiply(this.gamma))
    //   .subtract(vals.slice([0, -1]));

    // // compute GAE-Lambda advantage, assign in place.
    // this.advBuf.slice(path_slice).assign(core.discountCumSum(deltas, this.gamma * this.lam), false);

    // // compute ret with td lambda
    // this.retBuf.slice(path_slice).assign(this.advBuf.slice(path_slice).add(this.valBuf.slice(path_slice)), false);

    const path_slice = [this.pathStartIdx, this.ptr];
    let lastGaeLam = 0;
    for (let t = this.ptr - 1; t >= this.pathStartIdx; t--) {
      const nextVal = t === this.ptr - 1 ? lastVal : this.valBuf.get(t + 1);
      const nextNonTerminal = t === this.ptr - 1 ? (done ? 0 : 1) : 1;
      const delta = this.rewBuf.get(t) + this.gamma * nextVal * nextNonTerminal - this.valBuf.get(t);
      lastGaeLam = delta + this.gamma * this.lam * nextNonTerminal * lastGaeLam;
      this.advBuf.set(t, lastGaeLam);
    }
    this.retBuf.slice(path_slice).assign(this.advBuf.slice(path_slice).add(this.valBuf.slice(path_slice)), false);
    // for (let t = this.ptr - 1; t >= this.pathStartIdx; t--) {
    //   console.log(
    //     `t: ${t}, done: ${t === this.ptr - 1 ? (done ? 1 : 0) : 0}, reward: ${this.rewBuf.get(
    //       t
    //     )}, value: ${this.valBuf.get(t)}, adv: ${this.advBuf.get(t)}, ret: ${this.retBuf.get(t)}`
    //   );
    // }
    this.pathStartIdx = this.ptr;
  }

  public get(): PPOBufferComputations {
    if (this.ptr !== this.maxSize) {
      throw new Error("Buffer isn't full yet!");
    }
    this.pathStartIdx = 0;
    this.ptr = 0;

    // move to tensors for use by update method and nicer functions
    // let advBuf = np.toTensor(this.advBuf);

    // // normalization trick
    // const stats = await ct.statisticsScalar(advBuf, { max: true, min: true }, true);
    // advBuf = advBuf.sub(stats.mean).div(stats.std);
    // this.advBuf = await np.fromTensor(advBuf);
    return {
      obs: np.toTensor(this.obsBuf),
      mask: np.toTensor(this.maskBuf),
      act: np.toTensor(this.actBuf) as Tensor1D,
      ret: np.toTensor(this.retBuf) as Tensor1D,
      adv: np.toTensor(this.advBuf) as Tensor1D,
      logp: np.toTensor(this.logpBuf) as Tensor1D,
    };
  }
}

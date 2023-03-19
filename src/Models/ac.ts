import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { readFileSync } from 'fs';
import { Box, Discrete, Shape, Space } from 'rl-ts/lib/Spaces';
import { Distribution } from 'rl-ts/lib/utils/Distributions';
import { Normal } from 'rl-ts/lib/utils/Distributions/normal';
import { Categorical } from '../utils/Distributions/categorical';

let global_gaussian_actor_log_std_id = 0;

/** Create a MLP model */
export const createMLP = (
  in_shape: Shape,
  out_dim: number,
  hidden_sizes: number[],
  activation: ActivationIdentifier,
  gain: number,
  conv: boolean,
  name?: string
) => {
  // const input = tf.input({ shape: [in_dim, in_dim, 6] });
  // let layer = tf.layers
  //   .conv2d({
  //     filters: 64,
  //     kernelSize: 2,
  //     strides: 1,
  //     activation: 'relu',
  //   })
  //   .apply(input);
  // layer = tf.layers.flatten().apply(layer);
  // layer = tf.layers.dense({ units: hidden_sizes[0], activation }).apply(layer);
  const input = tf.input({ shape: in_shape });
  let layer: tf.SymbolicTensor | tf.SymbolicTensor[] | tf.Tensor | tf.Tensor[];
  if (in_shape.length === 1) {
    layer = tf.layers.dense({ units: hidden_sizes[0], activation }).apply(input);
  } else if (conv) {
    layer = tf.layers
      .conv2d({
        filters: 64,
        kernelSize: 3,
        strides: 1,
        activation: 'relu',
      })
      .apply(input);
    layer = tf.layers.flatten().apply(layer);
    layer = tf.layers.dense({ units: hidden_sizes[0], activation }).apply(layer);
  } else {
    layer = tf.layers.flatten().apply(input);
    layer = tf.layers.dense({ units: hidden_sizes[0], activation }).apply(layer);
  }

  for (const size of hidden_sizes.slice(1)) {
    layer = tf.layers
      .dense({
        units: size,
        activation,
        // kernelConstraint: tf.constraints.maxNorm({ maxValue: 0.5, axis: 0 }),
        // biasConstraint: tf.constraints.maxNorm({ maxValue: 0.5, axis: 0 }),
        kernelInitializer: tf.initializers.orthogonal({ gain: tf.sqrt(2).arraySync() as number }),
        biasInitializer: tf.initializers.zeros(),
      })
      .apply(layer);
  }
  layer = tf.layers
    .dense({
      units: out_dim,
      activation: 'linear',
      // kernelConstraint: tf.constraints.maxNorm({ maxValue: 0.5, axis: 0 }),
      // biasConstraint: tf.constraints.maxNorm({ maxValue: 0.5, axis: 0 }),
      kernelInitializer: tf.initializers.orthogonal({ gain }),
      biasInitializer: tf.initializers.zeros(),
    })
    .apply(layer);
  return tf.model({ inputs: input, outputs: layer as SymbolicTensor, name });
};

export abstract class Actor<Observation extends tf.Tensor> {
  public actor_net: tf.LayersModel;
  abstract _distribution(obs: Observation): Distribution;
  abstract _log_prob_from_distribution(pi: Distribution, act: tf.Tensor): tf.Tensor;
  abstract apply(obs: Observation, act: tf.Tensor): { pi: Distribution; logp_a: tf.Tensor | null };
}
export abstract class Critic<Observation extends tf.Tensor> {
  abstract apply(obs: Observation): tf.Tensor;
}
export abstract class ActorCritic<Observation extends tf.Tensor> {
  abstract pi: Actor<Observation>;
  abstract v: Critic<Observation>;
  abstract step(obs: Observation): {
    a: tf.Tensor;
    logp_a: tf.Tensor | null;
    v: tf.Tensor;
  };
  abstract act(obs: Observation): tf.Tensor;
  abstract save(path: string): Promise<void>;
  abstract load(path: string): Promise<void>;
}

export abstract class ActorBase<Observation extends tf.Tensor> extends Actor<Observation> {
  apply(obs: Observation, act: tf.Tensor | null) {
    const pi = this._distribution(obs);
    let logp_a = null;
    if (act !== null) {
      logp_a = this._log_prob_from_distribution(pi, act);
    }
    return {
      pi,
      logp_a,
    };
  }
}

export class MLPGaussianActor extends ActorBase<tf.Tensor> {
  public mu_net: tf.LayersModel;
  public log_std: tf.Variable;
  public mu: tf.Variable;
  constructor(
    obs_shape: Shape,
    public act_dim: number,
    hidden_sizes: number[],
    activation: ActivationIdentifier,
    conv: boolean
  ) {
    super();
    this.log_std = tf.variable(
      tf.zeros([act_dim], 'float32'),
      true,
      `gaussian_actor_log_std_${global_gaussian_actor_log_std_id++}`
    );
    this.mu_net = createMLP(obs_shape, act_dim, hidden_sizes, activation, 0.01, conv, 'MLP Gaussian Actor');
    this.mu = tf.variable(tf.tensor(0));
    this.actor_net = this.mu_net;
  }
  _distribution(obs: tf.Tensor) {
    const mu = this.mu_net.apply(obs) as tf.Tensor; // [B, act_dim]
    const batch_size = mu.shape[0];
    const std = tf.exp(this.log_std).expandDims(0).tile([batch_size, 1]); // from [act_dim] shaped to [B, act_dim]
    return new Normal(mu, std);
  }
  _log_prob_from_distribution(pi: Normal, act: tf.Tensor): tf.Tensor {
    return pi.logProb(act).sum(-1);
  }
}

export class MLPCategoricalActor extends ActorBase<tf.Tensor> {
  public logits_net: tf.LayersModel;
  constructor(
    obs_shape: Shape,
    act_dim: number,
    hidden_sizes: number[],
    activation: ActivationIdentifier,
    conv: boolean
  ) {
    super();
    this.logits_net = createMLP(obs_shape, act_dim, hidden_sizes, activation, 0.01, conv);
    this.actor_net = this.logits_net;
  }
  _distribution(obs: tf.Tensor): Distribution {
    const logits = this.logits_net.apply(obs) as tf.Tensor;
    return new Categorical(logits);
  }
  _log_prob_from_distribution(pi: Distribution, act: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
    const prob = pi.logProb(act);
    return prob;
  }
}

export class MLPCritic extends Critic<tf.Tensor> {
  public v_net: tf.LayersModel;
  constructor(obs_shape: Shape, hidden_sizes: number[], activation: ActivationIdentifier, conv: boolean) {
    super();
    this.v_net = createMLP(obs_shape, 1, hidden_sizes, activation, 1, conv, 'MLP Critic');
  }
  apply(obs: tf.Tensor) {
    // TODO check need squeeze?
    return this.v_net.apply(obs) as tf.Tensor;
  }
}

export class MLPActorCritic extends ActorCritic<tf.Tensor> {
  public pi: ActorBase<tf.Tensor>;
  public v: MLPCritic;
  constructor(
    public observationSpace: Space<any>,
    public actionSpace: Space<any>,
    hidden_sizes: number[],
    activation: ActivationIdentifier = 'tanh',
    conv: boolean
  ) {
    super();
    let act_dim = actionSpace.shape[0];
    // if (actionSpace instanceof Discrete) {
    //   act_dim = 1;
    // }
    if (actionSpace instanceof Box) {
      this.pi = new MLPGaussianActor(observationSpace.shape, act_dim, hidden_sizes, activation, conv);
    } else if (actionSpace instanceof Discrete) {
      this.pi = new MLPCategoricalActor(observationSpace.shape, act_dim, hidden_sizes, activation, conv);
      // this.pi = new MLPGaussianActor(observationSpace.shape, act_dim, hidden_sizes, activation, conv);
    } else {
      throw new Error('This action space is not supported');
    }
    this.v = new MLPCritic(observationSpace.shape, hidden_sizes, activation, conv);
  }
  step(obs: tf.Tensor) {
    const pi = this.pi._distribution(obs);
    const a = pi.sample();
    // console.log('TCL ~ a:', a);
    const logp_a = this.pi._log_prob_from_distribution(pi, a);
    const v = this.v.apply(obs);
    return {
      a,
      logp_a,
      v,
    };
  }
  act(obs: tf.Tensor) {
    return this.step(obs).a;
  }

  print() {
    this.pi.actor_net.summary();
    this.v.v_net.summary();
  }

  async save(path: string) {
    await this.pi.actor_net.save(`file://${path}-pi`);
    await this.v.v_net.save(`file://${path}-v`);
  }

  async load(path: string) {
    const piFsPath = `${path}-pi/model.json`;
    const piContent = JSON.parse(readFileSync(piFsPath).toString());
    console.log('load pi model:', piContent);
    const mu_net = await tf.loadLayersModel(`file://${path}-pi/model.json`);
    this.pi.actor_net = mu_net;

    const vfFsPath = `${path}-v/model.json`;
    const vfFsContent = JSON.parse(readFileSync(vfFsPath).toString());
    console.log('load vf model:', vfFsContent);
    const v_net = await tf.loadLayersModel(`file://${path}-v/model.json`);
    this.v.v_net = v_net;
  }
}

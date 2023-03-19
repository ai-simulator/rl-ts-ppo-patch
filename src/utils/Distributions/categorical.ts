import * as tf from '@tensorflow/tfjs';
import nj, { NdArray } from 'ndarray';
import * as np from 'rl-ts/lib/utils/np';
import { tensorLikeToNdArray, tensorLikeToTensor } from 'rl-ts/lib/utils/np';
import { Distribution } from '.';

/**
 * A categorical distribution
 */
export class Categorical extends Distribution {
  public logits: NdArray;
  public tf_logits: tf.Tensor;

  constructor(init_tf_logits: tf.Tensor) {
    super(init_tf_logits.shape, 'Categorical');
    const tf_logits = init_tf_logits.sub(init_tf_logits.logSumExp(-1, true));
    this.logits = tensorLikeToNdArray(tf_logits);
    this.tf_logits = tf_logits;
    // console.log('TCL ~ this.tf_logits:', this.tf_logits.arraySync());
    // console.log('TCL ~ this.logits:', this.logits);
  }
  sample(): tf.Tensor {
    let logits = this.logits_parameter();
    // console.log('TCL ~ this.logits:', this.logits.shape);
    const sample = tf.buffer([logits.shape[0]], 'float32');
    // console.log('TCL ~ this.logits:', this.logits);
    // console.log('TCL ~ this.logits:', this.logits.arraySync());
    const logits_2d = tf.reshape(logits, [-1, this._num_categories(logits)]);
    // console.log('TCL ~ logits_2d:', logits_2d);
    // console.log('TCL ~ logits_2d:', logits_2d.arraySync());
    for (let i = 0; i < sample.size; i++) {
      const loc = sample.indexToLoc(i);
      const value = tf.multinomial(logits_2d as tf.Tensor2D, 1);
      // console.log('TCL ~ value:', value);
      // console.log('TCL ~ value:', value.arraySync());
      sample.set(value.dataSync()[0], ...loc);
    }
    // console.log('TCL ~ sample.toTensor():', sample.toTensor());
    // console.log('TCL ~ sample.toTensor():', sample.toTensor().arraySync());
    // console.log('TCL ~ sample.toTensor().expandDims(0):', sample.toTensor().expandDims(0).arraySync());
    return sample.toTensor();
  }

  private _num_categories(logits: tf.Tensor): number {
    return logits.shape[logits.shape.length - 1];
  }

  logProb(value: tf.Tensor): tf.Tensor {
    // let logits = this.logits_parameter();
    value = tf.cast(value, 'int32');
    // const boardcastShape = tf.broadcastArgs(value, logits).shape;
    // value = tf.broadcastTo(value, boardcastShape);
    // const log_pmf = tf.broadcastTo(logits, boardcastShape);
    // value = value.expandDims(-1);
    // console.log('TCL ~ logits:', logits);
    // console.log('TCL ~ logits:', logits);
    // console.log('TCL ~ logits:', logits.arraySync());
    // console.log('TCL ~ value:', value);
    // console.log('TCL ~ value:', value.arraySync());
    // console.log('TCL ~ logits.squeeze([-1]):', logits.squeeze());
    // console.log('TCL ~ logits.gather(value):', logits.squeeze().gather(value).arraySync());
    // console.log('TCL ~ logits:', logits);
    // console.log('TCL ~ logits:', logits.arraySync());
    // console.log('TCL ~ value:', value);
    // console.log('TCL ~ value:', value.arraySync());
    // console.log('TCL ~ this.tf_logits:', this.tf_logits);
    // console.log('TCL ~ this.tf_logits:', this.tf_logits.arraySync());
    const logProb = tf.gather(this.tf_logits, value, 1, 1);
    // const logProb = tf.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [11, 1]);
    // console.log('TCL ~ emtpy:', emtpy);
    // console.log('TCL ~ logProb:', logProb);
    // console.log('TCL ~ logProb:', logProb.arraySync());
    return logProb;
  }
  logits_parameter() {
    // @ts-ignore
    return tensorLikeToTensor(this.logits);
  }
  entropy() {
    const probs = tf.softmax(this.logits_parameter());
    const p_log_p = tf.mul(this.logits_parameter(), probs);
    // console.log('TCL ~ tf.neg(p_log_p.sum(-1)):', tf.neg(p_log_p.sum(-1)));
    return tf.neg(p_log_p.sum(-1));
  }
}

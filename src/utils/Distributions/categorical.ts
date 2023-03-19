import * as tf from '@tensorflow/tfjs';
import * as np from 'rl-ts/lib/utils/np';
import { NdArray } from 'numjs';
import { Distribution } from '.';

/**
 * A categorical distribution
 */
export class Categorical extends Distribution {
  public logits: tf.Tensor;

  constructor(public tf_logits: tf.Tensor) {
    super(tf_logits.shape, 'Categorical');
    tf_logits = tf_logits.sub(tf_logits.logSumExp(-1, true));
    this.logits = tf_logits;
  }
  sample(): tf.Tensor {
    const sample = tf.buffer([1], 'float32');
    // console.log('TCL ~ this.logits:', this.logits);
    // console.log('TCL ~ this.logits:', this.logits.arraySync());
    const logits_2d = tf.reshape(this.logits, [-1, this._num_categories(this.logits)]);
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
    return sample.toTensor().expandDims(0);
  }

  private _num_categories(logits: tf.Tensor): number {
    return logits.shape[logits.shape.length - 1];
  }

  logProb(value: tf.Tensor): tf.Tensor {
    let logits = tf.softmax(this.logits_parameter());
    value = tf.cast(value, 'int32');
    // const boardcastShape = tf.broadcastArgs(value, logits).shape;
    // value = tf.broadcastTo(value, boardcastShape);
    // const log_pmf = tf.broadcastTo(logits, boardcastShape);
    // value = value.squeeze();
    // console.log('TCL ~ logits:', logits);
    // console.log('TCL ~ logits:', logits.arraySync());
    // console.log('TCL ~ value:', value.arraySync());
    // console.log('TCL ~ logits.squeeze([-1]):', logits.squeeze());
    // console.log('TCL ~ logits.gather(value):', logits.squeeze().gather(value).arraySync());
    // console.log('TCL ~ logits:', logits);
    // console.log('TCL ~ logits:', logits.arraySync());
    // console.log('TCL ~ value:', value);
    // console.log('TCL ~ value:', value.arraySync());
    const logProb = logits.gather(value, 1);
    // console.log('TCL ~ logProb:', logProb);
    // console.log('TCL ~ logProb:', logProb.arraySync());
    return logProb;
  }
  logits_parameter() {
    return this.logits;
  }
  entropy() {
    const probs = tf.softmax(this.logits);
    const p_log_p = tf.mul(this.logits, probs);
    return tf.neg(p_log_p.sum(-1));
  }
}

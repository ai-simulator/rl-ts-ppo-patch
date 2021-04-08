import { Shape, Space } from '.';
import { randomRange } from '../utils/random';

/**
 * discrete space with values from set {0, 1, ..., n - 1}
 */
export class Discrete extends Space<number> {
  constructor(public n: number) {
    super([]);
  }
  sample(): number {
    return Math.floor(randomRange(this.rng, 0, this.n));
  }
  contains(x: number): boolean {
    return x < this.n && x >= 0;
  }
  to_jsonable(sample_n: number[]) {
    return sample_n;
  }
  from_jsonable(sample_n: number[]) {
    return sample_n;
  }
}
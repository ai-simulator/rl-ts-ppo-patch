import cwise from 'cwise';
import { dupe } from './dup';

// implement _pack here for hermes issue
// https://github.com/facebook/hermes/issues/954
var do_unpack = cwise({
  args: ['array', 'scalar', 'index'],
  body: function unpackCwise(arr, a, idx) {
    'show source';
    var v = a,
      i;
    for (i = 0; i < idx.length - 1; ++i) {
      v = v[idx[i]];
    }
    v[idx[idx.length - 1]] = arr;
  },
});
export function _unpack(arr) {
  var result = dupe(arr.shape);
  do_unpack(arr, result);
  return result;
}

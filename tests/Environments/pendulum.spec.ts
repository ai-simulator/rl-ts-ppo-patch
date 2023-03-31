import { expect } from 'chai';
import * as RL from '../../src';
import { Examples } from '../../src/Environments';
describe('Test Pendulum', () => {
  it('should run', async () => {
    const env = new Examples.Pendulum();
    env.maxEpisodeSteps = 10;
    let state = env.reset();
    while (true) {
      const action = env.actionSpace.sample();
      const { observation, done } = env.step(action);
      state = observation;
      expect(state.shape).to.eql([3]);
      if (done) break;
    }
  });
});

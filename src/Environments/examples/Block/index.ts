import { Environment, RenderModes } from '../../../Environments';
import { Box, Discrete } from '../../../Spaces';
import nj, { NdArray } from 'numjs';
import { fromTensorSync, tensorLikeToNdArray, toTensor } from '../../../utils/np';
import { Game } from './model/game';
import * as tf from '@tensorflow/tfjs';
import { Move } from './model/move';

export type State = NdArray<number>;
export type Observation = NdArray<number>;
export type Action = number;
export type ActionSpace = Discrete;
export type ObservationSpace = Box;
export type Reward = number;

export interface BlockConfigs {
  game: Game;
}

export function moveToAction(move: Move, width: number) {
  return move[1] * width + move[0];
}

export function actionToMove(action: number, width: number): Move {
  return [action % width, Math.floor(action / width)];
}

/**
 * Block environment
 */
export class Block extends Environment<Box, ActionSpace, Observation, State, Action, Reward> {
  public observationSpace: ObservationSpace;
  public actionSpace: Discrete;
  public game: Game;
  public state: NdArray<number>;

  public timestep = 0;
  public globalTimestep = 0;

  constructor(configs: BlockConfigs) {
    super('Block');

    this.game = configs.game;
    this.state = nj.array(this.game.board.points.flat());
    this.observationSpace = new Box(0, 1, [configs.game.config.height, configs.game.config.width, 6], 'float32');
    this.actionSpace = new Discrete(configs.game.config.height * configs.game.config.width);
  }

  invalidActionMask(): NdArray<number> {
    const validActions = this.game.getValidMoves().map((move) => moveToAction(move, this.game.config.width));
    const invalidActionMask = nj.ones(this.actionSpace.n);
    validActions.forEach((action) => {
      invalidActionMask.set(action, 0);
    });
    return invalidActionMask;
  }

  updateState(): void {
    const buffer = tf.buffer([this.game.config.height, this.game.config.width, 6]);

    this.game.board.points.forEach((row, rowi) => {
      row.forEach((value, coli) => {
        buffer.set(1, rowi, coli, value ? value : 0);
      });
    });
    // next shape
    // maintain the same shape as the board
    this.game.board.points.forEach((row, rowi) => {
      row.forEach((_, coli) => {
        const value = this.game.nextShape.points[rowi]?.[coli];
        // this.game.nextShape.points.forEach((row, rowi) => {
        buffer.set(1, rowi, coli, value ? value + 2 : 2);
      });
    });
    // next shape queue
    this.game.board.points.forEach((row, rowi) => {
      row.forEach((_, coli) => {
        const value = this.game.nextShapeQueue.front().points[rowi]?.[coli];
        // this.game.nextShapeQueue.front().points.forEach((row, rowi) => {
        buffer.set(1, rowi, coli, value ? value + 4 : 4);
      });
    });

    this.state = fromTensorSync(buffer.toTensor());
  }

  reset(): State {
    this.game.startNewGame();
    this.updateState();
    this.timestep = 0;
    return this.state;
  }
  step(action: Action) {
    const info: any = {};
    const a = tensorLikeToNdArray(action).get(0);
    const move = actionToMove(a, this.game.config.width);
    // console.log('');
    // console.log(this.game.getTextOutput());
    // console.log(renderShapeAsString(this.game.nextShape));
    // console.log('TCL ~ action:', action);
    // console.log('TCL ~ move:', move);

    if (!this.actionSpace.contains(a)) {
      throw new Error(`${action} is invalid action in Block env`);
    }

    let [valid, scoreDelta, coinDelta] = this.game.getNextStateMutate(move);
    if (valid) {
      const [cvalid, cscore, cchip] = this.game.computerMove();
      scoreDelta += cscore;
    }
    this.timestep += 1;
    this.globalTimestep += 1;

    let done = this.game.checkLose();

    // https://github.com/openai/gym/blob/v0.21.0/gym/wrappers/time_limit.py#L21-L22
    // if (this.timestep >= this.maxEpisodeSteps) {
    //   info['TimeLimit.truncated'] = !done;
    //   info['terminal_observation'] = this.state;
    //   done = true;
    // }

    let reward = valid ? scoreDelta : -1;
    // console.log('TCL ~ reward:', reward);

    this.updateState();
    // console.log(this.game.getTextOutput());
    // console.log(renderShapeAsString(this.game.nextShape));
    // console.log('TCL ~ this.state:', this.state.selection.data);
    // console.log('TCL ~ this.state:', this.state.reshape(this.game.config.height, this.game.config.width, 6).tolist());
    // console.log('TCL ~ reward done:', reward, done);
    return {
      observation: this.state,
      reward,
      done,
      info,
    };
  }

  async render(
    mode: RenderModes,
    configs: { fps: number; episode?: number; rewards?: number } = { fps: 60 }
  ): Promise<void> {
    console.log(this.game.getTextOutput());
  }
}

import { Environment, RenderModes } from '..';
import { NotImplementedError } from '../../Errors';
import { Box2D, Dict, Discrete } from '../../Spaces';

type State = { grid: number[][]; agentPos: Position };
type Action = number;
type Position = { x: number; y: number };
type ActionSpace = Discrete;
type ObservationSpace = Dict<State>;
type Reward = number;
const TERMINAL = 1;
const NON_TERMINAL = 0;

/**
 * Simple GridWorld based on the gridworld presented in Chapters 3-4 in the Intro to RL book by Barto Sutton
 */
export class SimpleGridWorld extends Environment<ActionSpace, ObservationSpace, Action, State, Reward> {
  // TODO: Expand gridworld to have other kinds of rewards and states (e.g. keys, chests, lava etc...)

  /** 
   * {
   *    grid: a 2D array with 0 representing a non-terminal tile/state and 1 representing a terminal tile
   *    agentPos: Position of the agent in the environmnet in the form { x: number, y: number }
   * }
   */
  public observationSpace: ObservationSpace;
  /** 0, 1, 2, 3 represent North, East, South, West directions */
  public actionSpace = new Discrete(4);

  public state: State;

  constructor(
    public width: number,
    public height: number,
    public targetPositions: Position[],
    public startPosition: Position
  ) {
    super();
    this.observationSpace = new Dict({
      grid: new Box2D(0, 1, [width, height]),
      agentPos: new Dict({
        x: new Discrete(4),
        y: new Discrete(4),
      }),
    });
    this.state = this.genState();

    if (this.targetPositions.some((target) => !this.posOnGrid(target))) {
      throw new Error('One of the target positions are off the grid');
    }
    if (!this.posOnGrid(startPosition)) {
      throw new Error(`Start Position ${startPosition} is off the grid`);
    }
  }
  step(action: Action) {
    let reward = -1;
    let newPos = this.translate(this.state.agentPos, action);
    let done = false;
    if (this.posOnGrid(newPos)) {
      this.state.agentPos = newPos;
    }
    if (this.posIsInTargetPositions(this.state.agentPos)) {
      done = true;
    }
    return {
      observation: this.getObs(),
      reward,
      done,
      info: { width: this.width, height: this.height },
    };
  }
  posIsInTargetPositions(pos: Position) {
    return this.targetPositions.some((target) => target.x == pos.x && target.y == pos.y);
  }
  posOnGrid(pos: Position) {
    return !(pos.x < 0 || pos.y < 0 || pos.x >= this.width || pos.y >= this.height);
  }
  private translate(pos: Position, action: Action) {
    switch (action) {
      case 0:
        return { x: pos.x, y: pos.y - 1 };
      case 1:
        return { x: pos.x + 1, y: pos.y };
      case 2:
        return { x: pos.x, y: pos.y + 1 };
      case 3:
        return { x: pos.x - 1, y: pos.y };
      default:
        throw new Error(`Invalid action ${action}`);
    }
  }
  reset(): State {
    this.state = this.genState();
    return this.getObs();
  }
  private getObs(): State {
    return JSON.parse(JSON.stringify(this.state));
  }
  render(mode: RenderModes): void {
    let obs = this.getObs();
    if (mode === 'human') {
      for (let y = 0; y < this.height; y++) {
        console.log(obs.grid[y]);
      }
    } else {
      throw new NotImplementedError("");
    }
  }
  private genState(): State {
    let grid = this.genGrid();
    let agentPos = this.startPosition;
    return { grid, agentPos };
  }
  private genGrid(): number[][] {
    let grid = new Array(this.height);
    for (let y = 0; y < this.height; y++) {
      grid[y] = new Array(this.width);
      for (let x = 0; x < this.width; x++) {
        grid[y][x] = NON_TERMINAL;
      }
    }
    this.targetPositions.forEach((pos) => {
      grid[pos.y][pos.x] = TERMINAL;
    });
    return grid;
  }
  /**
   * Defines the dynamics of SimpleGridWorld
   * 
   * Note that this expects the state is well-formed and not inconsistent
   */
  dynamics(sucessorState: State, reward: Reward, state: State, action: Action) {
    if (reward !== -1) return 0; // reward is always -1;
    
    let {x, y} = this.translate(state.agentPos, action);
    if (!this.posOnGrid({x, y})) {
      if (sucessorState.agentPos.x !== state.agentPos.x || sucessorState.agentPos.y !== state.agentPos.y) {
        return 0;
      }
    };
    if (this.posIsInTargetPositions(state.agentPos)) {
      return 0; // if in target position, episode is over, there can be no more actions
    }
    return 1;
  };
}

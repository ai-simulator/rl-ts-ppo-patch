import { GameHighScore } from '../constants/GameHighScore';
import { HeuristicValues } from '../constants/HeuristicValues';
import { LEVEL_NAME_KEY } from '../constants/Level';
import { Queue } from '../util/queue';
import { GameDisplayState } from './game';
import { GameConfig } from './gameConfig';
import { Move } from './move';
import { Shape } from './shape';

export interface BaseGameInterface {
  config: GameConfig;
  nextShape: Shape;
  nextShapeQueue: Queue<Shape>;
  getNextStateMutate(move: Move): [valid: boolean, scoreDelta: number, coinDelta: number];
  setupTest(number: number): void;
  startNewGame(): void;
  getValidMoves(): Move[];
  checkLose(): boolean;
  computerMove(): [valid: boolean, scoreDelta: number, coinDelta: number];
  getTextOutput(): string;
  clearBoardAnCells(): void;
  resetStates(): void;
  getHeuristicValues(move: Move): HeuristicValues;
}

export interface DisplayGameInterface extends BaseGameInterface {
  score: number;
  lastMoveTime: number;
  message: string;
  startLevel(level: LEVEL_NAME_KEY): void;
  getGameDisplayState(): GameDisplayState;
  getAgentGame(): AgentGameInterface;
  setPlannedMove(move: Move): void;
  removePlannedMove(): void;
  getLastGameHighScore(): GameHighScore;
}

export interface AgentGameInterface extends BaseGameInterface {
  points: number[][];
  clone(): AgentGameInterface;
  savePointsCopy(): void;
  restorePointsCopy(): void;
  saveStateCopy(): void;
  restoreStateCopy(): void;
}

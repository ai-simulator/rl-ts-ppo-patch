import {
  DEFAULT_BOARD_SIZE,
  DEFAULT_COIN_POINT_ADD_MULTIPLIER,
  DEFAULT_COIN_POINT_CLEAR_MULTIPLIER,
  DEFAULT_SCORE_POINT_MULTIPLIER,
} from '../constants/Upgrade';
import { allSet, basicSet, Shape, squareSet, verticalShape } from './shape';

export type BaseGameConfig = {
  height: number;
  width: number;
  scorePointAddedMultiplier: number;
  scorePointClearedMultiplier: number;
  coinPointAddedMultiplier: number;
  coinPointClearedMultiplier: number;
  clearLineVertical: boolean;
  clearLineHorizontal: boolean;
  shapes: Shape[];
  canRotate: boolean;
  scoreThreshold: number | undefined;
  nextShapeQueueCount: number;
};

export type GameConfig = BaseGameConfig;

export const DEFAULT_CLEAR_LINE_GAME_CONFIG: GameConfig = {
  height: DEFAULT_BOARD_SIZE,
  width: DEFAULT_BOARD_SIZE,
  scorePointAddedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  scorePointClearedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  coinPointAddedMultiplier: DEFAULT_COIN_POINT_ADD_MULTIPLIER,
  coinPointClearedMultiplier: DEFAULT_COIN_POINT_CLEAR_MULTIPLIER,
  clearLineHorizontal: true,
  clearLineVertical: true,
  canRotate: false,
  scoreThreshold: undefined,
  nextShapeQueueCount: 3,
  ...basicSet,
};

export const SIMPLE_CONFIG: GameConfig = {
  height: 4,
  width: 4,
  scorePointAddedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  scorePointClearedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  coinPointAddedMultiplier: DEFAULT_COIN_POINT_ADD_MULTIPLIER,
  coinPointClearedMultiplier: DEFAULT_COIN_POINT_CLEAR_MULTIPLIER,
  clearLineHorizontal: true,
  clearLineVertical: true,
  canRotate: false,
  scoreThreshold: undefined,
  nextShapeQueueCount: 3,
  ...basicSet,
};

export const ALL_SHAPES_CLEAR_LINE_GAME_CONFIG: GameConfig = {
  height: DEFAULT_BOARD_SIZE,
  width: DEFAULT_BOARD_SIZE,
  scorePointAddedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  scorePointClearedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  coinPointAddedMultiplier: DEFAULT_COIN_POINT_ADD_MULTIPLIER,
  coinPointClearedMultiplier: DEFAULT_COIN_POINT_CLEAR_MULTIPLIER,
  clearLineHorizontal: true,
  clearLineVertical: true,
  canRotate: false,
  scoreThreshold: undefined,
  nextShapeQueueCount: 3,
  ...allSet,
};

export const ALL_SHAPES_SIZE_7_GAME_CONFIG: GameConfig = {
  height: 7,
  width: 7,
  scorePointAddedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  scorePointClearedMultiplier: DEFAULT_SCORE_POINT_MULTIPLIER,
  coinPointAddedMultiplier: DEFAULT_COIN_POINT_ADD_MULTIPLIER,
  coinPointClearedMultiplier: DEFAULT_COIN_POINT_CLEAR_MULTIPLIER,
  clearLineHorizontal: true,
  clearLineVertical: true,
  canRotate: false,
  scoreThreshold: undefined,
  nextShapeQueueCount: 3,
  ...allSet,
};

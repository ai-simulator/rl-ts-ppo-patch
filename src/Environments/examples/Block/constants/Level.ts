import {
  DEFAULT_CLEAR_LINE_GAME_CONFIG,
  GameConfig,
} from '../model/gameConfig';
import {
  basicPlusSet,
  basicSet,
  challengePlusSet,
  challengeSet,
  expertSet,
  masterSet,
  standardPlusSet,
  standardSet,
  zShapeSet,
} from '../model/shape';
import { LOW_SCORE_THRESHOLD } from './Game';

export const CAMPAIGN_LEVELS = [
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  '10',
] as const;

export const RANKED_LEVEL_NAME = 'Expert';
export const MASTER_LEVEL_NAME = 'Master';
export const ENDLESS_LEVEL_NAME = 'Endless';
export const Z_LEVEL_NAME = 'Zigzag';

export const RANKED_LEVEL_KEY = 'challenge' as const;
export const MASTER_LEVEL_KEY = 'master' as const;
export const ENDLESS_LEVEL_KEY = 'endless' as const;
export const Z_LEVEL_KEY = 'zigzag' as const;

export const NON_RANKED_ALL_LEVELS = [ENDLESS_LEVEL_KEY] as const;

export const RANKED_ALL_LEVELS = [
  RANKED_LEVEL_KEY,
  MASTER_LEVEL_KEY,
  Z_LEVEL_KEY,
] as const;

export const END_GAME_UNLOCK_LEVELS = [
  ...RANKED_ALL_LEVELS,
  ...NON_RANKED_ALL_LEVELS,
] as const;

export const ALL_LEVEL_LIST = [
  ...CAMPAIGN_LEVELS,
  ...NON_RANKED_ALL_LEVELS,
  ...RANKED_ALL_LEVELS,
] as const;

export const DEFAULT_LEVEL = ALL_LEVEL_LIST[0];

export const DEFAULT_LEVEL_UNLOCK_MAP: Record<LEVEL_NAME_KEY, boolean> = {
  ...ALL_LEVEL_LIST.reduce((acc, level) => {
    acc[level] = false;
    return acc;
  }, {} as Record<LEVEL_NAME_KEY, boolean>),
  [DEFAULT_LEVEL]: true,
};

export type LEVEL_NAME_KEY = typeof ALL_LEVEL_LIST[number];

export type LevelInfo = GameConfig & {
  shapeSetName: string;
  name: string;
  endgameLevel: boolean;
  ad?: boolean;
  readonly unlock: LEVEL_NAME_KEY[] | undefined | typeof END_GAME_UNLOCK_LEVELS;
};

const challengeLevel: LevelInfo = {
  ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
  name: RANKED_LEVEL_NAME,
  endgameLevel: true,
  scoreThreshold: undefined,
  width: 9,
  height: 9,
  canRotate: false,
  unlock: undefined,
  ...expertSet,
};

const masterLevel: LevelInfo = {
  ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
  name: MASTER_LEVEL_NAME,
  endgameLevel: true,
  scoreThreshold: undefined,
  width: 9,
  height: 9,
  canRotate: false,
  unlock: undefined,
  ...masterSet,
};

const zLevel: LevelInfo = {
  ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
  name: Z_LEVEL_NAME,
  endgameLevel: true,
  scoreThreshold: undefined,
  width: 9,
  height: 9,
  canRotate: false,
  unlock: undefined,
  ...zShapeSet,
};

const endlessLevel: LevelInfo = {
  ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
  name: ENDLESS_LEVEL_NAME,
  endgameLevel: false,
  scoreThreshold: undefined,
  width: 10,
  height: 10,
  canRotate: false,
  unlock: undefined,
  ad: true,
  ...standardSet,
};

export const LEVEL_MAP: Record<LEVEL_NAME_KEY, LevelInfo> = {
  1: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 1',
    endgameLevel: false,
    scoreThreshold: 50,
    width: 5,
    height: 5,
    canRotate: false,
    unlock: ['2'],
    ...basicSet,
  },
  2: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 2',
    endgameLevel: false,
    scoreThreshold: 100,
    width: 6,
    height: 6,
    canRotate: false,
    unlock: ['3'],
    ...basicPlusSet,
  },
  3: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 3',
    endgameLevel: false,
    scoreThreshold: 150,
    width: 7,
    height: 7,
    canRotate: false,
    unlock: ['4'],
    ...basicPlusSet,
  },
  4: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 4',
    endgameLevel: false,
    scoreThreshold: LOW_SCORE_THRESHOLD ? 204 : 250,
    width: 8,
    height: 7,
    canRotate: false,
    unlock: ['5'],
    ...standardSet,
  },
  5: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 5',
    endgameLevel: false,
    scoreThreshold: LOW_SCORE_THRESHOLD ? 205 : 300,
    width: 8,
    height: 8,
    canRotate: false,
    unlock: ['6'],
    ...standardSet,
  },
  6: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 6',
    endgameLevel: false,
    scoreThreshold: LOW_SCORE_THRESHOLD ? 206 : 400,
    width: 9,
    height: 8,
    canRotate: false,
    unlock: ['7'],
    ...standardSet,
  },
  7: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 7',
    endgameLevel: false,
    scoreThreshold: LOW_SCORE_THRESHOLD ? 207 : 500,
    width: 9,
    height: 9,
    canRotate: false,
    unlock: ['8'],
    ...standardSet,
  },
  8: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 8',
    endgameLevel: false,
    scoreThreshold: LOW_SCORE_THRESHOLD ? 208 : 600,
    width: 9,
    height: 9,
    canRotate: false,
    unlock: ['9'],
    ...standardPlusSet,
  },
  9: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 9',
    endgameLevel: false,
    scoreThreshold: LOW_SCORE_THRESHOLD ? 209 : 800,
    width: 9,
    height: 9,
    canRotate: false,
    unlock: ['10'],
    ...challengeSet,
  },
  10: {
    ...DEFAULT_CLEAR_LINE_GAME_CONFIG,
    name: 'Level 10',
    endgameLevel: false,
    scoreThreshold: LOW_SCORE_THRESHOLD ? 210 : 1000,
    width: 9,
    height: 9,
    canRotate: false,
    unlock: END_GAME_UNLOCK_LEVELS,
    ...challengePlusSet,
  },
  endless: endlessLevel,
  challenge: challengeLevel,
  master: masterLevel,
  zigzag: zLevel,
};

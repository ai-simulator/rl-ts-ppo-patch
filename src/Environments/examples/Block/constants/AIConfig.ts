import { OPTION_INFO } from './OptionInfo';
import { formatNumber } from '../util';
import { LEVEL_MAP } from './Level';

export type AI_CONFIG_KEY = keyof AIConfigType;

export const PARAMS_DISPLAY_NAME = 'Tuning';

export enum SELECTOR_TYPE {
  SLIDER = 'SLIDER',
  SELECT = 'SELECT',
  PRIORITY = 'PRIORITY',
}

export const POSITIVE_FACTORS = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10] as const;
export type POSITIVE_FACTOR_VALUE = typeof POSITIVE_FACTORS[number];

export const NEGATIVE_FACTORS = [-10, -5, -2, -1, -0.5, -0.2, -0.1, 0] as const;
export type NEGATIVE_FACTOR_VALUE = typeof NEGATIVE_FACTORS[number];

export interface AI_CONFIG_INFO extends OPTION_INFO {
  symbol: string;
  options: number[] | typeof POSITIVE_FACTORS | typeof NEGATIVE_FACTORS;
  selectorType: SELECTOR_TYPE;
  isFactor?: boolean;
  optionsMappingFunc?: (option: number) => string;
  valueMappingFunc?: (value: number) => string;
  isRewardFactor?: boolean;
  sliderStep?: number;
  decimals?: number;
  webLinkName?: string;
}

export type AIModelTypeName = 'dqn' | 'ppo';

export const DQNConfigKeys = [
  'dqnWidth',
  'dqnHeight',
  'dqnFactorMovePenalty',
  'dqnFactorScore',
  'dqnAlpha',
  'dqnGamma',
  'dqnEpsilonInit',
  'dqnEpsilonFinal',
  'dqnEpsilonDecayFrames',
] as const;

export type DQNConfigKey = typeof DQNConfigKeys[number];

export type DQNConfigType = {
  [K in DQNConfigKey]: number;
};

export type PPOConfigType = {};

export type AIModelConfigType = DQNConfigType | PPOConfigType;

export const DEFAULT_DQN_CONFIG: DQNConfigType = {
  dqnFactorMovePenalty: -1,
  dqnAlpha: 0.00005,
  dqnGamma: 0.9,
  dqnEpsilonInit: 0.4,
  dqnEpsilonFinal: 0.05,
  dqnEpsilonDecayFrames: 500000,
  dqnWidth: LEVEL_MAP['challenge'].width,
  dqnHeight: LEVEL_MAP['challenge'].height,
  dqnFactorScore: 0.2,
};

export type BaseAIModelType = {
  type: AIModelTypeName;
  config: AIModelConfigType;
  frames: number;
  lastSaveTime: number;
  createTime: number;
};

export type DQNModelType = BaseAIModelType & {
  type: 'dqn';
  config: DQNConfigType;
};

export type PPOModelType = BaseAIModelType & {
  type: 'ppo';
  config: PPOConfigType;
};

export type AIModelType = DQNModelType | PPOModelType;

export function getModelSummary(model: AIModelType) {
  if (model.type === 'dqn') {
    return `Size: ${model.config.dqnHeight}*${
      model.config.dqnWidth
    } | Frames: ${formatNumber(model.frames)}`;
  } else if (model.type === 'ppo') {
    return '-';
  }
  return '-';
}

export type AIConfigType = {
  // heuristics
  heurClearLineFactor: POSITIVE_FACTOR_VALUE;
  heurEdgeFactor: NEGATIVE_FACTOR_VALUE;
  heurEmptySpaceFactor: POSITIVE_FACTOR_VALUE;
  // MOCA
  mcNumMovesAhead: number;
  mcNumSimulations: number;
  mcScoreFactor: POSITIVE_FACTOR_VALUE;
  mcEdgesFactor: NEGATIVE_FACTOR_VALUE;
  mcEmptyCellsFactor: POSITIVE_FACTOR_VALUE;
  // mcLinesClearedFactor: POSITIVE_FACTOR_VALUE;
  // mcEmptySpaceFactor: POSITIVE_FACTOR_VALUE;
  // MOCA II
  mc2NumMovesAhead: number;
  mc2NumSimulations: number;
  // EXPM
  expmDepth: number;
  expmMoveThreshold: number;
  // expmScoreFactor: POSITIVE_FACTOR_VALUE;
  // expmLinesClearedFactor: POSITIVE_FACTOR_VALUE;
  expmEdgesFactor: NEGATIVE_FACTOR_VALUE;
  // expmEmptySpaceFactor: POSITIVE_FACTOR_VALUE;
  expmEmptyCellsFactor: POSITIVE_FACTOR_VALUE;
  // expmHolesFactor: NEGATIVE_FACTOR_VALUE;
  // EXPM II
  expm2Depth: number;
  expm2MoveThreshold: number;
  // PMGS
  pmgsGames: number;
} & DQNConfigType;

export const DEFAULT_AI_CONFIG: AIConfigType = {
  // heuristics
  heurClearLineFactor: 1,
  heurEdgeFactor: -0.1,
  heurEmptySpaceFactor: 5, // TODO: Get a good factor
  // MOCA
  mcNumMovesAhead: 2,
  mcNumSimulations: 50,
  // TODO: Find good defaults
  mcScoreFactor: 1,
  mcEdgesFactor: -0.1,
  mcEmptyCellsFactor: 1,
  // mcLinesClearedFactor: 1,
  // mcEmptySpaceFactor: 5,
  // MOCA II
  mc2NumMovesAhead: 3,
  mc2NumSimulations: 50,
  // EXPM
  expmDepth: 2,
  expmEdgesFactor: -0.2,
  expmMoveThreshold: 25,
  expmEmptyCellsFactor: 1,
  // expmScoreFactor: 0.1,
  // expmLinesClearedFactor: 1,
  // expmEmptySpaceFactor: 5,
  // expmHolesFactor: 0,
  pmgsGames: 50,
  // EXPM II
  expm2Depth: 3,
  expm2MoveThreshold: 20,
  // DQN
  ...DEFAULT_DQN_CONFIG,
};

export const HEURISTIC_BASE_SCORE = 100;

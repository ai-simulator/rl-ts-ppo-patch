import { AI_LIST } from './AI';

import {
  COST_TIER_1,
  COST_TIER_2_5,
  COST_TIER_3,
  COST_TIER_3_2,
  COST_TIER_3_5,
  COST_TIER_4_2,
  COST_TIER_4_5,
  COST_TIER_5,
  COST_TIER_5_2,
} from './Chip';
import { COLOR_SCHEME, COLOR_SCHEMES } from './Colors';
import { OPTION_INFO } from './OptionInfo';

export const DEFAULT_BOARD_SIZE = 9;

export const DEFAULT_SCORE_POINT_MULTIPLIER = 2;
export const DEFAULT_COIN_POINT_ADD_MULTIPLIER = 0;
export const DEFAULT_COIN_POINT_CLEAR_MULTIPLIER = 1;

export enum OptionState {
  LOCKED = 'locked',
  DISABLED = 'disable',
  SELECTED = 'select',
  ENABLED = 'enable',
}

export const SPEED_UPGRADE_KEY_LIST = [
  'solveSpeed_tier_1',
  'solveSpeed_tier_2',
  'solveSpeed_tier_3',
  'solveSpeed_tier_4',
  'solveSpeed_tier_5',
] as const;
export type SPEED_UPGRADE_KEY = typeof SPEED_UPGRADE_KEY_LIST[number];

export const RENDER_MODE_KEY_LIST = [
  'graphics_mode',
  'text_mode',
  'no_ui_mode',
] as const;

export type RENDER_MODE_KEY = typeof RENDER_MODE_KEY_LIST[number];

export const UPGRADE_KEY_LIST = [
  ...SPEED_UPGRADE_KEY_LIST,
  ...COLOR_SCHEMES,
  'aiLog',
  'nextShapeQueue',
  'fps',
  ...RENDER_MODE_KEY_LIST,
  'batch_solver',
] as const;

export type UPGRADE_KEY = typeof UPGRADE_KEY_LIST[number];

export const OPTION_KEY_LIST = [...AI_LIST, ...UPGRADE_KEY_LIST];

export type OPTION_KEY = typeof OPTION_KEY_LIST[number];

export interface UPGRADE_INFO extends OPTION_INFO {
  prerequisites?: UPGRADE_KEY[];
  mutuallyExclusive?: UPGRADE_KEY[];
  value: number;
  isColorScheme?: boolean;
  groupCount?: number;
  scoreRequirement?: number;
  isMapLayout?: boolean;
  defaultState: OptionState;
}

export type UpgradeConfigType = Record<UPGRADE_KEY, OptionState>;
export type ColorConfigType = Record<COLOR_SCHEME, OptionState>;
export type SpeedConfigType = Record<SPEED_UPGRADE_KEY, OptionState>;

export const UPGRADE_MAP: Record<UPGRADE_KEY, UPGRADE_INFO> = {
  solveSpeed_tier_1: {
    name: 'Z1',
    value: 2,
    cost: 0,
    next: 'solveSpeed_tier_2',
    mutuallyExclusive: [...SPEED_UPGRADE_KEY_LIST],
    groupCount: SPEED_UPGRADE_KEY_LIST.length,
    defaultState: OptionState.ENABLED,
  },
  solveSpeed_tier_2: {
    name: 'Z3',
    value: 3,
    cost: COST_TIER_1,
    prerequisites: ['solveSpeed_tier_1'],
    next: 'solveSpeed_tier_3',
    mutuallyExclusive: [...SPEED_UPGRADE_KEY_LIST],
    groupCount: SPEED_UPGRADE_KEY_LIST.length,
    defaultState: OptionState.LOCKED,
  },
  solveSpeed_tier_3: {
    name: 'Z5',
    value: 4,
    cost: COST_TIER_2_5,
    prerequisites: ['solveSpeed_tier_2'],
    next: 'solveSpeed_tier_4',
    mutuallyExclusive: [...SPEED_UPGRADE_KEY_LIST],
    groupCount: SPEED_UPGRADE_KEY_LIST.length,
    defaultState: OptionState.LOCKED,
  },
  solveSpeed_tier_4: {
    name: 'Z7',
    value: 10,
    cost: COST_TIER_3_2,
    prerequisites: ['solveSpeed_tier_3'],
    next: 'solveSpeed_tier_5',
    mutuallyExclusive: [...SPEED_UPGRADE_KEY_LIST],
    groupCount: SPEED_UPGRADE_KEY_LIST.length,
    defaultState: OptionState.LOCKED,
  },
  solveSpeed_tier_5: {
    name: 'Z9',
    value: 20,
    cost: COST_TIER_4_5,
    prerequisites: ['solveSpeed_tier_4'],
    mutuallyExclusive: [...SPEED_UPGRADE_KEY_LIST],
    groupCount: SPEED_UPGRADE_KEY_LIST.length,
    defaultState: OptionState.LOCKED,
  },
  aiLog: {
    name: 'AI Log',
    value: 0,
    cost: COST_TIER_2_5,
    defaultState: OptionState.LOCKED,
    groupCount: 2,
  },
  nextShapeQueue: {
    name: 'Next Shape Queue',
    value: 0,
    cost: COST_TIER_3,
    defaultState: OptionState.LOCKED,
    groupCount: 2,
  },
  fps: {
    name: 'FPS Counter',
    value: 0,
    cost: COST_TIER_3_5,
    defaultState: OptionState.LOCKED,
  },
  graphics_mode: {
    name: 'Graphics',
    value: 0,
    cost: 0,
    defaultState: OptionState.ENABLED,
    mutuallyExclusive: [...RENDER_MODE_KEY_LIST],
    groupCount: RENDER_MODE_KEY_LIST.length,
  },
  text_mode: {
    name: 'Text',
    value: 0,
    cost: COST_TIER_4_2,
    defaultState: OptionState.LOCKED,
    mutuallyExclusive: [...RENDER_MODE_KEY_LIST],
    groupCount: RENDER_MODE_KEY_LIST.length,
  },
  no_ui_mode: {
    name: 'No UI',
    value: 0,
    cost: COST_TIER_5,
    defaultState: OptionState.LOCKED,
    mutuallyExclusive: [...RENDER_MODE_KEY_LIST],
    groupCount: RENDER_MODE_KEY_LIST.length,
  },
  batch_solver: {
    name: 'Batch Solver',
    value: 0,
    cost: COST_TIER_5_2,
    defaultState: OptionState.LOCKED,
  },
  ...COLOR_SCHEMES.reduce((acc, key) => {
    acc[key] = {
      name: key,
      value: 0,
      cost: 0,
      mutuallyExclusive: [...COLOR_SCHEMES],
      isColorScheme: true,
      defaultState: OptionState.LOCKED,
    };
    return acc;
  }, {} as Record<COLOR_SCHEME, UPGRADE_INFO>),
};

export const DEFAULT_UPGRADE_CONFIG: UpgradeConfigType = {
  ...UPGRADE_KEY_LIST.reduce((acc, key) => {
    acc[key] = UPGRADE_MAP[key].defaultState;
    return acc;
  }, {} as Record<UPGRADE_KEY, OptionState>),
} as const;

export const DEFAULT_SPEED = UPGRADE_MAP['solveSpeed_tier_1'].value;

export const getNewUpgradeConfig = (
  prevUpgradeConfig: UpgradeConfigType,
  upgrade: UPGRADE_KEY,
  value: OptionState
) => {
  const newUpgradeConfig = {
    ...prevUpgradeConfig,
    [upgrade]: value,
  };
  const mutuallyExclusive = UPGRADE_MAP[upgrade].mutuallyExclusive;
  if (mutuallyExclusive) {
    for (let i = 0; i < mutuallyExclusive.length; i++) {
      const u = mutuallyExclusive[i];
      if (u === upgrade) continue;
      if (prevUpgradeConfig[u] === OptionState.LOCKED) continue;
      newUpgradeConfig[u] = OptionState.DISABLED;
    }
  }
  return newUpgradeConfig;
};

export const getSolveSpeed = (upgradeConfig: UpgradeConfigType) => {
  let speed: number = DEFAULT_SPEED;
  const speedUpgrades = SPEED_UPGRADE_KEY_LIST;
  for (let index = 0; index < speedUpgrades.length; index++) {
    const upgrade = speedUpgrades[index];
    if (
      upgradeConfig[upgrade] === OptionState.ENABLED &&
      UPGRADE_MAP[upgrade].value > speed
    ) {
      speed = UPGRADE_MAP[upgrade].value;
    }
  }
  return speed;
};

export const getCPUEnabled = (upgradeConfig: UpgradeConfigType) => {
  let speed: number = DEFAULT_SPEED;
  let cpuName: string = UPGRADE_MAP[SPEED_UPGRADE_KEY_LIST[0]].name;
  const speedUpgrades = SPEED_UPGRADE_KEY_LIST;
  for (let index = 0; index < speedUpgrades.length; index++) {
    const upgrade = speedUpgrades[index];
    if (
      upgradeConfig[upgrade] === OptionState.ENABLED &&
      UPGRADE_MAP[upgrade].value > speed
    ) {
      speed = UPGRADE_MAP[upgrade].value;
      cpuName = UPGRADE_MAP[upgrade].name;
    }
  }
  return cpuName;
};

export const requirementTitle = 'Unmet Requirements';

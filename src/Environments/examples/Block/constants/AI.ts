import { AgentGameInterface } from '../model/gameInterface';
import { Move } from '../model/move';
import { AIConfigType } from './AIConfig';

export const AI_LIST = [
  'RAND',
  'FILO',
  'CLLI',
  'EDGE',
  'EES',
  'FILO',
  'HEUR',
  'HEUR2',
  'MOCA',
  'MOCA2',
  'EXPM',
  'EXPM2',
  'PMGS',
  'DQN',
] as const;

export type AI_NAME_KEY = typeof AI_LIST[number];
export const DEFAULT_AI = AI_LIST[0];

const AI_GROUP_BASELINE = 'Baseline' as const;
const AI_GROUP_SIMPLE_HEURISTIC = 'Basic Heuristics' as const;
const AI_GROUP_CLASSIC_HEURISTIC = 'Classical Heuristics' as const;
const AI_GROUP_ADVANCED_HEURISTIC = 'Advanced Heuristics' as const;
const AI_GROUP_ADVANCED_SEARCH = 'Advanced Search' as const;
const AI_GROUP_COMBINATIONS = 'Combinations' as const;
const AI_GROUP_SIMULATIONS = 'Monte Carlo Methods' as const;
const AI_GROUP_ML = 'Machine Learning' as const;
const AI_GROUP_HUMAN = '?' as const;

export const AI_GROUP_LIST = [
  AI_GROUP_BASELINE,
  AI_GROUP_SIMPLE_HEURISTIC,
  AI_GROUP_ADVANCED_HEURISTIC,
  // AI_GROUP_CLASSIC_HEURISTIC,
  AI_GROUP_SIMULATIONS,
  AI_GROUP_ADVANCED_SEARCH,
  AI_GROUP_ML,
  // AI_GROUP_COMBINATIONS,
  // AI_GROUP_HUMAN,
] as const;

export type AI_GROUP_NAME = typeof AI_GROUP_LIST[number];

export const AI_GROUP_MAPPING: Record<AI_GROUP_NAME, AI_NAME_KEY[]> = {
  [AI_GROUP_BASELINE]: ['RAND'],
  [AI_GROUP_SIMPLE_HEURISTIC]: ['EES', 'CLLI', 'FILO', 'EDGE'],
  [AI_GROUP_ADVANCED_HEURISTIC]: ['HEUR', 'HEUR2'],
  // [AI_GROUP_CLASSIC_HEURISTIC]: [],
  // [AI_GROUP_COMBINATIONS]: ['DFSD', 'BFSD'],
  [AI_GROUP_ML]: ['DQN'],
  [AI_GROUP_SIMULATIONS]: ['MOCA', 'MOCA2', 'PMGS'],
  [AI_GROUP_ADVANCED_SEARCH]: ['EXPM', 'EXPM2'],
  // [AI_GROUP_HUMAN]: ['HOMO'],
};

export type AI_MOVE_RESULT = [move: Move | undefined, message: string];

export type AI_FUNC = (
  state: AgentGameInterface,
  config: AIConfigType
) => AI_MOVE_RESULT;

// export type AI_FUNC_MULTI_MOVES = (
//   state: AgentGameInterface,
//   config: AIConfigType
// ) => Move[] | undefined;

export function isTrainableDQN(AI: AI_NAME_KEY): boolean {
  return AI === 'DQN';
}

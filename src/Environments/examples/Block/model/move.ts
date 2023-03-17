import { formatNumberDecimalIfNecessary } from '../util';

export type Move = [x: number, y: number];

export function getRandomMoveMsg(move: Move) {
  if (!move) {
    return `Random move not found`;
  }
  return `Found random move: (${move[0]},${move[1]})`;
}

export function getFallbackRandomMoveMsg(move: Move) {
  if (!move) {
    return `Fallback random move not found`;
  }
  return `Fallback to random move: (${move[0]},${move[1]})`;
}

export function getBestMoveMsg(move: Move) {
  if (!move) {
    return `Best move not found`;
  }
  return `Found best move: (${move[0]},${move[1]})`;
}

export function getBestMoveScoreMsg(move: Move, score: number) {
  return `Found best move with score ${formatNumberDecimalIfNecessary(
    score,
    2
  )}: (${move[0]},${move[1]})`;
}

export function getBestMoveRewardMsg(move: Move, reward: number) {
  return `Found best move with reward ${formatNumberDecimalIfNecessary(
    reward,
    2
  )}: (${move[0]},${move[1]})`;
}

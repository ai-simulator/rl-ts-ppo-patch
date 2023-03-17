// use short keys for storage and easier migration
export type GameHighScore = {
  score: number;
  moveCount: number;
  time: number;
  h: string;
  w: string;
  durSec: string;
  cellsCleared: number;
};

export const REVIEW_SCORE = 1200;

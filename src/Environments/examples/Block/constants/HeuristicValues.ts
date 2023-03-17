export class HeuristicValues {
  linesCleared!: number;
  edges!: number;
  essRatio!: number;
  success!: boolean;
  scoreDelta!: number;
}

export class PointsHeuristicValues {
  emptyCells!: number;
  edges!: number;
  essRatio!: number;
  // holes: number;
}

export class FastPointsHeuristicValues {
  emptyCells!: number;
  edges!: number;
}

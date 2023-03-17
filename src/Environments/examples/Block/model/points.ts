import {
  FastPointsHeuristicValues,
  PointsHeuristicValues,
} from '../constants/HeuristicValues';
import { Shape } from './shape';
import { Board } from './board';

export function getPointsCopy(points: number[][]): number[][] {
  const pointsCopy = [];
  for (let i = 0; i < points.length; i++) {
    pointsCopy.push([]);
    for (let j = 0; j < points[i].length; j++) {
      pointsCopy[i][j] = points[i][j];
    }
  }
  return pointsCopy;
}

export function getPointsHeuristicValues(
  points: number[][],
  shapes: Shape[]
): PointsHeuristicValues {
  // avoid unnecessary varible assignments to improve performance
  // get all heuristics all at once to avoid looping through the same points multiple times
  let emptyCells = 0;
  let edges = 0;
  let numPossibleShapes = 0;
  let numTotalShapes = shapes.length;
  const xMax = points[0].length;
  const yMax = points.length;
  for (let k = 0; k < numTotalShapes; k++) {
    const shape = shapes[k];
    let valid = false;
    for (let i = 0; i < yMax; i++) {
      for (let j = 0; j < xMax; j++) {
        if (k === 0) {
          // count edge and empty cells
          const element = points[i][j];
          const above = i > 0 ? points[i - 1][j] : 0;
          const below = i < yMax - 1 ? points[i + 1][j] : 0;
          if (element === 1) {
            // block cell different from 2 adjacents
            if (above === 0) {
              if (!points[i][j - 1]) {
                edges++;
              }
              if (!points[i][j + 1]) {
                edges++;
              }
            }
            if (below === 0) {
              if (!points[i][j + 1]) {
                edges++;
              }
              if (!points[i][j - 1]) {
                edges++;
              }
            }
          } else if (element === 0) {
            // empty cell
            emptyCells++;
            // empty cell different from 3 adjacents
            if (above === 1) {
              if (points[i][j - 1] === 1 && points[i - 1][j - 1] === 1) {
                edges++;
              }
              if (points[i][j + 1] === 1 && points[i - 1][j + 1] === 1) {
                edges++;
              }
            }
            if (below === 1) {
              if (points[i][j - 1] === 1 && points[i + 1][j - 1] === 1) {
                edges++;
              }
              if (points[i][j + 1] === 1 && points[i + 1][j + 1] === 1) {
                edges++;
              }
            }
          }
        }
        if (!valid && checkAddShapeValid(j, i, shape, xMax, yMax, points)) {
          valid = true;
        }
      }
    }
    if (valid) {
      numPossibleShapes++;
    }
  }

  const essRatio = numPossibleShapes / numTotalShapes;

  return {
    emptyCells,
    edges,
    essRatio,
    // holes: getNumberOfHoles(pointsCopy),
  };
}

export function getFastPointsHeuristicValues(
  points: number[][]
): FastPointsHeuristicValues {
  // avoid unnecessary varible assignments to improve performance
  // get all heuristics all at once to avoid looping through the same points multiple times
  let emptyCells = 0;
  let edges = 0;
  const xMax = points[0].length;
  const yMax = points.length;
  for (let i = 0; i < yMax; i++) {
    for (let j = 0; j < xMax; j++) {
      // count edge and empty cells
      const element = points[i][j];
      const above = i > 0 ? points[i - 1][j] : 0;
      const below = i < yMax - 1 ? points[i + 1][j] : 0;
      if (element === 1) {
        // block cell different from 2 adjacents
        if (above === 0) {
          if (!points[i][j - 1]) {
            edges++;
          }
          if (!points[i][j + 1]) {
            edges++;
          }
        }
        if (below === 0) {
          if (!points[i][j + 1]) {
            edges++;
          }
          if (!points[i][j - 1]) {
            edges++;
          }
        }
      } else if (element === 0) {
        // empty cell
        emptyCells++;
        // empty cell different from 3 adjacents
        if (above === 1) {
          if (points[i][j - 1] === 1 && points[i - 1][j - 1] === 1) {
            edges++;
          }
          if (points[i][j + 1] === 1 && points[i - 1][j + 1] === 1) {
            edges++;
          }
        }
        if (below === 1) {
          if (points[i][j - 1] === 1 && points[i + 1][j - 1] === 1) {
            edges++;
          }
          if (points[i][j + 1] === 1 && points[i + 1][j + 1] === 1) {
            edges++;
          }
        }
      }
    }
  }

  return {
    emptyCells,
    edges,
  };
}

export function checkAddShapeValid(
  x: number,
  y: number,
  shape: Shape,
  boardWidth: number,
  boardHeight: number,
  boardPoints: number[][]
) {
  // validity checks
  const { height, width, points } = shape;
  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      const point = points[i][j];
      if (point > 0) {
        const xTranslated = x + j;
        const yTranslated = y + i;
        if (xTranslated >= boardWidth || yTranslated >= boardHeight) {
          return false;
        }
        if (boardPoints[yTranslated][xTranslated] > 0) {
          return false;
        }
      }
    }
  }
  return true;
}

// find disconnected components of 0s
export function getNumberOfHoles(points: number[][]) {
  const visited = {};
  function dfs(x: number, y: number) {
    if (x < 0 || x >= points[0].length || y < 0 || y >= points.length) {
      return;
    }
    if (points[y][x] !== 0 || visited[`${x}-${y}`]) {
      return;
    }
    visited[`${x}-${y}`] = true;
    dfs(x - 1, y);
    dfs(x + 1, y);
    dfs(x, y - 1);
    dfs(x, y + 1);
  }

  let count = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = 0; j < points[i].length; j++) {
      if (points[i][j] === 0 && !visited[`${j}-${i}`]) {
        count++;
        dfs(j, i);
      }
    }
  }

  return count;
}

export function getEdgeCount(points: number[][]) {
  // this function is called many times
  // avoid variable assignments and object destructuring
  let edgeCount = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = 0; j < points[i].length; j++) {
      const element = points[i][j];
      if (element === 1) {
        // block cell different from 2 adjacents
        if (element !== (points[i - 1] ? points[i - 1][j] : 0)) {
          if (element !== points[i][j - 1] || 0) {
            edgeCount++;
          }
          if (element !== points[i][j + 1] || 0) {
            edgeCount++;
          }
        }
        if (element !== (points[i + 1] ? points[i + 1][j] : 0)) {
          if (element !== points[i][j + 1] || 0) {
            edgeCount++;
          }
          if (element !== points[i][j - 1] || 0) {
            edgeCount++;
          }
        }
      } else if (element === 0) {
        // empty cell different from 3 adjacents
        if (element !== (points[i - 1] ? points[i - 1][j] : 0)) {
          if (
            element !== (points[i][j - 1] || 0) &&
            element !== (points[i - 1] ? points[i - 1][j - 1] : 0)
          ) {
            edgeCount++;
          }
          if (
            element !== (points[i][j + 1] || 0) &&
            element !== (points[i - 1] ? points[i - 1][j + 1] : 0)
          ) {
            edgeCount++;
          }
        }
        if (element !== (points[i + 1] ? points[i + 1][j] : 0)) {
          if (
            element !== (points[i][j - 1] || 0) &&
            element !== (points[i + 1] ? points[i + 1][j - 1] : 0)
          ) {
            edgeCount++;
          }
          if (
            element !== (points[i][j + 1] || 0) &&
            element !== (points[i + 1] ? points[i + 1][j + 1] : 0)
          ) {
            edgeCount++;
          }
        }
      }
    }
  }

  return edgeCount;
}

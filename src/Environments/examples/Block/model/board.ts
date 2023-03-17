import { checkAddShapeValid, getEdgeCount } from './points';
import { renderShapeAsString, Shape } from './shape';

export class Board extends Shape {
  constructor(points: number[][]) {
    super(points);
  }

  clone() {
    const pointsCopy = [];
    for (let i = 0; i < this.points.length; i++) {
      pointsCopy.push([]);
      for (let j = 0; j < this.points[i].length; j++) {
        pointsCopy[i][j] = this.points[i][j];
      }
    }
    return new Board(pointsCopy);
  }

  static _getClearableColumns(width, points) {
    const clearableColumns: number[] = [];
    const maxColumns = width;
    for (let i = 0; i < maxColumns; i++) {
      let clearable = true;
      for (let j = 0; j < points.length; j++) {
        const cell = points[j][i];
        if (cell !== 1) {
          clearable = false;
          break;
        }
      }
      if (clearable) {
        clearableColumns.push(i);
      }
    }
    return clearableColumns;
  }

  getClearableColumns(): number[] {
    return Board._getClearableColumns(this.width, this.points);
  }

  static _getClearableRows(height, points) {
    const clearableRows: number[] = [];
    const maxRows = height;
    for (let i = 0; i < maxRows; i++) {
      let clearable = true;
      for (let j = 0; j < points[i].length; j++) {
        if (points[i][j] !== 1) {
          clearable = false;
          break;
        }
      }
      if (clearable) {
        clearableRows.push(i);
      }
    }
    return clearableRows;
  }

  getClearableRows(): number[] {
    return Board._getClearableRows(this.height, this.points);
  }

  static _clearRow(row: number, width, points) {
    let count = 0;
    for (let i = 0; i < width; i++) {
      count += points[row][i] === 1 ? 1 : 0;
      points[row][i] = 0;
    }
    return count;
  }

  clearRow(row: number) {
    return Board._clearRow(row, this.width, this.points);
  }

  static _clearColumn(column, height, points) {
    let count = 0;
    for (let i = 0; i < height; i++) {
      count += points[i][column] === 1 ? 1 : 0;
      points[i][column] = 0;
    }
    return count;
  }

  clearColumn(column: number) {
    return Board._clearColumn(column, this.height, this.points);
  }

  static _getPointsCovered(width, height, points) {
    let count = 0;
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        if (points[i][j] > 0) {
          count++;
        }
      }
    }
    return count;
  }

  getPointsCovered() {
    return Board._getPointsCovered(this.width, this.height, this.points);
  }

  static _;

  checkAddShapeValid(x: number, y: number, shape: Shape): boolean {
    return checkAddShapeValid(x, y, shape, this.width, this.height, this.points);
  }

  static _tryAddShape(
    x,
    y,
    shape,
    boardWidth,
    boardHeight,
    boardPoints
  ): [success: boolean, pointsAdded: [x: number, y: number][]] {
    let pointsAdded: [x: number, y: number][] = [];
    const { height, width, points } = shape;

    const valid = checkAddShapeValid(x, y, shape, boardWidth, boardHeight, boardPoints);
    if (!valid) {
      return [false, pointsAdded];
    }

    // perform adding
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const point = points[i][j];
        if (point > 0) {
          const xTranslated = x + j;
          const yTranslated = y + i;
          boardPoints[yTranslated][xTranslated] = point;
          pointsAdded.push([xTranslated, yTranslated]);
        }
      }
    }
    return [true, pointsAdded];
  }

  tryAddShape(x: number, y: number, shape: Shape): [success: boolean, pointsAdded: [x: number, y: number][]] {
    return Board._tryAddShape(x, y, shape, this.width, this.height, this.points);
  }

  getEdgeCount() {
    return getEdgeCount(this.points);
  }

  print() {
    console.log(renderShapeAsString(this));
  }
}

export function getEmptyBoard(width: number, height: number) {
  const points = [];
  for (let i = 0; i < height; i++) {
    points[i] = [];
    for (let j = 0; j < width; j++) {
      points[i][j] = 0;
    }
  }
  return new Board(points);
}

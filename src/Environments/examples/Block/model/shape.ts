import { getRandomElement } from '../util';
import { Cell } from './cell';

// 0 -> empty
// 1 -> block

const FILLED_SHAPE = 'X';
const EMPTY_SHAPE = '.';

export class Shape {
  height: number;
  width: number;
  points: number[][]; // x,y
  blockCount: number;

  constructor(points: number[][]) {
    this.height = points.length; // x
    this.width = points[0].length; // y
    this.points = points;
    this.blockCount = 0;
    for (let i = 0; i < this.points.length; i++) {
      const row = this.points[i];
      for (let j = 0; j < row.length; j++) {
        const cell = row[j];
        if (cell === 1) {
          this.blockCount++;
        }
      }
    }
  }

  getCells(id: number): Cell[][] {
    const cells: Cell[][] = [];
    const xMax = this.width;
    const yMax = this.height;
    for (let y = 0; y < yMax; y++) {
      cells[y] = [];
      for (let x = 0; x < xMax; x++) {
        cells[y][x] = new Cell(x, y, this.points[y][x], id);
      }
    }
    return cells;
  }
}

export function renderShapeAsString(shape: Shape) {
  const { height, width, points } = shape;
  return renderPoints(points, width, height);
}

export function renderPoints(points: number[][], width: number, height: number) {
  let str = '\n';
  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      str += points[i][j] ? FILLED_SHAPE : EMPTY_SHAPE;
    }
    str += '\n';
  }
  return str;
}

export const squareShape = new Shape([
  [1, 1],
  [1, 1],
]);

// prettier-ignore
export const verticalShape = new Shape([
  [1],
  [1],
  [1],
  [1],
]);

export const horizontalShape = new Shape([[1, 1, 1, 1]]);

// prettier-ignore
export const tUpShape = new Shape([
  [0, 1, 0],
  [1, 1, 1],
]);

// prettier-ignore
export const tdownShape = new Shape([
  [1, 1, 1],
  [0, 1, 0],
]);

// prettier-ignore
export const lShape = new Shape([
  [1, 0],
  [1, 0],
  [1, 1],
]);

// prettier-ignore
export const jShape = new Shape([
  [0, 1],
  [0, 1],
  [1, 1],
]);

// prettier-ignore
export const zShape = new Shape([
  [1, 1, 0],
  [0, 1, 1],
]);

// prettier-ignore
export const zVerticalShape = new Shape([
  [0, 1],
  [1, 1],
  [1, 0],
]);

// prettier-ignore
export const sShape = new Shape([
  [0, 1, 1],
  [1, 1, 0],
]);

// prettier-ignore
export const sVerticalShape = new Shape([
  [1, 0],
  [1, 1],
  [0, 1],
]);

// prettier-ignore
export const lShapeHorizontal = new Shape([
  [0, 0, 1],
  [1, 1, 1],
]);

// prettier-ignore
export const jShapeHorizontal = new Shape([
  [1, 1, 1],
  [0, 0, 1],
]);

// Basic Set
// Basic Set+
// Standard Set
// Standard Set+
// Challenge Set
// Challenge Set+
// Expert Set
// Expert Set+
// Master Set
// Master Set+
// Tricky Set

export type ShapeSet = {
  shapeSetName: string;
  shapes: Shape[];
};

const SHAPES_1 = [squareShape, verticalShape, horizontalShape];
const SHAPES_1_PLUS = [...SHAPES_1, lShape];

export const basicSet: ShapeSet = {
  shapeSetName: 'Basic',
  shapes: SHAPES_1,
};

export const basicPlusSet: ShapeSet = {
  shapeSetName: 'Basic+',
  shapes: SHAPES_1_PLUS,
};

const SHAPES_2 = [...SHAPES_1, lShape, jShape];
const SHAPES_2_PLUS = [...SHAPES_2, tdownShape];

export const standardSet: ShapeSet = {
  shapeSetName: 'Standard',
  shapes: SHAPES_2,
};

export const standardPlusSet: ShapeSet = {
  shapeSetName: 'Standard+',
  shapes: SHAPES_2_PLUS,
};

const SHAPES_3 = [...SHAPES_2, tdownShape, tUpShape];
const SHAPES_3_PLUS = [...SHAPES_3, zShape];

export const challengeSet: ShapeSet = {
  shapeSetName: 'Challenge',
  shapes: SHAPES_3,
};

export const challengePlusSet: ShapeSet = {
  shapeSetName: 'Challenge+',
  shapes: SHAPES_3_PLUS,
};

const SHAPES_4 = [...SHAPES_3, zShape, sShape];
const SHAPES_4_PLUS = [...SHAPES_4, sShape];

const SHAPES_EXPERT = [
  squareShape,
  tdownShape,
  tUpShape,
  zShape,
  sShape,
  lShape,
  jShape,
  lShapeHorizontal,
  jShapeHorizontal,
];

export const expertSet: ShapeSet = {
  shapeSetName: 'Expert',
  shapes: SHAPES_EXPERT,
};

export const expertPlusSet: ShapeSet = {
  shapeSetName: 'Expert+',
  shapes: SHAPES_4_PLUS,
};

const SHAPES_MASTER = [squareShape, zShape, sShape, lShape, jShape, lShapeHorizontal, jShapeHorizontal];

const SHAPES_5 = [...SHAPES_4, lShapeHorizontal, jShapeHorizontal];

const SHAPES_TRICKY = [squareShape, zShape, sShape, lShape, jShape];

const SHAPES_Z = [zShape, sShape, zVerticalShape, sVerticalShape];

const SHAPES_ALL = SHAPES_5;

export const masterSet: ShapeSet = {
  shapeSetName: 'Master',
  shapes: SHAPES_MASTER,
};

export const trickySet: ShapeSet = {
  shapeSetName: 'Tricky',
  shapes: SHAPES_TRICKY,
};

export const zShapeSet: ShapeSet = {
  shapeSetName: 'ZigZag',
  shapes: SHAPES_Z,
};

export const allSet: ShapeSet = {
  shapeSetName: 'All',
  shapes: SHAPES_ALL,
};

export const MAX_SHAPE_HEIGHT = Math.max(...SHAPES_ALL.map((s) => s.height));
// export const MAX_SHAPE_WIDTH = Math.max(...ALL_SHAPES.map((s) => s.width));

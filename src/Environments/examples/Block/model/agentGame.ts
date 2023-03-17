import {
  PointsHeuristicValues,
  HeuristicValues,
} from '../constants/HeuristicValues';
import { LEVEL_MAP, LEVEL_NAME_KEY } from '../constants/Level';
import { getRandomElement } from '../util';
import { Queue } from '../util/queue';
import { Board } from './board';
import { GameConfig } from './gameConfig';
import { AgentGameInterface } from './gameInterface';
import { Move } from './move';
import { checkAddShapeValid, getPointsCopy } from './points';
import { renderPoints, Shape } from './shape';

export class AgentGame implements AgentGameInterface {
  config: GameConfig;
  points: number[][];
  savedPoints: number[][];
  savedShape: Shape;
  nextShape: Shape;
  nextShapeQueue: Queue<Shape> = new Queue();
  savedNextShapeQueueArr: Shape[] = [];
  pendingComputerMove: boolean;

  constructor(
    config: GameConfig,
    points: number[][],
    nextShape: Shape,
    nextShapeQueue: Queue<Shape>
  ) {
    this.config = config;
    this.points = points;
    this.savedPoints = [];
    this.nextShape = nextShape;
    this.savedShape = nextShape;
    this.nextShapeQueue = nextShapeQueue;
    this.savedNextShapeQueueArr = nextShapeQueue.toArray();
    this.pendingComputerMove = false;
  }

  resetStates(): void {
    this.nextShape = getRandomElement(this.config.shapes);
    this.savedShape = this.nextShape;
    this.pendingComputerMove = false;
  }

  clone(): AgentGameInterface {
    const newGame = new AgentGame(
      this.config,
      getPointsCopy(this.points),
      this.nextShape,
      this.nextShapeQueue.clone()
    );
    return newGame;
  }

  savePointsCopy(): void {
    this.savedPoints = getPointsCopy(this.points);
    this.savedShape = this.nextShape;
  }

  restorePointsCopy(): void {
    this.points = getPointsCopy(this.savedPoints);
    this.nextShape = this.savedShape;
  }

  saveStateCopy(): void {
    this.savedPoints = getPointsCopy(this.points);
    this.savedShape = this.nextShape;
    this.savedNextShapeQueueArr = this.nextShapeQueue.toArray();
  }

  restoreStateCopy(): void {
    this.points = getPointsCopy(this.savedPoints);
    this.nextShape = this.savedShape;
    this.nextShapeQueue = Queue.fromArray(this.savedNextShapeQueueArr);
  }

  clearBoardAnCells(): void {
    this.points = [];
    this.savedPoints = [];
  }

  getHeuristicValues(move: Move): HeuristicValues {
    // this is a procedure repeatedly called many times
    // avoid unnecessary varible assignments to improve performance
    // get all heuristics all at once to avoid looping through the same points multiple times
    const pointsCopy = getPointsCopy(this.points);
    const [success, pointsAdded] = Board._tryAddShape(
      move[0],
      move[1],
      this.nextShape,
      this.config.width,
      this.config.height,
      pointsCopy
    );

    let scoreDelta = pointsAdded.length * this.config.scorePointAddedMultiplier;

    const clearableRows: number[] = [];
    for (let i = 0; i < this.config.height; i++) {
      let clearable = true;
      for (let j = 0; j < pointsCopy[i].length; j++) {
        if (pointsCopy[i][j] !== 1) {
          clearable = false;
          break;
        }
      }
      if (clearable) {
        clearableRows.push(i);
      }
    }

    const clearableColumns: number[] = [];
    for (let i = 0; i < this.config.width; i++) {
      let clearable = true;
      for (let j = 0; j < pointsCopy.length; j++) {
        const cell = pointsCopy[j][i];
        if (cell !== 1) {
          clearable = false;
          break;
        }
      }
      if (clearable) {
        clearableColumns.push(i);
      }
    }

    let pointsCleared = 0;

    // clear lines
    for (let i = 0; i < clearableRows.length; i++) {
      pointsCleared += Board._clearRow(
        clearableRows[i],
        this.config.width,
        pointsCopy
      );
    }
    for (let i = 0; i < clearableColumns.length; i++) {
      pointsCleared += Board._clearColumn(
        clearableColumns[i],
        this.config.height,
        pointsCopy
      );
    }

    scoreDelta += pointsCleared * this.config.scorePointClearedMultiplier;

    let edges = 0;
    let numPossibleShapes = 0;
    let numTotalShapes = this.config.shapes.length;
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let k = 0; k < numTotalShapes; k++) {
      const shape = this.config.shapes[k];
      let valid = false;
      for (let i = 0; i < yMax; i++) {
        for (let j = 0; j < xMax; j++) {
          if (k === 0) {
            // count edge
            const element = pointsCopy[i][j];
            const above = i > 0 ? pointsCopy[i - 1][j] : 0;
            const below = i < yMax - 1 ? pointsCopy[i + 1][j] : 0;
            if (element === 1) {
              // block cell different from 2 adjacents
              if (above === 0) {
                if (!pointsCopy[i][j - 1]) {
                  edges++;
                }
                if (!pointsCopy[i][j + 1]) {
                  edges++;
                }
              }
              if (below === 0) {
                if (!pointsCopy[i][j + 1]) {
                  edges++;
                }
                if (!pointsCopy[i][j - 1]) {
                  edges++;
                }
              }
            } else if (element === 0) {
              // empty cell different from 3 adjacents
              if (above === 1) {
                if (
                  pointsCopy[i][j - 1] === 1 &&
                  pointsCopy[i - 1][j - 1] === 1
                ) {
                  edges++;
                }
                if (
                  pointsCopy[i][j + 1] === 1 &&
                  pointsCopy[i - 1][j + 1] === 1
                ) {
                  edges++;
                }
              }
              if (below === 1) {
                if (
                  pointsCopy[i][j - 1] === 1 &&
                  pointsCopy[i + 1][j - 1] === 1
                ) {
                  edges++;
                }
                if (
                  pointsCopy[i][j + 1] === 1 &&
                  pointsCopy[i + 1][j + 1] === 1
                ) {
                  edges++;
                }
              }
            }
          }
          if (
            !valid &&
            checkAddShapeValid(
              j,
              i,
              shape,
              this.config.width,
              this.config.height,
              pointsCopy
            )
          ) {
            valid = true;
          }
        }
      }
      if (valid) {
        numPossibleShapes++;
      }
    }

    const essRatio = numPossibleShapes / numTotalShapes;

    const linesCleared = clearableRows.length + clearableColumns.length;
    return {
      linesCleared,
      edges,
      essRatio,
      success,
      scoreDelta,
    };
  }

  getNextStateMutate(
    move: Move
  ): [valid: boolean, scoreDelta: number, coinDelta: number] {
    const [x, y] = move;
    const {
      points,
      nextShape,
      config: { width, height },
    } = this;
    // add shape
    const [success, pointsAdded] = Board._tryAddShape(
      x,
      y,
      this.nextShape,
      width,
      height,
      points
    );
    let scoreDelta = 0;
    let coinDelta = 0;
    if (success) {
      scoreDelta += pointsAdded.length * this.config.scorePointAddedMultiplier;
      coinDelta += pointsAdded.length * this.config.coinPointAddedMultiplier;
    }

    this.pendingComputerMove = true;
    return [success, scoreDelta, coinDelta];
  }

  computerMove(): [valid: boolean, scoreDelta: number, coinDelta: number] {
    this.nextShape = this.nextShapeQueue.dequeue();
    this.nextShapeQueue.enqueue(getRandomElement(this.config.shapes));
    let scoreDelta = 0;
    let coinDelta = 0;
    const pointsCleared = this.clearLine();
    scoreDelta += pointsCleared * this.config.scorePointClearedMultiplier;
    coinDelta += pointsCleared * this.config.coinPointClearedMultiplier;
    // clear line if successful
    this.pendingComputerMove = false;
    return [true, scoreDelta, coinDelta];
  }

  setupTest(number: number): void {
    // prettier ignore
    const points = [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    if (number === 0) {
      console.log('demo');
      this.config.width = 10;
      this.config.height = 10;
      this.points = points;
    }
  }

  startNewGame(): void {
    this.resetStates();
    this.clearBoardAnCells();
  }

  startLevel(level: LEVEL_NAME_KEY): void {
    console.log(`Update game config to ${level}`);
    this.config = LEVEL_MAP[level];

    console.log(
      `Start game for ${level} with config ${JSON.stringify(this.config)}`
    );
    this.startNewGame();
  }

  /**
   * Helper function used for the EES heuristic.
   * Returns True if the current shape can be fit in the board else False.
   * @param shape
   */
  isValidShape(shape: Shape): boolean {
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let x = 0; x < xMax; x++) {
      for (let y = 0; y < yMax; y++) {
        const valid = checkAddShapeValid(
          x,
          y,
          shape,
          this.config.width,
          this.config.height,
          this.points
        );
        if (valid) {
          return true;
        }
      }
    }
    return false;
  }

  getValidMoves(): Move[] {
    const {
      points,
      nextShape,
      config: { width, height },
    } = this;
    const validMoves: Move[] = [];
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let x = 0; x < xMax; x++) {
      for (let y = 0; y < yMax; y++) {
        const valid = checkAddShapeValid(
          x,
          y,
          this.nextShape,
          width,
          height,
          points
        );
        if (valid) {
          validMoves.push([x, y]);
        }
      }
    }
    return validMoves;
  }

  checkLose(): boolean {
    if (this.pendingComputerMove) {
      return false;
    }
    const {
      points,
      nextShape,
      config: { width, height },
    } = this;
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let x = 0; x < xMax; x++) {
      for (let y = 0; y < yMax; y++) {
        const valid = checkAddShapeValid(
          x,
          y,
          this.nextShape,
          width,
          height,
          points
        );
        if (valid) {
          return false;
        }
      }
    }
    return true;
  }

  getTextOutput(): string {
    return renderPoints(this.points, this.config.width, this.config.height);
  }

  print() {
    console.log(
      renderPoints(this.points, this.config.width, this.config.height)
    );
  }

  private clearLine() {
    if (this.config.clearLineHorizontal && this.config.clearLineVertical) {
      return this.clearLineBoth();
    } else if (this.config.clearLineHorizontal) {
      return this.clearLineHorizontal();
    } else if (this.config.clearLineVertical) {
      return this.clearLineVertical();
    }
    return 0;
  }

  private clearLineHorizontal() {
    let count = 0;
    const clearableRows = this.getClearableRows();
    for (let i = 0; i < clearableRows.length; i++) {
      count += Board._clearRow(
        clearableRows[i],
        this.config.width,
        this.points
      );
    }
    return count;
  }

  private clearLineVertical() {
    let count = 0;
    const clearableColumns = this.getClearableColumns();
    for (let i = 0; i < clearableColumns.length; i++) {
      count += Board._clearColumn(
        clearableColumns[i],
        this.config.height,
        this.points
      );
    }
    return count;
  }

  private clearLineBoth() {
    let count = 0;
    // need to calculate both in case of overlaps
    const clearableRows = this.getClearableRows();
    const clearableColumns = this.getClearableColumns();
    for (let i = 0; i < clearableRows.length; i++) {
      count += Board._clearRow(
        clearableRows[i],
        this.config.width,
        this.points
      );
    }
    for (let i = 0; i < clearableColumns.length; i++) {
      count += Board._clearColumn(
        clearableColumns[i],
        this.config.height,
        this.points
      );
    }
    return count;
  }

  private getClearableRows(): number[] {
    return Board._getClearableRows(this.config.height, this.points);
  }

  private getClearableColumns(): number[] {
    return Board._getClearableColumns(this.config.width, this.points);
  }
}

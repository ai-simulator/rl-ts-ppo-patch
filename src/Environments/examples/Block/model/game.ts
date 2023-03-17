import { GameHighScore } from '../constants/GameHighScore';
import { HeuristicValues } from '../constants/HeuristicValues';
import { LEVEL_MAP, LEVEL_NAME_KEY } from '../constants/Level';
import { getRandomElement } from '../util';
import { Queue } from '../util/queue';
import { AgentGame } from './agentGame';
import { Board, getEmptyBoard } from './board';
import { Cell } from './cell';
import { cellToDisplayCell, DisplayCell } from './displayCell';
import { GameConfig } from './gameConfig';
import { AgentGameInterface, DisplayGameInterface } from './gameInterface';
import { Move } from './move';
import { getPointsCopy } from './points';
import { renderShapeAsString, Shape } from './shape';

export type GameDisplayState = {
  width: number;
  height: number;
  cells: DisplayCell[];
  score: number;
  lastPointsAdded: [x: number, y: number][];
  nextShapeCells: DisplayCell[];
  nextNextShapeCells: DisplayCell[];
  nextShape: Shape;
  nextShapeQueue: Queue<Shape>;
  clearableRows: number[];
  clearableColumns: number[];
  plannedMove: Move | undefined;
};

export class Game implements DisplayGameInterface {
  config: GameConfig;
  gameId: number;
  lastMoveTime: number;
  score: number;
  coinsCollected: number;
  board: Board;
  nextShape: Shape;
  lastPointsAdded: [x: number, y: number][];
  cells!: Cell[][];
  message: string;
  moveCount: number;
  startTime: number;
  pendingComputerMove: boolean;
  plannedMove: Move | undefined;
  nextShapeQueue: Queue<Shape>;
  cellsCleared: number;

  constructor(config: GameConfig) {
    this.config = config;
    this.gameId = Date.now();
    this.lastMoveTime = Date.now();
    this.score = 0;
    this.coinsCollected = 0;
    this.board = getEmptyBoard(config.width, config.height);
    this.nextShape = getRandomElement(this.config.shapes);
    this.nextShapeQueue = new Queue();
    while (this.nextShapeQueue.size() < config.nextShapeQueueCount) {
      this.nextShapeQueue.enqueue(getRandomElement(this.config.shapes));
    }
    this.lastPointsAdded = [];
    // this.cells = this.board.getCells(this.gameId);
    this.message = '';
    this.moveCount = 0;
    this.startTime = Date.now();
    this.pendingComputerMove = false;
    this.plannedMove = undefined;
    this.cellsCleared = 0;
  }

  resetStates(): void {
    this.gameId = Date.now();
    this.lastMoveTime = Date.now();
    this.score = 0;
    this.coinsCollected = 0;
    this.nextShape = getRandomElement(this.config.shapes);
    this.nextShapeQueue = new Queue();
    while (this.nextShapeQueue.size() < this.config.nextShapeQueueCount) {
      this.nextShapeQueue.enqueue(getRandomElement(this.config.shapes));
    }
    this.lastPointsAdded = [];
    this.message = '';
    this.moveCount = 0;
    this.startTime = Date.now();
    this.pendingComputerMove = false;
    this.plannedMove = undefined;
    this.cellsCleared = 0;
  }

  getAgentGame(): AgentGameInterface {
    const newGame = new AgentGame(
      this.config,
      getPointsCopy(this.board.points),
      this.nextShape,
      this.nextShapeQueue.clone()
    );
    return newGame;
  }

  clearBoardAnCells(): void {
    this.board = getEmptyBoard(this.config.width, this.config.height);
    this.cells = this.board.getCells(this.gameId);
  }

  setPlannedMove(move: Move) {
    this.lastMoveTime = Date.now();
    this.plannedMove = move;
  }

  removePlannedMove() {
    this.lastMoveTime = Date.now();
    this.plannedMove = undefined;
  }

  updateCells(): void {
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let x = 0; x < xMax; x++) {
      for (let y = 0; y < yMax; y++) {
        this.cells[y][x].type = this.board.points[y][x];
      }
    }
  }

  getLastGameHighScore(): GameHighScore {
    return {
      score: this.score,
      moveCount: this.moveCount,
      time: Date.now(),
      h: this.config.height + '',
      w: this.config.width + '',
      durSec: this.getGameDurationSeconds() + '',
      cellsCleared: this.cellsCleared,
    };
  }

  private getGameDurationSeconds() {
    return Math.round((Date.now() - this.startTime) / 1000);
  }

  getGameDisplayState(): GameDisplayState {
    this.updateCells();
    return {
      width: this.config.width,
      height: this.config.height,
      // TODO: optimize this while force re-render
      cells: this.cells.flat(1).map(cellToDisplayCell),
      score: this.score,
      lastPointsAdded: this.lastPointsAdded,
      // TODO: optimize this while force re-render
      nextShapeCells: this.nextShape.getCells(this.gameId).flat(1).map(cellToDisplayCell),
      nextNextShapeCells: this.nextShapeQueue.front().getCells(this.gameId).flat(1).map(cellToDisplayCell),
      nextShape: this.nextShape,
      nextShapeQueue: this.nextShapeQueue,
      clearableRows: this.getClearableRows(),
      clearableColumns: this.getClearableColumns(),
      plannedMove: this.plannedMove,
    };
  }

  getHeuristicValues(move: Move): HeuristicValues {
    const [x, y] = move;
    const boardCopy = this.board.clone();
    const [success, pointsAdded] = boardCopy.tryAddShape(x, y, this.nextShape);

    let scoreDelta = pointsAdded.length * this.config.scorePointAddedMultiplier;

    const clearableHorizontal = boardCopy.getClearableRows();
    const clearableVertical = boardCopy.getClearableColumns();

    let pointsCleared = 0;

    // clear lines
    for (let i = 0; i < clearableHorizontal.length; i++) {
      pointsCleared += boardCopy.clearRow(clearableHorizontal[i]);
    }
    for (let i = 0; i < clearableVertical.length; i++) {
      pointsCleared += boardCopy.clearColumn(clearableVertical[i]);
    }

    scoreDelta += pointsCleared * this.config.scorePointClearedMultiplier;

    // For EES: can fit other shapes after adding piece and clearing board
    let numPossibleShapes = 0;
    let numTotalShapes = this.config.shapes.length;
    for (let i = 0; i < numTotalShapes; i++) {
      const shape = this.config.shapes[i];
      if (this.isValidShape(boardCopy, shape)) {
        numPossibleShapes++;
      }
    }
    const essRatio = numPossibleShapes / numTotalShapes;

    const linesCleared = clearableHorizontal.length + clearableVertical.length;
    const edges = boardCopy.getEdgeCount();
    return {
      linesCleared,
      edges,
      essRatio,
      success,
      scoreDelta,
    };
  }

  getNextStateMutate(move: Move): [valid: boolean, scoreDelta: number, coinDelta: number] {
    const [x, y] = move;
    // add shape
    const [success, pointsAdded] = this.board.tryAddShape(x, y, this.nextShape);
    let scoreDelta = 0;
    let coinDelta = 0;
    if (success) {
      scoreDelta += pointsAdded.length * this.config.scorePointAddedMultiplier;
      coinDelta += pointsAdded.length * this.config.coinPointAddedMultiplier;
      this.score += scoreDelta;
      this.coinsCollected += coinDelta;
      this.lastPointsAdded = pointsAdded;
      this.moveCount++;
    }

    this.lastMoveTime = Date.now();
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
    this.score += scoreDelta;
    this.coinsCollected += coinDelta;
    this.cellsCleared += pointsCleared;
    // clear line if successful
    this.lastMoveTime = Date.now();
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
      this.board.points = points;
    }
  }

  startNewGame(): void {
    this.resetStates();
    this.clearBoardAnCells();
  }

  startLevel(level: LEVEL_NAME_KEY): void {
    console.log(`Update game config to ${level}`);
    this.config = LEVEL_MAP[level];

    console.log(`Start game for ${level} with config ${JSON.stringify(this.config)}`);
    this.startNewGame();
  }

  /**
   * Helper function used for the EES heuristic.
   * Returns True if the current shape can be fit in the board else False.
   * @param shape
   */
  isValidShape(board: Board, shape: Shape): boolean {
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let x = 0; x < xMax; x++) {
      for (let y = 0; y < yMax; y++) {
        const valid = board.checkAddShapeValid(x, y, shape);
        if (valid) {
          return true;
        }
      }
    }
    return false;
  }

  getValidMoves(): Move[] {
    const validMoves: Move[] = [];
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let x = 0; x < xMax; x++) {
      for (let y = 0; y < yMax; y++) {
        const valid = this.board.checkAddShapeValid(x, y, this.nextShape);
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
    const xMax = this.config.width;
    const yMax = this.config.height;
    for (let x = 0; x < xMax; x++) {
      for (let y = 0; y < yMax; y++) {
        const valid = this.board.checkAddShapeValid(x, y, this.nextShape);
        if (valid) {
          return false;
        }
      }
    }
    return true;
  }

  getTextOutput(): string {
    return renderShapeAsString(this.board);
  }

  print() {
    this.board.print();
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
      count += this.board.clearRow(clearableRows[i]);
    }
    return count;
  }

  private clearLineVertical() {
    let count = 0;
    const clearableColumns = this.getClearableColumns();
    for (let i = 0; i < clearableColumns.length; i++) {
      count += this.board.clearColumn(clearableColumns[i]);
    }
    return count;
  }

  private clearLineBoth() {
    let count = 0;
    // need to calculate both in case of overlaps
    const clearableRows = this.getClearableRows();
    const clearableColumns = this.getClearableColumns();
    for (let i = 0; i < clearableRows.length; i++) {
      count += this.board.clearRow(clearableRows[i]);
    }
    for (let i = 0; i < clearableColumns.length; i++) {
      count += this.board.clearColumn(clearableColumns[i]);
    }
    return count;
  }

  private getClearableRows(): number[] {
    return this.board.getClearableRows();
  }

  private getClearableColumns(): number[] {
    return this.board.getClearableColumns();
  }
}

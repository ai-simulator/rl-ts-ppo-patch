import { CellType } from '../constants/CellType';

export class Cell {
  x: number;
  y: number;
  type: CellType;
  id: string;

  constructor(x: number, y: number, type: CellType, gameId: number) {
    this.type = type;
    this.x = x;
    this.y = y;
    this.id = `${type}-${gameId}-${x}-${y}`;
  }
}

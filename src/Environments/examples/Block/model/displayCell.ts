import { CellType } from '../constants/CellType';
import { Cell } from './cell';

// optimized cell data structure for display and re-rendering

export type DisplayCell = [x: number, y: number, type: CellType, id: string];

export function cellToDisplayCell(cell: Cell): DisplayCell {
  return [cell.x, cell.y, cell.type, cell.id];
}

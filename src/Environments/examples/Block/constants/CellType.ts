export enum CellType {
  Empty = 0,
  Block = 1,
  Wall = 2,
}

export const CellTypeCount = 2;

export function CellTypeDisplayNameMapping(cellType: CellType): string {
  switch (cellType) {
    case CellType.Empty:
      return 'Empty';
    case CellType.Block:
      return 'Block';
    case CellType.Wall:
      return 'Wall';
  }
}

// export class BoardCell {
//   type: CellType;
//   id: string;

//   constructor(x: number, y: number, type: CellType, gameId: number) {
//     this.type = type;
//     this.id = `${type}-${gameId}-${x}-${y}`;
//   }

//   is(type: CellType) {
//     return this.type === type;
//   }
// }

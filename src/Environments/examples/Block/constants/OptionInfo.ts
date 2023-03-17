import { OPTION_KEY } from './Upgrade';

export interface OPTION_INFO {
  name: string;
  cost: number;
  description?: string;
  wip?: boolean;
  configs?: readonly string[];
  prerequisites?: OPTION_KEY[];
  next?: OPTION_KEY;
}

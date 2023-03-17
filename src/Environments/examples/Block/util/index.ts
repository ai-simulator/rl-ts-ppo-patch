export function getRandomElement<T>(array: T[]) {
  return array[Math.floor(Math.random() * array.length)];
}

export const formatNumberDecimalIfNecessary = (n: number, decimal: number): string => {
  return +n.toFixed(decimal) + '';
};

export const formatBoolean = (n: number) => (n ? 'True' : 'False');

export const formatNumber = (n: number) => {
  if (n < 1e3) return n + '';
  if (n >= 1e3 && n < 1e6) return +Math.floor((n / 1e3) * 100) / 100 + 'K';
  if (n >= 1e6 && n < 1e9) return +Math.floor((n / 1e6) * 100) / 100 + 'M';
  if (n >= 1e9 && n < 1e12) return +Math.floor((n / 1e9) * 100) / 100 + 'B';
  if (n >= 1e12) return +Math.floor((n / 1e12) * 100) / 100 + 'T';
  return n + '';
};

export const formatNumberWithDecimals = (n: number) => {
  if (n < 1e3) return n + '';
  if (n >= 1e3 && n < 1e6) return (Math.floor((n / 1e3) * 100) / 100).toFixed(2) + 'K';
  if (n >= 1e6 && n < 1e9) return (Math.floor((n / 1e6) * 100) / 100).toFixed(2) + 'M';
  if (n >= 1e9 && n < 1e12) return (Math.floor((n / 1e9) * 100) / 100).toFixed(2) + 'B';
  if (n >= 1e12) return (Math.floor((n / 1e12) * 100) / 100).toFixed(2) + 'T';
  return n + '';
};

export function getDateStr() {
  return new Date().toISOString().slice(0, 10);
}

export function formatDateStr(date: number) {
  return new Date(date).toISOString().slice(0, 10);
}

export function capitalizeFirstLetter(str: string) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function average(array: number[]) {
  let total = 0;
  for (let i = 0; i < array.length; i++) {
    total += array[i];
  }
  return total / array.length;
}

export function formatLog(index: number, log: string) {
  return `[${String(index).padStart(4, '0')}]${log}`;
}

export function formatTimeSeconds(seconds: number) {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m${seconds % 60}s`;
}

export function getRandomInteger(min: number, max: number): number {
  return Math.floor((max - min) * Math.random()) + min;
}

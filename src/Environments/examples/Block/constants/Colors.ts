const monoMain = '#272727';
const monoMainAlt = '#414141';
const monoSecondary = '#666666';
const monoOpposite = '#f5f5f5';
const monoOppositeDarken1 = '#dcdcdc';
const monoOppositeDarken2 = '#c2c2c2';
const monoOppositeDarken3 = '#939393';

export const transparentColor = 'transparent';

const colorMaps = {
  light: {
    display: '#f3b27a',
    primary: '#f3b27a',
    highlight: '#ef974b',
    secondary: '#8f7a66',
    text: '#8f7a66',
    cellBg: '#e6e0d9',
    secondary1: '#f9f6f2',
    secondary2: '#e6e0d9',
    secondary3: '#cfc5bc',
    secondary4: '#776e65',
    base: '#fff',
    transparent: 'transparent',
    locked: '#cccccc',
    primaryAlpha: addAlpha('#f3b27a', 0.7),
  },
  dark: {
    display: '#0a0a0a',
    primary: '#bbbbbb',
    highlight: '#c1c1c1',
    secondary: '#f6f6f6',
    text: '#f6f6f6',
    cellBg: '#0a0a0a',
    secondary1: '#292929',
    secondary2: '#3d3d3d',
    secondary3: '#525252',
    secondary4: '#666666',
    base: '#0a0a0a',
    transparent: 'transparent',
    locked: '#b9b9b9',
    primaryAlpha: addAlpha('#bbbbbb', 0.7),
  },
  invert: {
    display: '#6d5d4e',
    primary: '#f3b27a',
    highlight: '#ef974b',
    secondary: '#e6e0d9',
    text: '#e6e0d9',
    cellBg: '#8f7a66',
    secondary1: '#6d5d4e',
    secondary2: '#9b8773',
    secondary3: '#dcd3ca',
    secondary4: '#f1eae1',
    base: '#332c25',
    transparent: 'transparent',
    locked: '#afafaf',
    primaryAlpha: addAlpha('#f3b27a', 0.7),
  },
  retro: {
    display: '#de7e44',
    primary: '#de7e44',
    highlight: '#7c6593',
    secondary: '#7c6593',
    text: '#406a5a',
    cellBg: '#74ab97',
    secondary1: '#a5c9bc',
    secondary2: '#80b3a0',
    secondary3: '#5d9b84',
    secondary4: '#568f7a',
    base: '#fff',
    transparent: 'transparent',
    locked: '#cccccc',
    primaryAlpha: addAlpha('#de7e44', 0.7),
  },
  mono: {
    display: '#ffffff',
    primary: monoMain,
    highlight: '#0a0a0a',
    secondary: monoMainAlt,
    text: monoMain,
    cellBg: monoOppositeDarken1,
    secondary1: monoOpposite,
    secondary2: monoOppositeDarken2,
    secondary3: monoMain,
    secondary4: monoMain,
    base: monoOpposite,
    transparent: 'transparent',
    locked: monoSecondary,
    primaryAlpha: monoOppositeDarken3,
  },
};

export const COLOR_SCHEMES = [
  'light',
  'dark',
  'invert',
  'retro',
  'mono',
] as const;
export const DEFAULT_COLOR_SCHEME = 'light';
export type COLOR_SCHEME = typeof COLOR_SCHEMES[number];

let Colors: Record<COLOR_SCHEME, Record<string, string>> = {
  light: {},
  dark: {},
  invert: {},
  retro: {},
  mono: {},
};

for (let i = 0; i < COLOR_SCHEMES.length; i++) {
  const colorScheme = COLOR_SCHEMES[i];
  Colors[colorScheme] = {
    display: colorMaps[colorScheme].display,
    backgroundGlobal: colorMaps[colorScheme].secondary1,
    text: colorMaps[colorScheme].text,
    textInverse: colorMaps[colorScheme].base,
    modalMask: colorMaps[colorScheme].base,
    transparent: colorMaps[colorScheme].transparent,
    // section
    sectionBorder: colorMaps[colorScheme].secondary,
    sectionIconColor: colorMaps[colorScheme].text,

    // board
    primaryCellBackground: colorMaps[colorScheme].primary,
    primaryBackground: colorMaps[colorScheme].primary,
    highlightBackground: colorMaps[colorScheme].highlight,
    path: addAlpha(colorMaps[colorScheme].primary, 0.8),
    boardBackground: transparentColor,
    boardGridlines: transparentColor,
    emptyCellBackground: colorMaps[colorScheme].cellBg,
    secondaryCellBackground: colorMaps[colorScheme].primaryAlpha,
    wallCellBackground: colorMaps[colorScheme].locked,
    // segment
    segmentBackground: colorMaps[colorScheme].secondary2,
    // buttons
    primaryAction: colorMaps[colorScheme].primary,
    secondaryAction: colorMaps[colorScheme].secondary,
    // option states
    enabled: colorMaps[colorScheme].secondary,
    selectedBorder: colorMaps[colorScheme].secondary,
    disabled: colorMaps[colorScheme].transparent,
    locked: colorMaps[colorScheme].locked,
    // highscore table
    backgroundAlt: colorMaps[colorScheme].secondary2,
    // info box
    boxBackground: colorMaps[colorScheme].transparent,
    descBoxBackground: colorMaps[colorScheme].base,
    boxBoarder: colorMaps[colorScheme].secondary2,
    boxTitleText: colorMaps[colorScheme].secondary,
    boxValueBackground: colorMaps[colorScheme].secondary1,
    boxValueText: colorMaps[colorScheme].secondary4,
    inputPlaceHolderText: colorMaps[colorScheme].secondary3,
    // slider
    line: colorMaps[colorScheme].secondary3,
    lineAlt: colorMaps[colorScheme].secondary3,
    sliderBtn: colorMaps[colorScheme].primary,
    // scrollbar
    scrollbarBackground: colorMaps[colorScheme].secondary2,
    scrollbar: colorMaps[colorScheme].secondary3,
    // background icon
    backgroundIcon: colorMaps[colorScheme].secondary2,
    // draggable
    draggableBackground: colorMaps[colorScheme].secondary,
    draggableBackgroundActive: colorMaps[colorScheme].primary,
    // progress
    progressPrimary: colorMaps[colorScheme].primary,
  };
}

export default Colors;

function addAlpha(color: string, opacity: number) {
  // coerce values so ti is between 0 and 1.
  var _opacity = Math.round(Math.min(Math.max(opacity || 1, 0), 1) * 255);
  return color + _opacity.toString(16).toUpperCase();
}

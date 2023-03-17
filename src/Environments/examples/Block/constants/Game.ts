export const HUMAN_MOVE_DELAY = 300;

export let DEBUG = false;
export let DEBUG_PERFORMANCE = false;
export let DEBUG_SENTRY = false;
export let DEBUG_APPSFLYER = false;
export let CLEAR = false;
export let DISABLE_ANIMATION = false;
export let TEXT_UI = false;
export let NO_UI = false;
export let LOG_PERFORMANCE = false;
export let DEMO = false;
export let LOW_LIMIT = false;
export let LOW_SCORE_THRESHOLD = false;
export let SECRET_THRESHOLD = 15;
export let TEST_TRAIN = false;
export let DISABLE_ADS = false;
export let DEBUG_EMULATOR = false;
export let CHEAP_PURCHASE = false;

// if (typeof __DEV__ !== 'undefined') {
//   if (__DEV__) {
//     DEBUG_SENTRY = true;
//     DEBUG_APPSFLYER = true;
//     DEBUG = false;
//     CLEAR = false;
//     TEXT_UI = false;
//     DISABLE_ANIMATION = false;
//     LOG_PERFORMANCE = true;
//     DEBUG_PERFORMANCE = false;
//     DEMO = false;
//     SECRET_THRESHOLD = 3;
//     LOW_LIMIT = false;
//     TEST_TRAIN = false;
//     LOW_SCORE_THRESHOLD = true;
//     DISABLE_ADS = false;
//     DEBUG_EMULATOR = true;
//     CHEAP_PURCHASE = true;
//   }
// }

export const TRAINING = false;
export let LOG_PERF_INTERVAL = 1000;

if (TRAINING) {
  TEXT_UI = false;
  NO_UI = true;
  LOG_PERFORMANCE = true;
  // LOG_PERF_INTERVAL = 10000;
  // LOG_PERF_INTERVAL = 100000;
  LOG_PERF_INTERVAL = 200000;
}

// export const REQUEST_REVIEW_INTERVAL = 1000 * 60 * 60 * 24 * 7;
export const REQUEST_REVIEW_INTERVAL = 1000 * 60 * 60 * 24;

export const DISCORD_INVITE_URL = 'https://discord.gg/2hnregZTz5';

export const AI_TERM_URL_PREFIX = 'https://ai-simulator.com/terms/';

export const PRIVACY_POLICY_LINK = 'https://ai-simulator.com/privacy-block-puzzle/';

export const PATCH_LINK = 'https://ai-simulator.com/patch-notes/';

export const TEXT_SEPARATOR = '·';

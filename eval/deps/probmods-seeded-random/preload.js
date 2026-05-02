// Preloaded by node (via NODE_OPTIONS='--require <this>') BEFORE webppl's
// own modules load. Overrides Math.random so that any downstream module
// capturing `const rnd = Math.random` during its own init picks up our
// deterministic override, not the original native random.
//
// Seed comes from WEBPPL_MATH_RANDOM_SEED env var (defaults to 42).

var raw = process.env.WEBPPL_MATH_RANDOM_SEED;
var seed = parseInt(raw, 10);
if (!Number.isFinite(seed)) { seed = 42; }
if (seed === 0) { seed = 1; }

var state = seed >>> 0;
Math.random = function() {
  state = (state + 0x6D2B79F5) | 0;
  var t = state;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

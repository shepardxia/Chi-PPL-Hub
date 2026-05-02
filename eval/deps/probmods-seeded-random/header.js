// Kept as a minimal WebPPL package so `eval/deps/probmods-seeded-random/`
// is a self-contained dep directory. The actual Math.random override
// happens in preload.js, loaded via `node -r` BEFORE WebPPL's modules
// initialize — overriding from a package header would be too late since
// WebPPL and its deps may capture a reference to Math.random during their
// own module-init.

module.exports = function(env) {
  return {};
};

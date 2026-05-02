// Loads probmods2's bundled box2d.js + physics.js (as shipped to readers
// via `custom_js` in chapter frontmatter) in a vm sandbox with minimal
// browser-shim globals. Exposes `physics`, `worldWidth`, `worldHeight`
// to WebPPL. `physics.run(steps, world)` works (headless simulation);
// `physics.animate` is a no-op since there's no canvas.

var fs = require("fs");
var path = require("path");
var vm = require("vm");

// Minimal jQuery-like stub. physics.js only uses $-chains inside animate(),
// which we no-op; but the script body references $("<...>") during import
// via animate's definition, so we keep the stub behavior safe.
var jQueryStub = function() { return { appendTo: function() { return this; },
  attr: function() { return this; }, click: function() { return this; },
  append: function() { return this; } }; };

module.exports = function(env) {
  var projectRoot = path.resolve(__dirname, "..", "..", "..");
  var dataDir = path.join(projectRoot, "data/sources/probmods2/assets/js");
  var lodash = require(path.join(projectRoot, "data/sources/probmods2/node_modules/lodash"));

  // Context uses the vm's own intrinsics. Overriding Function/Array/etc.
  // breaks cross-realm instanceof inside box2d's constructor machinery.
  var ctx = vm.createContext({ console: console });
  vm.runInContext(
    "var window = this; var global = this;" +
    "var setTimeout = function(){}; var clearTimeout = function(){};" +
    "var requestAnimationFrame = function(){ return 0; };" +
    "var cancelAnimationFrame = function(){};",
    ctx
  );
  // Inject lodash and jQuery-stub as sandbox globals.
  ctx._ = lodash;
  ctx.$ = jQueryStub;
  ctx.jQuery = jQueryStub;
  ctx.wpEditor = { makeResultContainer: function() { return jQueryStub(); } };

  vm.runInContext(fs.readFileSync(path.join(dataDir, "box2d.js"), "utf8"), ctx);
  vm.runInContext(fs.readFileSync(path.join(dataDir, "physics.js"), "utf8"), ctx);

  var result = {};
  if (ctx.physics) {
    // Replace animate with a no-op (it needs DOM/canvas).
    ctx.physics.animate = function() { return []; };
    result.physics = ctx.physics;
  }
  if (ctx.worldWidth !== undefined) result.worldWidth = ctx.worldWidth;
  if (ctx.worldHeight !== undefined) result.worldHeight = ctx.worldHeight;
  return result;
};

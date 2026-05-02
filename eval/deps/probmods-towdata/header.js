var fs = require("fs");
var path = require("path");
var vm = require("vm");

module.exports = function(env) {
  // Navigate from eval/deps/probmods-towdata/ to project root
  var projectRoot = path.resolve(__dirname, "..", "..", "..");
  var dataDir = path.join(projectRoot, "data/sources/probmods2/assets/js");

  var ctx = {};
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(dataDir, "towData.js"), "utf8"), ctx);
  vm.runInContext(fs.readFileSync(path.join(dataDir, "towConfigurations.js"), "utf8"), ctx);

  var result = {};
  if (ctx.towData) result.towData = ctx.towData;
  if (ctx.towMeans) result.towMeans = ctx.towMeans;
  if (ctx.towConfigurations) result.towConfigurations = ctx.towConfigurations;
  return result;
};

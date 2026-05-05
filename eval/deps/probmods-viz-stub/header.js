// Headless no-op shim for browser-only display helpers used in DIPPL /
// probmods source corpora: `viz(...)`, `viz.<method>(...)`, `drawLines`,
// `drawPoints`, `drawPolygon`, `loadImage`, `print`, `display`.
//
// IMPORTANT: WebPPL CPS-transforms in-program function calls. Foreign
// functions exposed via package headers MUST follow the calling
// convention `f(s, k, a, ...args)` where `s` is the store, `k` is the
// continuation, and `a` is the address. They must call `k(s, result)`
// to continue execution; returning a plain value does not work
// (program halts silently).
//
// All side-effect helpers below ignore their arguments and call k with
// undefined. `print`/`display` pass their argument through to k so they
// can be used as identity (e.g., `print(value)` returns `value`).

module.exports = function(env) {
  var noop = function(s, k /*, a, ...args */) {
    return k(s, undefined);
  };
  var identity = function(s, k, a, x) {
    return k(s, x);
  };

  // viz is callable AND has method properties. NB: WebPPL CPS-transforms
  // bare identifier calls (`viz(x)`) but treats member calls (`viz.bar(x)`)
  // as plain host JS, with no continuation argument. So `viz` itself takes
  // the CPS signature, while `viz.<method>` takes the plain signature.
  var viz = function(s, k /*, a, ...args */) {
    return k(s, undefined);
  };
  var plainNoop = function() { return undefined; };
  viz.table = plainNoop;
  viz.bar = plainNoop;
  viz.line = plainNoop;
  viz.scatter = plainNoop;
  viz.hist = plainNoop;
  viz.heatMap = plainNoop;
  viz.density = plainNoop;
  viz.auto = plainNoop;
  viz.marginals = plainNoop;

  return {
    viz: viz,
    drawLines: noop,
    drawPoints: noop,
    drawPolygon: noop,
    loadImage: function(s, k /*, a, ...args */) {
      return k(s, {});
    },
    print: identity,
    display: identity,
  };
};

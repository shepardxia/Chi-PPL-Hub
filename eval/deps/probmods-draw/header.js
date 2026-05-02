// Headless no-op shim for the book's `Draw(w, h, visible)` canvas helper.
// The real thing (data/sources/probmods2/assets/js/draw.js) uses paper.js
// + a DOM canvas + jQuery — none of that works without a browser. For the
// eval harness we just need method calls to not error. Drawings are
// discarded; return values are undefined but chainable where needed.
//
// Methods observed in probmods2 usage: circle, rectangle, line. Extend
// if new ones appear. Unknown method calls will throw TypeError and that
// will surface as an unclassified runtime error (which is the signal we
// want — "add this method to the shim").

module.exports = function(env) {
  var noop = function() { return; };
  var makeCanvas = function(/* w, h, visible */) {
    return {
      circle: noop,
      rectangle: noop,
      line: noop,
      squiggle: noop,
      arc: noop,
      triangle: noop,
      polyline: noop,
      redraw: noop,
      destroy: noop,
      newPath: function() { return { add: noop, remove: noop }; },
      newPoint: function(x, y) { return { x: x, y: y }; },
      text: noop,
      polygon: noop,
    };
  };
  return { Draw: makeCanvas };
};

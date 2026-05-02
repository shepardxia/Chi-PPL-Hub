// JS header: defines globals that must NOT go through WebPPL's CPS transform.
// MCMC_Callbacks is called from JS-land (callbacks.js), so its `finish`
// function must be a plain JS function, not a CPS-transformed WebPPL one.

module.exports = function(env) {
  var noop = function() { return noop; };

  // editor: inter-block state passing (editor.put/get) + progress callback
  var editorStore = {};
  var editor = {
    put: function(k, v) { editorStore[k] = v; },
    get: function(k) { return editorStore[k]; },
    MCMCProgress: function() { return { finish: function() {} }; }
  };

  return {
    MCMC_Callbacks: {
      finalAccept: {
        finish: function(trace) {
          var ratio = (trace.info.total === 0) ? 0 : trace.info.accepted / trace.info.total;
          console.log('Acceptance ratio: ' + ratio);
        }
      }
    },
    editor: editor
  };
};

"""Batch B: process-models (4) + occams-razor (5) + social-cognition (6) = 15 atoms.

All atoms now end with the canonical answer expression."""
from atom_writer import write_atoms

ATOMS = []

# ─── process-models.md ──────────────────────────────────────────────
BLICKET_FOLD = (
    "var timeIt = function(func) {\n"
    "  var start = _.now();\n"
    "  func();\n"
    "  var end = _.now();\n"
    "  return end - start;\n"
    "};\n\n"
    "var detectingBlickets = function(evidence, baseRate, numSamples) {\n"
    "  return Infer({method: 'rejection', samples: numSamples}, function() {\n"
    "    var blicket = mem(function(block) { flip(baseRate) });\n"
    "    var power = function(block) { blicket(block) ? .95 : .05 };\n"
    "    var machineBeeps = function(blocks) {\n"
    "      blocks.length == 0\n"
    "        ? flip(0.05)\n"
    "        : flip(power(first(blocks))) || machineBeeps(rest(blocks))\n"
    "    };\n"
    "    condition(machineBeeps(evidence));\n"
    "    return blicket('A');\n"
    "  })\n"
    "};\n\n"
    "var marsData = [\n"
    "  {subjectID: 1, evidence: ['A'], response: true, RT: .9},\n"
    "  {subjectID: 1, evidence: ['A', 'B', 'C', 'D', 'E', 'F'], response: true, RT: 1.1},\n"
    "  {subjectID: 1, evidence: ['A', 'B', 'C'], response: true, RT: 1.2},\n"
    "  {subjectID: 2, evidence: ['A'], response: true, RT: 3.5},\n"
    "  {subjectID: 2, evidence: ['A', 'B', 'C', 'D', 'E', 'F'], response: false, RT: 4},\n"
    "  {subjectID: 2, evidence: ['A', 'B', 'C'], response: true, RT: 3.4},\n"
    "];\n\n"
    "var getModelRT = function(func, numRepeats) {\n"
    "  var rt = repeat(numRepeats, function() { timeIt(func) });\n"
    "  return Gaussian({mu: listMean(rt), sigma: Math.max(listVar(rt), 1)});\n"
    "};\n\n"
)
BLICKET_PROMPT_INTRO = (
    "Consider this Mars data set with response times for the blicket-detector task. "
    "Participants make inferences about a base rate by sampling some number of times: "
    "more samples = more accurate but slower (longer RT in ms).\n\n"
    "```\n"
    "var detectingBlickets = function(evidence, baseRate, numSamples) {\n"
    "  return Infer({method: 'rejection', samples: numSamples}, function() {\n"
    "    var blicket = mem(function(block) { flip(baseRate) });\n"
    "    var power = function(block) { blicket(block) ? .95 : .05 };\n"
    "    var machineBeeps = function(blocks) {\n"
    "      blocks.length == 0 ? flip(0.05) :\n"
    "        flip(power(first(blocks))) || machineBeeps(rest(blocks))\n"
    "    };\n"
    "    condition(machineBeeps(evidence));\n"
    "    return blicket('A');\n"
    "  })\n"
    "};\n"
    "var marsData = [/* per-trial {subjectID, evidence, response, RT} records */];\n"
    "var getModelRT = function(func, numRepeats) {\n"
    "  var rt = repeat(numRepeats, function() { timeIt(func) });\n"
    "  return Gaussian({mu: listMean(rt), sigma: Math.max(listVar(rt), 1)});\n"
    "};\n"
    "```\n\n"
)

ATOMS.append({
    "id": "probmods2-process-models/ex1",
    "source": "exercises/process-models.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": BLICKET_PROMPT_INTRO + (
        "Write a `dataAnalysis` function that infers the joint posterior over "
        "`baseRate` and `numSamples` by observing both each participant's `response` and "
        "`RT` for each datum in `marsData`. End with `Infer({method: 'MCMC', samples: "
        "500, burn: 100}, dataAnalysis)`."
    ),
    "groundtruth_code": (
        BLICKET_FOLD +
        "var dataAnalysis = function() {\n"
        "  var baseRate = uniform(0, 1);\n"
        "  var numSamples = randomInteger(100) + 1;\n\n"
        "  map(function(datapoint) {\n"
        "    var blicketModel = function() { \n"
        "      return detectingBlickets(datapoint.evidence, baseRate, numSamples)\n"
        "    };\n"
        "    observe(blicketModel(), datapoint.response);\n"
        "    observe(getModelRT(blicketModel, 10), datapoint.RT);\n"
        "  }, marsData);\n\n"
        "  return {baseRate, numSamples};\n"
        "};\n\n"
        "Infer({method: 'MCMC', samples: 500, burn: 100}, dataAnalysis);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-process-models/ex3",
    "source": "exercises/process-models.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": BLICKET_PROMPT_INTRO + (
        "There is some subject variability in RT. Modify the `dataAnalysis` model to "
        "allow each subject to have their own `baseRate` (use `mem` keyed on subjectID). "
        "Return `{subject1, subject2, numSamples}`. End with the Infer(...)."
    ),
    "groundtruth_code": (
        BLICKET_FOLD +
        "var dataAnalysis = function() {\n"
        "  var baseRate = mem(function(subjectID) { uniform(0, 1) });\n"
        "  var numSamples = randomInteger(100) + 1;\n\n"
        "  map(function(datapoint) {\n"
        "    var blicketModel = function() { \n"
        "      return detectingBlickets(datapoint.evidence, baseRate(datapoint.subjectID), numSamples)\n"
        "    };\n"
        "    observe(blicketModel(), datapoint.response);\n"
        "    observe(getModelRT(blicketModel, 10), datapoint.RT);\n"
        "  }, marsData);\n\n"
        "  return {subject1: baseRate(1), subject2: baseRate(2), numSamples: numSamples};\n"
        "};\n\n"
        "Infer({method: 'MCMC', samples: 500, burn: 100}, dataAnalysis);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-process-models/ex4",
    "source": "exercises/process-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": BLICKET_PROMPT_INTRO + (
        "Run the same per-subject BDA on Venus data instead of Mars:\n\n"
        "```\n"
        "var venusData = [\n"
        "  {subjectID: 1, evidence: ['A'], response: true, RT: .9},\n"
        "  {subjectID: 1, evidence: ['A','B','C','D','E','F'], response: true, RT: 4},\n"
        "  {subjectID: 1, evidence: ['A','B','C'], response: true, RT: 2},\n"
        "  {subjectID: 2, evidence: ['A'], response: true, RT: 1.5},\n"
        "  {subjectID: 2, evidence: ['A','B','C','D','E','F'], response: false, RT: 5},\n"
        "  {subjectID: 2, evidence: ['A','B','C'], response: true, RT: 2.2},\n"
        "];\n"
        "```\n\n"
        "Use a per-subject base rate (mem). End with the Infer(...) returning the "
        "joint posterior `{subject1, subject2, numSamples}`."
    ),
    "groundtruth_code": (
        "var timeIt = function(func) {\n"
        "  var start = _.now();\n"
        "  func();\n"
        "  var end = _.now();\n"
        "  return end - start;\n"
        "};\n\n"
        "var detectingBlickets = function(evidence, baseRate, numSamples) {\n"
        "  return Infer({method: 'rejection', samples: numSamples}, function() {\n"
        "    var blicket = mem(function(block) { flip(baseRate) });\n"
        "    var power = function(block) { blicket(block) ? .95 : .05 };\n"
        "    var machineBeeps = function(blocks) {\n"
        "      blocks.length == 0 ? flip(0.05) :\n"
        "        flip(power(first(blocks))) || machineBeeps(rest(blocks))\n"
        "    };\n"
        "    condition(machineBeeps(evidence));\n"
        "    return blicket('A');\n"
        "  })\n"
        "};\n\n"
        "var venusData = [\n"
        "  {subjectID: 1, evidence: ['A'], response: true, RT: .9},\n"
        "  {subjectID: 1, evidence: ['A', 'B', 'C', 'D', 'E', 'F'], response: true, RT: 4},\n"
        "  {subjectID: 1, evidence: ['A', 'B', 'C'], response: true, RT: 2},\n"
        "  {subjectID: 2, evidence: ['A'], response: true, RT: 1.5},\n"
        "  {subjectID: 2, evidence: ['A', 'B', 'C', 'D', 'E', 'F'], response: false, RT: 5},\n"
        "  {subjectID: 2, evidence: ['A', 'B', 'C'], response: true, RT: 2.2},\n"
        "];\n\n"
        "var getModelRT = function(func, numRepeats) {\n"
        "  var rt = repeat(numRepeats, function() { timeIt(func) });\n"
        "  return Gaussian({mu: listMean(rt), sigma: Math.max(listVar(rt), 1)});\n"
        "};\n\n"
        "var dataAnalysis = function() {\n"
        "  var baseRate = mem(function(subjectID) { uniform(0, 1) });\n"
        "  var numSamples = randomInteger(100) + 1;\n\n"
        "  map(function(datapoint) {\n"
        "    var blicketModel = function() { \n"
        "      return detectingBlickets(datapoint.evidence, baseRate(datapoint.subjectID), numSamples)\n"
        "    };\n"
        "    observe(blicketModel(), datapoint.response);\n"
        "    observe(getModelRT(blicketModel, 10), datapoint.RT);\n"
        "  }, venusData);\n\n"
        "  return {subject1: baseRate(1), subject2: baseRate(2), numSamples: numSamples};\n"
        "};\n\n"
        "Infer({method: 'MCMC', samples: 500, burn: 100}, dataAnalysis);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-process-models/ex5",
    "source": "exercises/process-models.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": BLICKET_PROMPT_INTRO + (
        "Compare hypotheses that aliens use rejection sampling versus enumeration. "
        "Make `algorithm` (one of 'rejection' or 'enumerate') a random variable per "
        "planet, parameterize `detectingBlickets` to take it, and combine Mars + Venus "
        "data into one dataset (each datum tagged with planet). Return "
        "`{algVenus, algMars}`. End with the Infer(...)."
    ),
    "groundtruth_code": (
        "var timeIt = function(func) {\n"
        "  var start = _.now();\n"
        "  func();\n"
        "  var end = _.now();\n"
        "  return end - start;\n"
        "};\n\n"
        "var detectingBlickets = function(evidence, baseRate, algorithm, numSamples) {\n"
        "  return Infer({method: algorithm, samples: numSamples}, function() {\n"
        "    var blicket = mem(function(block) { flip(baseRate) });\n"
        "    var power = function(block) { blicket(block) ? .95 : .05 };\n"
        "    var machineBeeps = function(blocks) {\n"
        "      blocks.length == 0 ? flip(0.05) :\n"
        "        flip(power(first(blocks))) || machineBeeps(rest(blocks))\n"
        "    };\n"
        "    condition(machineBeeps(evidence));\n"
        "    return blicket('A');\n"
        "  })\n"
        "};\n\n"
        "var data = [\n"
        "  {planet: 'Mars', subjectID: 1, evidence: ['A'], response: true, RT: .9},\n"
        "  {planet: 'Mars', subjectID: 1, evidence: ['A','B','C','D','E','F'], response: true, RT: 1.1},\n"
        "  {planet: 'Mars', subjectID: 1, evidence: ['A','B','C'], response: true, RT: 1.2},\n"
        "  {planet: 'Mars', subjectID: 2, evidence: ['A'], response: true, RT: 3.5},\n"
        "  {planet: 'Mars', subjectID: 2, evidence: ['A','B','C','D','E','F'], response: false, RT: 4},\n"
        "  {planet: 'Mars', subjectID: 2, evidence: ['A','B','C'], response: true, RT: 3.4},\n"
        "  {planet: 'Venus', subjectID: 3, evidence: ['A'], response: true, RT: .9},\n"
        "  {planet: 'Venus', subjectID: 3, evidence: ['A','B','C','D','E','F'], response: true, RT: 4},\n"
        "  {planet: 'Venus', subjectID: 3, evidence: ['A','B','C'], response: true, RT: 2},\n"
        "  {planet: 'Venus', subjectID: 4, evidence: ['A'], response: true, RT: 1.5},\n"
        "  {planet: 'Venus', subjectID: 4, evidence: ['A','B','C','D','E','F'], response: false, RT: 5},\n"
        "  {planet: 'Venus', subjectID: 4, evidence: ['A','B','C'], response: true, RT: 2.2},\n"
        "];\n\n"
        "var getModelRT = function(func, numRepeats) {\n"
        "  var rt = repeat(numRepeats, function() { timeIt(func) });\n"
        "  return Gaussian({mu: listMean(rt), sigma: Math.max(listVar(rt), 1)});\n"
        "};\n\n"
        "var dataAnalysis = function() {\n"
        "  var baseRate = mem(function(subjectID) { uniform(0, 1) });\n"
        "  var algorithm = mem(function(planet) { flip() ? 'rejection' : 'enumerate' });\n"
        "  var numSamples = randomInteger(100) + 1;\n\n"
        "  map(function(datapoint) {\n"
        "    var blicketModel = function() { \n"
        "      return detectingBlickets(datapoint.evidence, baseRate(datapoint.subjectID),\n"
        "                               algorithm(datapoint.planet), numSamples)\n"
        "    };\n"
        "    observe(blicketModel(), datapoint.response);\n"
        "    observe(getModelRT(blicketModel, 10), datapoint.RT);\n"
        "  }, data);\n\n"
        "  return {algVenus: algorithm('Venus'), algMars: algorithm('Mars')};\n"
        "};\n\n"
        "Infer({method: 'MCMC', samples: 500, burn: 100}, dataAnalysis);\n"
    ),
})

# ─── occams-razor.md ─────────────────────────────────────────────────
NUMGAME_FOLD = (
    "var maxNumber = 20;\n"
    "var filterByInRange =  function(set) {\n"
    "  var inRange = function(v) {v <= maxNumber && v >= 0};\n"
    "  return _.uniq(filter(inRange, set));\n"
    "};\n"
    "var genEvens = function() {\n"
    "  return filter(function(v) {return v % 2 == 0}, _.range(1, maxNumber));\n"
    "};\n"
    "var genOdds = function() {\n"
    "  return filter(function(v) {return (v + 1) % 2 == 0}, _.range(1, maxNumber));\n"
    "};\n"
    "var genMultiples = function(base) {\n"
    "  var multiples = map(function(v) {return base * v}, _.range(maxNumber));\n"
    "  return filterByInRange(multiples);\n"
    "};\n"
    "var genPowers = function(base) {\n"
    "  var powers = map(function(v) {return Math.pow(base, v)}, _.range(maxNumber));\n"
    "  return filterByInRange(powers);\n"
    "};\n"
    "var inSet = function(val, set) { return _.includes(set, val); };\n"
    "var makeRuleHypothesisSpace = function() {\n"
    "  var multipleRules = map(function(base) {return 'multiples_of_' + base}, _.range(1, 12));\n"
    "  var powerRules = map(function(base) {return 'powers_of_' + base}, _.range(1, 12));\n"
    "  return multipleRules.concat(powerRules).concat(['evens', 'odds']);\n"
    "};\n"
)

ATOMS.append({
    "id": "probmods2-occams-razor/ex1.2",
    "source": "exercises/occams-razor.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "We have a number-game model with rule-based hypotheses (powers of N, "
        "multiples of N, evens, odds) over numbers 1..20. Implement similarity-based "
        "interval hypotheses (integers in [a, b]). Implement `genSetFromInterval(a, b)`, "
        "`makeIntervalHypothesisSpace(start, end)` (returning names like "
        "'interval_1_2', 'interval_1_3', ...), and modify `getSetFromHypothesis` to "
        "handle interval rule names. Mix rule and interval hypotheses 50/50 in the "
        "prior. Helpers `genEvens, genOdds, genMultiples, genPowers, "
        "makeRuleHypothesisSpace, inSet, maxNumber` are available.\n\n"
        "End with `Infer(...)` returning posterior over `{hypothesis, testQueryResponse}` "
        "for examples = [3, 10] and testQuery = 12."
    ),
    "groundtruth_code": (
        NUMGAME_FOLD +
        "var genSetFromInterval = function(a, b) { return _.range(a, b+1); };\n\n"
        "var makeIntervalHypothesisSpace = function(start, end) {\n"
        "  var allIntervals = _.flatten(map(function(s) {\n"
        "    return map(function(e) { [s, e] }, genSetFromInterval(s+1, end));\n"
        "  }, genSetFromInterval(start, end)));\n"
        "  return map(function(x) { 'interval_' + x[0] + '_' + x[1] }, allIntervals);\n"
        "};\n\n"
        "var getSetFromHypothesis = function(rule) {\n"
        "  var parts = rule.split('_');\n"
        "  return (parts[0] == 'multiples' ? genMultiples(_.parseInt(parts[2])) :\n"
        "          parts[0] == 'powers' ? genPowers(_.parseInt(parts[2])) :\n"
        "          parts[0] == 'evens' ? genEvens() :\n"
        "          parts[0] == 'odds' ? genOdds() :\n"
        "          parts[0] == 'interval' ? genSetFromInterval(_.parseInt(parts[1]), _.parseInt(parts[2])) :\n"
        "          console.error('unknown rule' + rule));\n"
        "};\n\n"
        "var learnConcept = function(examples, testQuery) {\n"
        "  return Infer({method: 'enumerate'}, function() {\n"
        "    var rules = makeRuleHypothesisSpace();\n"
        "    var intervals = makeIntervalHypothesisSpace(1, maxNumber);\n"
        "    var hypothesis = flip(0.5) ? uniformDraw(rules) : uniformDraw(intervals);\n"
        "    var set = getSetFromHypothesis(hypothesis);\n"
        "    mapData({data: examples}, function(example) {\n"
        "      observe(Categorical({vs: set}), example);\n"
        "    });\n"
        "    return {hypothesis: hypothesis,\n"
        "            testQueryResponse: inSet(testQuery, set)};\n"
        "  });\n"
        "};\n\n"
        "learnConcept([3, 10], 12);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-occams-razor/ex1.3",
    "source": "exercises/occams-razor.md",
    "task_type": "write_from_scratch",
    "eval_mode": "value",
    "answer_shape": "value",
    "prompt": (
        "Using the number-game model with rules + interval hypotheses (50/50 mix in "
        "prior, mapping over examples with `observe(Categorical({vs: set}), example)`), "
        "compute for each integer query in [1, 20] the expected probability that the "
        "query is in the inferred set, given examples = [3, 6, 9]. End with the array "
        "of those expected probabilities (length 20).\n\n"
        "Helpers `genEvens, genOdds, genMultiples, genPowers, makeRuleHypothesisSpace, "
        "genSetFromInterval, makeIntervalHypothesisSpace, getSetFromHypothesis, "
        "learnConcept, inSet, maxNumber` are available."
    ),
    "groundtruth_code": (
        NUMGAME_FOLD +
        "var genSetFromInterval = function(a, b) { return _.range(a, b+1); };\n"
        "var makeIntervalHypothesisSpace = function(start, end) {\n"
        "  var allIntervals = _.flatten(map(function(s) {\n"
        "    return map(function(e) { [s, e] }, genSetFromInterval(s+1, end));\n"
        "  }, genSetFromInterval(start, end)));\n"
        "  return map(function(x) { 'interval_' + x[0] + '_' + x[1] }, allIntervals);\n"
        "};\n"
        "var getSetFromHypothesis = function(rule) {\n"
        "  var parts = rule.split('_');\n"
        "  return (parts[0] == 'multiples' ? genMultiples(_.parseInt(parts[2])) :\n"
        "          parts[0] == 'powers' ? genPowers(_.parseInt(parts[2])) :\n"
        "          parts[0] == 'evens' ? genEvens() :\n"
        "          parts[0] == 'odds' ? genOdds() :\n"
        "          parts[0] == 'interval' ? genSetFromInterval(_.parseInt(parts[1]), _.parseInt(parts[2])) :\n"
        "          console.error('unknown rule' + rule));\n"
        "};\n"
        "var learnConcept = function(examples, testQuery) {\n"
        "  return Infer({method: 'enumerate'}, function() {\n"
        "    var rules = makeRuleHypothesisSpace();\n"
        "    var intervals = makeIntervalHypothesisSpace(1, maxNumber);\n"
        "    var hypothesis = flip(0.5) ? uniformDraw(rules) : uniformDraw(intervals);\n"
        "    var set = getSetFromHypothesis(hypothesis);\n"
        "    mapData({data: examples}, function(example) {\n"
        "      observe(Categorical({vs: set}), example);\n"
        "    });\n"
        "    return {hypothesis: hypothesis,\n"
        "            testQueryResponse: inSet(testQuery, set)};\n"
        "  });\n"
        "};\n\n"
        "var examples = [3, 6, 9];\n"
        "var queries = genSetFromInterval(1, maxNumber);\n"
        "map(function(query) {\n"
        "  var post = learnConcept(examples, query);\n"
        "  return expectation(marginalize(post, function(x) { x.testQueryResponse }));\n"
        "}, queries);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-occams-razor/ex2.1",
    "source": "exercises/occams-razor.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given this Causal Power model where C is a possible cause of E:\n\n"
        "```\n"
        "var observedData = [{C:true, E:false}];\n"
        "Infer({method: 'MCMC', samples: 10000, lag:2}, function() {\n"
        "  var cp = uniform(0, 1);  // Causal power of C to cause E\n"
        "  var b = uniform(0, 1);   // Background probability of E\n"
        "  mapData({data: observedData}, function(datum) {\n"
        "    var E = (datum.C && flip(cp)) || flip(b);\n"
        "    condition(E == datum.E);\n"
        "  })\n"
        "  return {cp, b};\n"
        "})\n"
        "```\n\n"
        "Modify it into a Causal Support model: also infer whether there is a causal "
        "relation at all (`relation = flip()`). When `relation` is false, C does not "
        "affect E. Return `{relation, cp, b}`. End with the Infer(...)."
    ),
    "groundtruth_code": (
        "var observedData = [{C:true, E:false}];\n\n"
        "Infer({method: 'MCMC', samples: 10000, lag:2}, function() {\n"
        "  var relation = flip();\n"
        "  var cp = uniform(0, 1);\n"
        "  var b = uniform(0, 1);\n\n"
        "  mapData({data: observedData}, function(datum) {\n"
        "    var E = (relation && datum.C && flip(cp)) || flip(b);\n"
        "    condition(E == datum.E);\n"
        "  });\n\n"
        "  return {relation, cp, b};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-occams-razor/ex2.2",
    "source": "exercises/occams-razor.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given the Causal Support model (`relation, cp, b` latent; observedData = "
        "[{C:true, E:false}]; noisy-OR effect `E = (relation && C && flip(cp)) || "
        "flip(b)`):\n\n"
        "Single-site MH MCMC won't be efficient because changing `relation` alone "
        "rarely accepts. Improve efficiency by constructing the *marginal probability "
        "of E* directly via a small Infer, and use that in an `observe` (instead of "
        "sampling E and conditioning). Return `{relation, cp, b}`."
    ),
    "groundtruth_code": (
        "var observedData = [{C:true, E:false}];\n\n"
        "Infer({method: 'MCMC', samples: 10000, lag:2}, function() {\n"
        "  var relation = flip();\n"
        "  var cp = uniform(0, 1);\n"
        "  var b = uniform(0, 1);\n\n"
        "  var noisyOrMarginal = function(C) {\n"
        "    return Infer({method: 'enumerate'}, function() {\n"
        "      return (relation && C && flip(cp)) || flip(b);\n"
        "    });\n"
        "  };\n\n"
        "  mapData({data: observedData}, function(datum) {\n"
        "    observe(noisyOrMarginal(datum.C), datum.E);\n"
        "  });\n\n"
        "  return {relation, cp, b};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-occams-razor/ex2.3",
    "source": "exercises/occams-razor.md",
    "task_type": "write_from_scratch",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "cpValues": "value",
        "csValues": "value",
    }},
    "prompt": (
        "Compare Causal Power (CP) and Causal Support (CS) models on data where E "
        "occurs varying amounts with and without C.\n\n"
        "```\n"
        "var generateData = function(numEWithC, numEWithoutC) {\n"
        "  var eWithC = repeat(numEWithC, function() {return {C: true, E: true}});\n"
        "  var noEWithC = repeat(8 - numEWithC, function() {return {C: true, E: false}});\n"
        "  var eWithoutC = repeat(numEWithoutC, function() {return {C: false, E: true}});\n"
        "  var noEWithoutC = repeat(8 - numEWithoutC, function() {return {C: false, E: false}});\n"
        "  return _.flatten([eWithC, noEWithC, eWithoutC, noEWithoutC]);\n"
        "}\n"
        "var dataParams = [[8,8],[6,6],[4,4],[2,2],[0,0],[8,6],[6,4],[4,2],[2,0],[8,4],[6,2],[4,0],[8,2],[6,0],[8,0]];\n"
        "```\n\n"
        "Implement `cpPost(observedData)` (returning posterior over `cp`) and "
        "`csPost(observedData)` (returning posterior over `relation*cp`), both using "
        "the marginal-noisy-OR `observe` from Ex 2.2. Return an object literal "
        "`{cpValues, csValues}` where each is an array of `expectation(...)` of "
        "the corresponding posterior across all 15 dataParams configurations."
    ),
    "groundtruth_code": (
        "var generateData = function(numEWithC, numEWithoutC) {\n"
        "  var eWithC = repeat(numEWithC, function() {return {C: true, E: true}});\n"
        "  var noEWithC = repeat(8 - numEWithC, function() {return {C: true, E: false}});\n"
        "  var eWithoutC = repeat(numEWithoutC, function() {return {C: false, E: true}});\n"
        "  var noEWithoutC = repeat(8 - numEWithoutC, function() {return {C: false, E: false}});\n"
        "  return _.flatten([eWithC, noEWithC, eWithoutC, noEWithoutC]);\n"
        "};\n\n"
        "var dataParams = [[8, 8], [6, 6], [4, 4], [2, 2], [0, 0], [8, 6],\n"
        "                  [6, 4], [4, 2], [2, 0], [8, 4], [6, 2], [4, 0],\n"
        "                  [8, 2], [6, 0], [8, 0]];\n\n"
        "var data = map(function(x) { generateData(x[0], x[1]) }, dataParams);\n\n"
        "var cpPost = function(observedData) {\n"
        "  return Infer({method: 'MCMC', burn: 2000, samples: 1000, lag:2}, function() {\n"
        "    var cp = uniform(0, 1);\n"
        "    var b = uniform(0, 1);\n"
        "    var noisyOrMarginal = function(C) {\n"
        "      return Infer({method: 'enumerate'}, function() {\n"
        "        return (C && flip(cp)) || flip(b);\n"
        "      });\n"
        "    };\n"
        "    mapData({data: observedData}, function(datum) {\n"
        "      observe(noisyOrMarginal(datum.C), datum.E);\n"
        "    });\n"
        "    return cp;\n"
        "  });\n"
        "};\n\n"
        "var csPost = function(observedData) {\n"
        "  return Infer({method: 'MCMC', burn: 2000, samples: 1000, lag:2}, function() {\n"
        "    var relation = flip();\n"
        "    var cp = uniform(0, 1);\n"
        "    var b = uniform(0, 1);\n"
        "    var noisyOrMarginal = function(C) {\n"
        "      return Infer({method: 'enumerate'}, function() {\n"
        "        return (relation && C && flip(cp)) || flip(b);\n"
        "      });\n"
        "    };\n"
        "    mapData({data: observedData}, function(datum) {\n"
        "      observe(noisyOrMarginal(datum.C), datum.E);\n"
        "    });\n"
        "    return relation * cp;\n"
        "  });\n"
        "};\n\n"
        "({\n"
        "  cpValues: map(function(d) { expectation(cpPost(d)) }, data),\n"
        "  csValues: map(function(d) { expectation(csPost(d)) }, data)\n"
        "})\n"
    ),
})

# ─── social-cognition.md ─────────────────────────────────────────────
SALLY_FOLD = (
    "var actionPrior = Categorical({vs: ['a', 'b', 'c'], ps: [1/3, 1/3, 1/3]});\n"
    "var foodPrior = Categorical({vs: ['bagel', 'cookie', 'doughnut'], ps: [1/3, 1/3, 1/3]});\n\n"
    "var vendingMachine = function(state, action) {\n"
    "  return action == 'a' ? categorical({vs: ['bagel', 'cookie', 'doughnut'], ps: [.8, .1, .1]}) :\n"
    "         action == 'b' ? categorical({vs: ['bagel', 'cookie', 'doughnut'], ps: [.1, .8, .1]}) :\n"
    "         action == 'c' ? categorical({vs: ['bagel', 'cookie', 'doughnut'], ps: [.1, .1, .8]}) :\n"
    "         'nothing';\n"
    "};\n\n"
    "var chooseAction = function(goal, transition, state, deceive) {\n"
    "  return Infer({method: 'enumerate'}, function() {\n"
    "    var action = sample(actionPrior);\n"
    "    var outcome = transition(state, action);\n"
    "    condition(deceive ? !goal(outcome) : goal(outcome));\n"
    "    return action;\n"
    "  });\n"
    "};\n\n"
)

ATOMS.append({
    "id": "probmods2-social-cognition/ex1.1",
    "source": "exercises/social-cognition.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Sally chooses an action 'a', 'b', or 'c' that maps to a vending machine "
        "outcome (each action gives 80% chance of one specific food, 10% each for the "
        "others). If Sally is *deceptive* she chooses an action that does NOT lead to "
        "her goal food; otherwise she chooses one that does.\n\n"
        "Fill in the `condition(...)` calls. End with the Infer(...) call returning "
        "the posterior over Sally's goal food given that she is deceptive AND chose 'b'.\n\n"
        "```\n"
        "var actionPrior = Categorical({vs: ['a','b','c'], ps: [1/3,1/3,1/3]});\n"
        "var foodPrior = Categorical({vs: ['bagel','cookie','doughnut'], ps: [1/3,1/3,1/3]});\n"
        "var vendingMachine = /* maps action -> categorical food */;\n\n"
        "var chooseAction = function(goal, transition, state, deceive) {\n"
        "  return Infer({method: 'enumerate'}, function() {\n"
        "    var action = sample(actionPrior);\n"
        "    condition(...)\n"
        "    return action;\n"
        "  })\n"
        "};\n\n"
        "Infer({method: 'enumerate'}, function() {\n"
        "  var deceive = flip();\n"
        "  var goalFood = sample(foodPrior);\n"
        "  var goal = function(outcome) {return outcome == goalFood};\n"
        "  var sallyActionDist = chooseAction(goal, vendingMachine, 'state', deceive);\n"
        "  condition(...)\n"
        "  return goalFood;\n"
        "});\n"
        "```"
    ),
    "groundtruth_code": (
        SALLY_FOLD +
        "Infer({method: 'enumerate'}, function() {\n"
        "  var deceive = flip();\n"
        "  var goalFood = sample(foodPrior);\n"
        "  var goal = function(outcome) {return outcome == goalFood};\n"
        "  var sallyActionDist = chooseAction(goal, vendingMachine, 'state', deceive);\n"
        "  condition(deceive);\n"
        "  condition(sample(sallyActionDist) == 'b');\n"
        "  return goalFood;\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-social-cognition/ex1.2",
    "source": "exercises/social-cognition.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Sally chooses 'a', 'b', or 'c'. If deceptive she avoids her goal food; "
        "otherwise she picks an action whose outcome IS her goal food. The vending "
        "machine: action 'a' -> 80% bagel, 'b' -> 80% cookie, 'c' -> 80% doughnut "
        "(10% each for the others).\n\n"
        "You observe Sally choose 'b' twice in a row. End with the Infer(...) "
        "returning the posterior over her goal food."
    ),
    "groundtruth_code": (
        SALLY_FOLD +
        "Infer({method: 'enumerate'}, function() {\n"
        "  var deceive = flip();\n"
        "  var goalFood = sample(foodPrior);\n"
        "  var goal = function(outcome) {return outcome == goalFood};\n"
        "  var sallyActionDist = chooseAction(goal, vendingMachine, 'state', deceive);\n"
        "  condition(sample(sallyActionDist) == 'b');\n"
        "  condition(sample(sallyActionDist) == 'b');\n"
        "  return goalFood;\n"
        "});\n"
    ),
})

MONTY_FOLD = (
    "var removeBadItems = function(l, badItems) {\n"
    "  return reduce(function(badItem, remainingL) {\n"
    "    return remove(badItem, remainingL)\n"
    "  }, l, badItems);\n"
    "};\n\n"
    "var doors = [1, 2, 3];\n\n"
)

def monty_atom(part_id, prompt_extra, monty_func_name, monty_func_body):
    return {
        "id": f"probmods2-social-cognition/{part_id}",
        "source": "exercises/social-cognition.md",
        "task_type": "fill_in_blank",
        "eval_mode": "record",
        "answer_shape": {"record": {
            "stay": "distribution",
            "switch": "distribution",
        }},
        "prompt": (
            "Monty Hall problem variant. Alice picks one of three doors (with a prize "
            "behind one); Monty opens a different door not containing the prize. "
            f"{prompt_extra} Compute P(Alice wins) under both 'stay' and 'switch'.\n\n"
            "```\n"
            "var doors = [1, 2, 3];\n"
            f"var {monty_func_name} = function(aliceDoor, prizeDoor) {{\n"
            "  return Infer({method: 'enumerate'}, function() {\n"
            "    return ...  // Monty's door distribution\n"
            "  })\n"
            "};\n"
            "```\n\n"
            "Define a `model(switches)` that returns a boolean (Alice wins) given "
            "her strategy. Return an object literal with two distributions:\n"
            "  - `stay`: P(win | Alice doesn't switch)\n"
            "  - `switch`: P(win | Alice switches)"
        ),
        "groundtruth_code": (
            MONTY_FOLD +
            f"var {monty_func_name} = function(aliceDoor, prizeDoor) {{\n"
            "  return Infer({method: 'enumerate'}, function() {\n"
            f"{monty_func_body}"
            "  });\n"
            "};\n\n"
            "var model = function(switches) {\n"
            "  var aliceDoor = categorical({vs: doors});\n"
            "  var prizeDoor = categorical({vs: doors});\n"
            f"  var montyDoorDist = {monty_func_name}(aliceDoor, prizeDoor);\n"
            "  var montyDoor = sample(montyDoorDist);\n"
            "  condition(montyDoor != prizeDoor);\n"
            "  condition(montyDoor != aliceDoor);\n"
            "  var aliceDoor = switches ? removeBadItems(doors, [aliceDoor, montyDoor])[0] : aliceDoor;\n"
            "  return aliceDoor == prizeDoor;\n"
            "};\n\n"
            "({\n"
            "  stay: Infer({method: 'enumerate'}, function() { return model(false); }),\n"
            "  switch: Infer({method: 'enumerate'}, function() { return model(true); })\n"
            "})\n"
        ),
    }

ATOMS.append(monty_atom(
    "ex2.1",
    "In this variant, Monty picks a door *uniformly at random* from all three (we condition that his door is not Alice's and not the prize).",
    "montyRandom",
    "    return categorical({vs: doors});\n",
))

ATOMS.append(monty_atom(
    "ex2.2",
    "In this variant (the original), Monty deliberately picks a door that is neither Alice's nor the prize.",
    "montyAvoidBoth",
    "    var montyDoor = categorical({vs: doors});\n"
    "    condition(montyDoor != aliceDoor);\n"
    "    condition(montyDoor != prizeDoor);\n"
    "    return montyDoor;\n",
))

ATOMS.append(monty_atom(
    "ex2.4",
    "In this variant, Monty picks a door uniformly at random but only avoids Alice's door (he might inadvertently reveal the prize).",
    "montyAvoidAlice",
    "    var montyDoor = categorical({vs: doors});\n"
    "    condition(montyDoor != aliceDoor);\n"
    "    return montyDoor;\n",
))

ATOMS.append(monty_atom(
    "ex2.5",
    "In this variant, Monty picks a door uniformly at random but only avoids the prize door (he might inadvertently pick Alice's door).",
    "montyAvoidPrize",
    "    var montyDoor = categorical({vs: doors});\n"
    "    condition(montyDoor != prizeDoor);\n"
    "    return montyDoor;\n",
))


if __name__ == "__main__":
    write_atoms(ATOMS, append=True)

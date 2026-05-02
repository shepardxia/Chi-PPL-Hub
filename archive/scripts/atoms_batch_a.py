"""Batch A: bayesian-data-analysis (1) + conditional-dependence (2) +
learning-as-conditional-inference (3) + mixture-models (3) = 9 atoms.

All atoms now end with the canonical answer expression."""
from atom_writer import write_atoms

ATOMS = []

# ─── bayesian-data-analysis.md ───────────────────────────────────────
ATOMS.append({
    "id": "probmods2-bayesian-data-analysis/ex1.2",
    "source": "exercises/bayesian-data-analysis.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given this binomial model where predictive distributions are computed for "
        "the same number of attempts as the observed data:\n\n"
        "```\n"
        "var k = 1; var n = 20;\n"
        "var priorDist = Uniform({a: 0, b: 1});\n"
        "var model = function() {\n"
        "  var p = sample(priorDist);\n"
        "  observe(Binomial({p: p, n: n}), k);\n"
        "  var posteriorPredictive = binomial(p, n);\n"
        "  var prior_p = sample(priorDist);\n"
        "  var priorPredictive = binomial(prior_p, n);\n"
        "  return {prior: prior_p, priorPredictive, posterior: p, posteriorPredictive};\n"
        "}\n"
        "Infer({method: \"MCMC\", samples: 2500, lag: 50}, model);\n"
        "```\n\n"
        "Predictive distributions can be used to predict the outcome of a *different* "
        "experiment with `new_n != n`. Modify the model so the posterior predictive "
        "uses `new_n = 5` attempts (while observed data is still k=1 success in n=20 "
        "attempts). Use Beta(a=1, b=1) as the prior. End with the Infer(...) call."
    ),
    "groundtruth_code": (
        "var k = 1;\n"
        "var n = 20;\n"
        "var new_n = 5;\n"
        "var priorDist = Beta({a: 1, b: 1});\n\n"
        "var model = function() {\n"
        "   var p = sample(priorDist);\n"
        "   observe(Binomial({p : p, n: n}), k);\n"
        "   var posteriorPredictive = binomial(p, new_n);\n"
        "   var prior_p = sample(priorDist);\n"
        "   var priorPredictive = binomial(prior_p, n);\n"
        "   return {\n"
        "       prior: prior_p, priorPredictive : priorPredictive,\n"
        "       posterior : p, posteriorPredictive : posteriorPredictive\n"
        "   };\n"
        "};\n\n"
        "Infer({method: \"MCMC\", samples: 2500, lag: 50}, model);\n"
    ),
})

# ─── conditional-dependence.md ───────────────────────────────────────
EPI_PROMPT_BASE = (
    "Imagine that you are an epidemiologist determining people's cause of death. "
    "There are two main diseases: cancer (rare, P = 0.00001, often fatal P(death|cancer)=0.9) "
    "and the common cold (P = 0.2, rarely fatal P(death|cold)=0.00006). Very rarely, "
    "people die of other causes (P = 0.000000001). "
)
EPI_GT_BASE = (
    "var cancer = flip(0.00001);\n"
    "var cold = flip(0.2);\n"
    "var death_by_cancer = cancer && flip(0.9);\n"
    "var death_by_cold = cold && flip(0.00006);\n"
    "var other_death = flip(0.000000001);\n"
    "var death = death_by_cancer || death_by_cold || other_death;\n"
)

ATOMS.append({
    "id": "probmods2-conditional-dependence/ex1.a",
    "source": "exercises/conditional-dependence.md",
    "task_type": "write_from_scratch",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "prior": "distribution",
        "death": "distribution",
        "deathAndCold": "distribution",
        "deathAndNoCold": "distribution",
    }},
    "prompt": EPI_PROMPT_BASE + (
        "Return an object literal with four marginals over `cancer`:\n"
        "  - `prior`: unconditional\n"
        "  - `death`: given `death`\n"
        "  - `deathAndCold`: given `death && cold`\n"
        "  - `deathAndNoCold`: given `death && !cold`"
    ),
    "groundtruth_code": (
        "({\n"
        "  prior: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    return cancer;\n"
        "  }),\n"
        "  death: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    condition(death);\n    return cancer;\n"
        "  }),\n"
        "  deathAndCold: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    condition(death && cold);\n    return cancer;\n"
        "  }),\n"
        "  deathAndNoCold: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    condition(death && !cold);\n    return cancer;\n"
        "  })\n"
        "})\n"
    ),
})

ATOMS.append({
    "id": "probmods2-conditional-dependence/ex1.b",
    "source": "exercises/conditional-dependence.md",
    "task_type": "write_from_scratch",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "prior": "distribution",
        "death": "distribution",
        "deathAndCancer": "distribution",
        "deathAndNoCancer": "distribution",
    }},
    "prompt": EPI_PROMPT_BASE + (
        "Return an object literal with four marginals over `cold`:\n"
        "  - `prior`: unconditional\n"
        "  - `death`: given `death`\n"
        "  - `deathAndCancer`: given `death && cancer`\n"
        "  - `deathAndNoCancer`: given `death && !cancer`"
    ),
    "groundtruth_code": (
        "({\n"
        "  prior: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    return cold;\n"
        "  }),\n"
        "  death: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    condition(death);\n    return cold;\n"
        "  }),\n"
        "  deathAndCancer: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    condition(death && cancer);\n    return cold;\n"
        "  }),\n"
        "  deathAndNoCancer: Infer({method: 'enumerate'}, function() {\n"
        f"{EPI_GT_BASE}    condition(death && !cancer);\n    return cold;\n"
        "  })\n"
        "})\n"
    ),
})

# ─── learning-as-conditional-inference.md ────────────────────────────
ATOMS.append({
    "id": "probmods2-learning-as-conditional-inference/ex1.1",
    "source": "exercises/learning-as-conditional-inference.md",
    "task_type": "modify_given",
    "eval_mode": "value",
    "answer_shape": "value",
    "prompt": (
        "Given this 'fair-vs-uniform' coin model:\n\n"
        "```\n"
        "var weightPosterior = function(observedData){\n"
        "  return Infer({method: 'MCMC', burn:1000, samples: 10000}, function() {\n"
        "    var isFair = flip(0.9);\n"
        "    var realWeight = isFair ? 0.5 : uniform({a:0, b:1});\n"
        "    var coin = Bernoulli({p: realWeight});\n"
        "    var obsFn = function(datum){ observe(coin, datum=='h') };\n"
        "    mapData({data: observedData}, obsFn);\n"
        "    return realWeight;\n"
        "  })\n"
        "}\n"
        "```\n\n"
        "This implies a two-faced coin and any other biased coin are equally likely. "
        "Adjust the model so that within the biased class (probability 0.1 in the prior), "
        "the coin is two-faced with probability 0.7 and otherwise uniform on (0, 1). "
        "End your program with `var fullDataSet = repeat(50, function() { 'h' }); var "
        "observedDataSizes = [0,1,2,4,6,8,10,12,15,20,25,30,40,50]; map(function(N) { "
        "expectation(weightPosterior(fullDataSet.slice(0, N))) }, observedDataSizes)` "
        "(an array of expected coin weights at each data size)."
    ),
    "groundtruth_code": (
        "var weightPosterior = function(observedData) {\n"
        "  return Infer({method: 'MCMC', burn:1000, samples: 10000}, function() {\n"
        "    var isFair = flip(0.9);\n"
        "    var isTwoFaced = flip(0.7);\n"
        "    var realWeight = isFair ? 0.5 : (isTwoFaced ? 1 : uniform({a:0, b:1}));\n"
        "    var coin = Bernoulli({p: realWeight});\n"
        "    var obsFn = function(datum) { observe(coin, datum=='h') };\n"
        "    mapData({data: observedData}, obsFn);\n"
        "    return realWeight;\n"
        "  })\n"
        "};\n\n"
        "var fullDataSet = repeat(50, function() { 'h' });\n"
        "var observedDataSizes = [0,1,2,4,6,8,10,12,15,20,25,30,40,50];\n"
        "map(function(N) { expectation(weightPosterior(fullDataSet.slice(0, N))) }, observedDataSizes);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-learning-as-conditional-inference/ex2.1",
    "source": "exercises/learning-as-conditional-inference.md",
    "task_type": "fill_in_blank",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "prior": "distribution",
        "post": "distribution",
    }},
    "prompt": (
        "Given this Beta(10,10)-prior coin model and a data set alternating heads/tails "
        "50 times each:\n\n"
        "```\n"
        "var pseudoCounts = {a: 10, b: 10};\n"
        "var weightPosterior = function(observedData){\n"
        "  return Infer({method: 'MCMC', burn:1000, samples: 1000}, function() {\n"
        "    var coinWeight = sample(Beta(pseudoCounts));\n"
        "    var coinDist = Bernoulli({p: coinWeight});\n"
        "    var obsFn = function(datum){ observe(coinDist, datum=='h') };\n"
        "    mapData({data: observedData}, obsFn);\n"
        "    return coinWeight;\n"
        "  })\n"
        "}\n"
        "var fullDataSet = repeat(50, function() { ['h', 't'] }).flat();\n"
        "```\n\n"
        "Compute the prior distribution and the posterior after observing the full data "
        "set. Return an object literal with two distributions:\n"
        "  - `prior`: the Beta(10,10) prior\n"
        "  - `post`: the posterior after all observations"
    ),
    "groundtruth_code": (
        "var pseudoCounts = {a: 10, b: 10};\n\n"
        "var weightPosterior = function(observedData){\n"
        "  return Infer({method: 'MCMC', burn:1000, samples: 1000}, function() {\n"
        "    var coinWeight = sample(Beta(pseudoCounts));\n"
        "    var coinDist = Bernoulli({p: coinWeight});\n"
        "    var obsFn = function(datum){ observe(coinDist, datum=='h') };\n"
        "    mapData({data: observedData}, obsFn);\n"
        "    return coinWeight;\n"
        "  })\n"
        "};\n\n"
        "var fullDataSet = repeat(50, function() { ['h', 't'] }).flat();\n\n"
        "({\n"
        "  prior: Beta(pseudoCounts),\n"
        "  post: weightPosterior(fullDataSet)\n"
        "})\n"
    ),
})

ATOMS.append({
    "id": "probmods2-learning-as-conditional-inference/ex2.2",
    "source": "exercises/learning-as-conditional-inference.md",
    "task_type": "modify_given",
    "eval_mode": "value",
    "answer_shape": "value",
    "prompt": (
        "Given this Beta(10,10)-prior coin model:\n\n"
        "```\n"
        "var pseudoCounts = {a: 10, b: 10};\n"
        "var weightPosterior = function(observedData){\n"
        "  return Infer({method: 'MCMC', burn:1000, samples: 1000}, function() {\n"
        "    var coinWeight = sample(Beta(pseudoCounts));\n"
        "    var coinDist = Bernoulli({p: coinWeight});\n"
        "    var obsFn = function(datum){ observe(coinDist, datum=='h') };\n"
        "    mapData({data: observedData}, obsFn);\n"
        "    return coinWeight;\n"
        "  })\n"
        "}\n"
        "```\n\n"
        "Compute the *variance* of the posterior at each of these data sizes: "
        "`[0,2,4,8,16,32,64,128,256,512]`, using the data set "
        "`repeat(256, function(){['h','t']}).flat()` (alternating). Variance is "
        "`expectation(posterior, function(x) { Math.pow(x - mean, 2) })` where "
        "`mean = expectation(posterior)`. End with the array of variances."
    ),
    "groundtruth_code": (
        "var pseudoCounts = {a: 10, b: 10};\n\n"
        "var weightPosterior = function(observedData){\n"
        "  return Infer({method: 'MCMC', burn:1000, samples: 1000}, function() {\n"
        "    var coinWeight = sample(Beta(pseudoCounts));\n"
        "    var coinDist = Bernoulli({p: coinWeight});\n"
        "    var obsFn = function(datum){ observe(coinDist, datum=='h') };\n"
        "    mapData({data: observedData}, obsFn);\n"
        "    return coinWeight;\n"
        "  })\n"
        "};\n\n"
        "var fullDataSet = repeat(256, function(){['h', 't']}).flat();\n"
        "var observedDataSizes = [0,2,4,8,16,32,64,128,256,512];\n"
        "map(function(N) {\n"
        "  var posterior = weightPosterior(fullDataSet.slice(0,N));\n"
        "  var mean = expectation(posterior);\n"
        "  return expectation(posterior, function(x) { Math.pow(x - mean, 2) });\n"
        "}, observedDataSizes);\n"
    ),
})

# ─── mixture-models.md ───────────────────────────────────────────────
ATOMS.append({
    "id": "probmods2-mixture-models/ex1.a",
    "source": "exercises/mixture-models.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Imagine you visit an alien planet and observe 10 aliens with three binary "
        "properties (antennae, green, blarghNoise). Implement a mixture model that "
        "assumes there are two kinds of aliens with different distributions over these "
        "properties, with a priori uncertainty over the distributions and over which "
        "kind each alien belongs to. Use Beta(.5, .5) priors on each per-property "
        "probability.\n\n"
        "Fill in the `// Your code here` parts of the scaffold and end with the "
        "Infer(...) call:\n\n"
        "```\n"
        "var properties = ['antennae', 'green', 'blarghNoise']\n"
        "var data = [\n"
        "  {antennae: false, green: false, blarghNoise: false},\n"
        "  {antennae: true,  green: true,  blarghNoise: true},\n"
        "  // ...8 more aliens...\n"
        "]\n\n"
        "var sampleGroupPrototype = mem(function(groupName) {\n"
        "  // Your code here: return {antennae: p1, green: p2, blarghNoise: p3}\n"
        "})\n\n"
        "Infer({method: 'MCMC', kernel: {HMC: {steps: 10, stepSize: .01}}, samples: 3000},\n"
        "      function(){\n"
        "  mapData({data: data}, function(datum) {\n"
        "    // Your code here: assign group, get prototype, observe properties\n"
        "  })\n"
        "  return {group1: sampleGroupPrototype('group1'),\n"
        "          group2: sampleGroupPrototype('group2')}\n"
        "});\n"
        "```"
    ),
    "groundtruth_code": (
        "var properties = ['antennae', 'green', 'blarghNoise'];\n"
        "var data = [\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : false, green: false, blarghNoise: false}\n"
        "];\n\n"
        "var sampleGroupPrototype = mem(function(groupName) {\n"
        "  var probs = repeat(3, function(){ beta(.5, .5)});\n"
        "  return _.zipObject(properties, probs);\n"
        "});\n\n"
        "Infer({method: 'MCMC', kernel: {HMC: {steps: 10, stepSize: .01}}, samples: 3000},\n"
        "      function(){\n"
        "  mapData({data: data}, function(datum) {\n"
        "    var group = flip() ? 'group1' : 'group2';\n"
        "    var prototype = sampleGroupPrototype(group);\n"
        "    mapData({data: properties}, function(property) {\n"
        "      observe(Bernoulli({p: prototype[property]}), datum[property]);\n"
        "    });\n"
        "  });\n"
        "  return {group1: sampleGroupPrototype('group1'),\n"
        "          group2: sampleGroupPrototype('group2')};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-mixture-models/ex1.b",
    "source": "exercises/mixture-models.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given the alien-mixture model from before (10 aliens with antennae, green, "
        "blarghNoise; two latent groups with Beta(.5,.5) priors on each property), "
        "extend the model to handle a new observation: you hear a 'blargh' sound from "
        "a crater but cannot see the alien. Add a new latent variable `mysteryGroup`, "
        "sample its prototype, observe `blarghNoise=true` for that prototype, and have "
        "the inference return `{group1, group2, mysteryGroup}`. End with the Infer(...)."
    ),
    "groundtruth_code": (
        "var properties = ['antennae', 'green', 'blarghNoise'];\n"
        "var data = [\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : true,  green: true,  blarghNoise: true},\n"
        "  {antennae : false, green: false, blarghNoise: false},\n"
        "  {antennae : false, green: false, blarghNoise: false}\n"
        "];\n"
        "var sampleGroupPrototype = mem(function(groupName) {\n"
        "  var probs = repeat(3, function(){ beta(.5, .5)});\n"
        "  return _.zipObject(properties, probs);\n"
        "});\n\n"
        "Infer({method: 'MCMC', kernel: {HMC: {steps: 10, stepSize: .01}}, samples: 3000},\n"
        "      function(){\n"
        "  mapData({data: data}, function(datum) {\n"
        "    var group = flip() ? 'group1' : 'group2';\n"
        "    var prototype = sampleGroupPrototype(group);\n"
        "    mapData({data: properties}, function(property) {\n"
        "      observe(Bernoulli({p: prototype[property]}), datum[property]);\n"
        "    });\n"
        "  });\n"
        "  var mysteryGroup = flip() ? 'group1' : 'group2';\n"
        "  var mysteryPrototype = sampleGroupPrototype(mysteryGroup);\n"
        "  observe(Bernoulli({p: mysteryPrototype['blarghNoise']}), true);\n"
        "  return {group1: sampleGroupPrototype('group1'),\n"
        "          group2: sampleGroupPrototype('group2'),\n"
        "          mysteryGroup: mysteryGroup};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-mixture-models/ex2.a",
    "source": "exercises/mixture-models.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Suppose 22 participants take a memory test scored 0..45. Some are bona fide; "
        "others are malingerers (deliberately scoring low). Implement a mixture model "
        "inferring which group each participant belongs to.\n\n"
        "Fill in the blanks and end with the Infer(...):\n\n"
        "```\n"
        "var scores = [45, 45, 44, 45, 44, 45, 45, 45, 45, 45, 30, 20, 6, 44, 44, 27, 25, 17, 14, 27, 35, 30]\n"
        "var subjIDs = _.range(scores.length)\n"
        "var data = map(function(datum) {return _.zipObject(['subjID', 'score'], datum)}, _.zip(subjIDs, scores));\n\n"
        "Infer({method: 'MCMC', samples: 10000}, function() {\n"
        "  // Your code here: define group success probs, per-participant group memership\n"
        "  var obsFn = function(datum){\n"
        "    observe(// Your code here: Binomial({p, n: 45}))\n"
        "  }\n"
        "  mapData({data: data}, obsFn)\n"
        "  // Your code here\n"
        "  return // Your code here\n"
        "});\n"
        "```\n\n"
        "Use Binomial({p, n: 45}) for each participant's score, with two latent group "
        "success probabilities (use `uniform(0.5, 1)` for the bona-fide group and "
        "`uniform(0, group_1_p)` for malingerers, ensuring p_malingerer < p_bona-fide). "
        "Return participant memberships plus group rates."
    ),
    "groundtruth_code": (
        "var scores = [45, 45, 44, 45, 44, 45, 45, 45, 45, 45, 30, 20, 6, 44, 44, 27, 25, 17, 14, 27, 35, 30];\n"
        "var subjIDs = _.range(scores.length);\n"
        "var data = map(function(datum) {return _.zipObject(['subjID', 'score'], datum)}, _.zip(subjIDs, scores));\n\n"
        "Infer({method: 'MCMC', samples: 10000}, function() {\n"
        "  var group_1_p = uniform(0.5, 1);\n"
        "  var group_2_p = uniform(0, group_1_p);\n"
        "  var participant2Group = mem(function(participantID) {\n"
        "    return flip() ? 'group1' : 'group2';\n"
        "  });\n"
        "  var group2Prob = mem(function(group) {\n"
        "    return group == 'group1' ? group_1_p : group_2_p;\n"
        "  });\n\n"
        "  var obsFn = function(datum){\n"
        "    var p = group2Prob(participant2Group(datum.subjID));\n"
        "    observe(Binomial({p: p, n: 45}), datum.score);\n"
        "  };\n"
        "  mapData({data: data}, obsFn);\n\n"
        "  var participantResults_ = map(function(datum) {return participant2Group(datum.subjID)}, data);\n"
        "  var participantResults = _.zipObject(_.range(participantResults_.length), participantResults_);\n"
        "  return _.merge(participantResults, {group_1_p: group_1_p, group_2_p: group_2_p});\n"
        "});\n"
    ),
})


if __name__ == "__main__":
    write_atoms(ATOMS, append=True)

"""Batch D2: inference-algorithms (9 atoms)."""
from atom_writer import write_atoms

ATOMS = []

HEART_FOLD = (
    "var onCurve = function(x, y) {\n"
    "  var x2 = x*x;\n"
    "  var term1 = y - Math.pow(x2, 1/3);\n"
    "  var crossSection = x2 + term1*term1 - 1;\n"
    "  return Math.abs(crossSection) < 0.01;\n"
    "};\n"
    "var xbounds = [-1, 1];\n"
    "var ybounds = [-1, 1.6];\n\n"
    "var xmu = 0.5 * (xbounds[0] + xbounds[1]);\n"
    "var ymu = 0.5 * (ybounds[0] + ybounds[1]);\n"
    "var xsigma = 0.5 * (xbounds[1] - xbounds[0]);\n"
    "var ysigma = 0.5 * (ybounds[1] - ybounds[0]);\n\n"
)
HEART_INTRO = (
    "Heart-shaped implicit curve: a point (x, y) is *on the curve* if "
    "`x^2 + (y - x^(2/3))^2 - 1 < 0.01` in absolute value. The model proposes "
    "(x, y) from independent Gaussians around the bounding box center, then "
    "conditions on `onCurve(x, y)`. The original model uses rejection sampling. "
    "Helpers `onCurve, xmu, ymu, xsigma, ysigma` are defined.\n\n"
)

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex1.1",
    "source": "exercises/inference-algorithms.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": HEART_INTRO + (
        "Try using MCMC with Metropolis-Hastings instead of rejection sampling on the "
        "same model (`var x = gaussian(xmu, xsigma); var y = gaussian(ymu, ysigma); "
        "condition(onCurve(x, y))`). Use 10000 samples with lag 10. End with the Infer(...)."
    ),
    "groundtruth_code": (
        HEART_FOLD +
        "var model = function() {\n"
        "  var x = gaussian(xmu, xsigma);\n"
        "  var y = gaussian(ymu, ysigma);\n"
        "  condition(onCurve(x, y));\n"
        "  return {x: x, y: y};\n"
        "};\n\n"
        "Infer({method: 'MCMC',\n"
        "       samples: 10000,\n"
        "       lag: 10}, model);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex1.2",
    "source": "exercises/inference-algorithms.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": HEART_INTRO + (
        "Change the *model* (not the algorithm) so MH MCMC successfully traces the curve. "
        "Use `diagCovGaussian({mu: Vector([xmu, ymu]), sigma: Vector([xsigma, ysigma])})` "
        "to *jointly* propose x and y as a vector centered at (xmu, ymu) with diagonal "
        "covariance (xsigma, ysigma). Use MCMC with 1000 samples, lag 100. End with the "
        "Infer(...)."
    ),
    "groundtruth_code": (
        HEART_FOLD +
        "var model = function() {\n"
        "  var xy = diagCovGaussian({mu: Vector([xmu, ymu]),\n"
        "                            sigma: Vector([xsigma, ysigma])});\n"
        "  var x = T.get(xy, 0);\n"
        "  var y = T.get(xy, 1);\n"
        "  condition(onCurve(x, y));\n"
        "  return {x: x, y: y};\n"
        "};\n\n"
        "Infer({method: 'MCMC',\n"
        "       samples: 1000,\n"
        "       lag: 100}, model);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex1.3",
    "source": "exercises/inference-algorithms.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": HEART_INTRO + (
        "Using the original model (independent x, y Gaussians), change the *algorithm* "
        "to HMC. Use HMC kernel with `steps: 10, stepSize: 0.5`, 10000 samples. End "
        "with the Infer(...)."
    ),
    "groundtruth_code": (
        HEART_FOLD +
        "var model = function() {\n"
        "  var x = gaussian(xmu, xsigma);\n"
        "  var y = gaussian(ymu, ysigma);\n"
        "  condition(onCurve(x, y));\n"
        "  return {x: x, y: y};\n"
        "};\n\n"
        "Infer({method: 'MCMC',\n"
        "       samples: 10000,\n"
        "       kernel: {HMC : { steps: 10, stepSize: .5 }} }, model);\n"
    ),
})

INTERPOLATE_FOLD = (
    "var interpolate = function(point1, point2, interpolationWeight) {\n"
    "  return (point1 * interpolationWeight +\n"
    "          point2 * (1 - interpolationWeight));\n"
    "};\n\n"
    "var model = function(){\n"
    "  var point1 = -10;\n"
    "  var point2 = uniform(-100, 100);\n"
    "  var interpolationWeight = uniform(0, 1);\n"
    "  var pointInMiddle = interpolate(point1, point2, interpolationWeight);\n"
    "  observe(Gaussian({mu: 0, sigma:0.1}), pointInMiddle);\n"
    "  return {point2, interpolationWeight, pointInMiddle};\n"
    "};\n\n"
)
INTERPOLATE_INTRO = (
    "Two-endpoint interpolation: `point1 = -10` is fixed; `point2` is uniform on "
    "[-100, 100]; `interpolationWeight` is uniform on [0, 1]; `pointInMiddle = "
    "interpolate(point1, point2, interpolationWeight)`. We condition on observing "
    "pointInMiddle close to 0 (Gaussian sigma=0.1).\n\n"
)

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex2.1",
    "source": "exercises/inference-algorithms.md",
    "task_type": "write_from_scratch",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "point2": "distribution",
        "interpolationWeight": "distribution",
    }},
    "prompt": INTERPOLATE_INTRO + (
        "Run MCMC inference (5000 samples, lag 100) over the model. Return an object "
        "literal `{point2, interpolationWeight}` where each value is the marginal "
        "distribution of the corresponding latent (use `marginalize`)."
    ),
    "groundtruth_code": (
        INTERPOLATE_FOLD +
        "var posterior = Infer({method: 'MCMC', samples: 5000, lag: 100}, model);\n\n"
        "({\n"
        "  point2: marginalize(posterior, function(x) {return x.point2}),\n"
        "  interpolationWeight: marginalize(posterior, function(x) {return x.interpolationWeight})\n"
        "})\n"
    ),
})

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex2.2",
    "source": "exercises/inference-algorithms.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": INTERPOLATE_INTRO + (
        "Run MCMC inference (5000 samples, lag 100). End with the marginal joint "
        "distribution over `(point2, interpolationWeight)` using `marginalize`."
    ),
    "groundtruth_code": (
        INTERPOLATE_FOLD +
        "var posterior = Infer({method: 'MCMC', samples: 5000, lag: 100}, model);\n"
        "marginalize(posterior, function(x) {\n"
        "  return {'point2': x.point2, 'inter': x.interpolationWeight};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex2.3",
    "source": "exercises/inference-algorithms.md",
    "task_type": "write_from_scratch",
    "eval_mode": "value",
    "answer_shape": "value",
    "prompt": INTERPOLATE_INTRO + (
        "Set MCMC parameters to `samples: 100, lag: 0`. Extract the array of "
        "`pointInMiddle` values (in order) from `posterior.samples`. End with that "
        "array of length 100."
    ),
    "groundtruth_code": (
        INTERPOLATE_FOLD +
        "var posterior = Infer({method: 'MCMC', samples: 100, lag: 0}, model);\n"
        "map(function(d) { d[\"value\"][\"pointInMiddle\"] }, posterior.samples);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex2.4",
    "source": "exercises/inference-algorithms.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": INTERPOLATE_INTRO + (
        "Rewrite the model to use rejection sampling (1000 samples). Convert the "
        "`observe` statement into a `condition(Math.abs(pointInMiddle) < 0.01)` "
        "constraint. End with the Infer(...) over `{point2, interpolationWeight, "
        "pointInMiddle}`."
    ),
    "groundtruth_code": (
        "var interpolate = function(point1, point2, interpolationWeight) {\n"
        "  return (point1 * interpolationWeight +\n"
        "          point2 * (1 - interpolationWeight));\n"
        "};\n\n"
        "var model = function(){\n"
        "  var point1 = -10;\n"
        "  var point2 = uniform(-100, 100);\n"
        "  var interpolationWeight = uniform(0, 1);\n"
        "  var pointInMiddle = interpolate(point1, point2, interpolationWeight);\n"
        "  condition(Math.abs(pointInMiddle) < 0.01);\n"
        "  return {point2, interpolationWeight, pointInMiddle};\n"
        "};\n\n"
        "Infer({method: 'rejection', samples: 1000}, model);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex2.5",
    "source": "exercises/inference-algorithms.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": INTERPOLATE_INTRO + (
        "Replace `uniform(-100, 100)` for `point2` with `uniformDrift({a: -100, b: 100, "
        "width: 0.1})` to use a drift kernel that proposes values near the previous "
        "sample. Use 500 samples. End with the Infer(...)."
    ),
    "groundtruth_code": (
        "var interpolate = function(point1, point2, interpolationWeight) {\n"
        "  return (point1 * interpolationWeight +\n"
        "          point2 * (1 - interpolationWeight));\n"
        "};\n\n"
        "var model = function(){\n"
        "  var point1 = -10;\n"
        "  var point2 = uniformDrift({a: -100, b: 100, width: .1});\n"
        "  var interpolationWeight = uniform(0, 1);\n"
        "  var pointInMiddle = interpolate(point1, point2, interpolationWeight);\n"
        "  observe(Gaussian({mu: 0, sigma:0.1}), pointInMiddle);\n"
        "  return {point2, interpolationWeight, pointInMiddle};\n"
        "};\n\n"
        "Infer({method: 'MCMC', samples: 500}, model);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-inference-algorithms/ex4.a",
    "source": "exercises/inference-algorithms.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Implement a topic model with two latent topics over the vocabulary "
        "['DNA', 'evolution', 'parsing', 'phonology']. For each word in each document, "
        "a topic is drawn from a per-document topic distribution, then a word is observed "
        "under that topic's distribution. Topic distributions over the vocabulary are "
        "drawn from Dirichlet(eta = ones); per-document topic distributions are drawn "
        "from Dirichlet(alpha = ones).\n\n"
        "```\n"
        "var vocabulary = ['DNA', 'evolution', 'parsing', 'phonology'];\n"
        "var eta = ones([vocabulary.length, 1])\n"
        "var numTopics = 2\n"
        "var alpha = ones([numTopics, 1])\n\n"
        "var corpus = /* 6 short documents */;\n\n"
        "Infer({method: 'MCMC', samples: 200, lag: 50}, function() {\n"
        "  var topics = repeat(numTopics, function() {\n"
        "    return T.toScalars(dirichlet({alpha: eta}))\n"
        "  })\n"
        "  mapData({data: corpus}, function(doc) {\n"
        "    // your code: per-doc topic dist, then per-word topic + observe\n"
        "  })\n"
        "  return topics\n"
        "});\n"
        "```\n\n"
        "End with the Infer(...) returning the joint posterior over topic distributions."
    ),
    "groundtruth_code": (
        "var vocabulary = ['DNA', 'evolution', 'parsing', 'phonology'];\n"
        "var eta = ones([vocabulary.length, 1]);\n\n"
        "var numTopics = 2;\n"
        "var alpha = ones([numTopics, 1]);\n\n"
        "var corpus = [\n"
        "  'DNA evolution DNA evolution DNA evolution DNA evolution DNA evolution'.split(' '),\n"
        "  'DNA evolution DNA evolution DNA evolution DNA evolution DNA evolution'.split(' '),\n"
        "  'DNA evolution DNA evolution DNA evolution DNA evolution DNA evolution'.split(' '),\n"
        "  'parsing phonology parsing phonology parsing phonology parsing phonology parsing phonology'.split(' '),\n"
        "  'parsing phonology parsing phonology parsing phonology parsing phonology parsing phonology'.split(' '),\n"
        "  'parsing phonology parsing phonology parsing phonology parsing phonology parsing phonology'.split(' ')\n"
        "];\n\n"
        "Infer({method: 'MCMC', samples: 200, lag: 50}, function() {\n"
        "  var topics = repeat(numTopics, function() {\n"
        "    return T.toScalars(dirichlet({alpha: eta}));\n"
        "  });\n\n"
        "  mapData({data: corpus}, function(doc) {\n"
        "    var docTopicDist = dirichlet({alpha: alpha});\n"
        "    mapData({data: doc}, function(word) {\n"
        "      var z = discrete(docTopicDist);\n"
        "      var topic = topics[z];\n"
        "      observe(Categorical({vs: vocabulary, ps: topic}), word);\n"
        "    });\n"
        "  });\n"
        "  return topics;\n"
        "});\n"
    ),
})


if __name__ == "__main__":
    write_atoms(ATOMS, append=True)

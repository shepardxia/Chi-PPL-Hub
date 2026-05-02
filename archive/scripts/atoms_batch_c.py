"""Batch C: hierarchical-models (7) + 05-observing-sequences (8) = 15 atoms."""
from atom_writer import write_atoms

ATOMS = []

# ─── hierarchical-models.md ──────────────────────────────────────────
ATOMS.append({
    "id": "probmods2-hierarchical-models/ex1",
    "source": "exercises/hierarchical-models.md",
    "task_type": "fill_in_blank",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "observed": "distribution",
        "usealpha": "distribution",
    }},
    "prompt": (
        "Show that setting Dirichlet alpha = [2, 3, 1, 1, 1] is equivalent to setting "
        "alpha = [1, 1, 1, 1, 1] then observing the first category once and the second "
        "twice. Build the second model (with alpha = [2, 3, 1, 1, 1]) given the first.\n\n"
        "```\n"
        "var colors = ['black', 'blue', 'green', 'orange', 'red'];\n"
        "var observedData = [{bag: 'bag1', draw: 'blue'},\n"
        "                    {bag: 'bag1', draw: 'blue'},\n"
        "                    {bag: 'bag1', draw: 'black'}];\n\n"
        "var observed = Infer({method: 'MCMC', samples: 20000}, function() {\n"
        "  var makeBag = mem(function(bag) {\n"
        "    var colorProbs = dirichlet(ones([colors.length, 1]));\n"
        "    return Categorical({vs: colors, ps: colorProbs});\n"
        "  })\n"
        "  var obsFn = function(datum) { observe(makeBag(datum.bag), datum.draw); }\n"
        "  mapData({data: observedData}, obsFn);\n"
        "  return {bag1: sample(makeBag('bag1'))};\n"
        "})\n\n"
        "var usealpha = Infer({method: 'MCMC', samples: 20000}, function () {\n"
        "  // ...fill in: alpha = [2, 3, 1, 1, 1] without observation\n"
        "})\n"
        "```\n\n"
        "Return an object literal `{observed, usealpha}` with both posteriors."
    ),
    "groundtruth_code": (
        "var colors = ['black', 'blue', 'green', 'orange', 'red'];\n"
        "var observedData = [{bag: 'bag1', draw: 'blue'},\n"
        "                    {bag: 'bag1', draw: 'blue'},\n"
        "                    {bag: 'bag1', draw: 'black'}];\n\n"
        "var observed = Infer({method: 'MCMC', samples: 20000}, function() {\n"
        "  var makeBag = mem(function(bag) {\n"
        "    var colorProbs = dirichlet(ones([colors.length, 1]));\n"
        "    return Categorical({vs: colors, ps: colorProbs});\n"
        "  });\n"
        "  var obsFn = function(datum) { observe(makeBag(datum.bag), datum.draw); };\n"
        "  mapData({data: observedData}, obsFn);\n"
        "  return {bag1: sample(makeBag('bag1'))};\n"
        "});\n\n"
        "var usealpha = Infer({method: 'MCMC', samples: 20000}, function () {\n"
        "  var makeBag = mem(function(bag) {\n"
        "    var colorProbs = dirichlet(Vector([2, 3, 1, 1, 1]));\n"
        "    return Categorical({vs: colors, ps: colorProbs});\n"
        "  });\n"
        "  return {bag1: sample(makeBag('bag1'))};\n"
        "});\n\n"
        "({observed: observed, usealpha: usealpha})\n"
    ),
})

ATOMS.append({
    "id": "probmods2-hierarchical-models/ex2.1",
    "source": "exercises/hierarchical-models.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Apples in a barrel: each apple is rotten with probability p, where p ~ "
        "Beta(.1, .2) (mass at endpoints). Implement `makeBarrel(barrelName)` "
        "returning a function `barrel(n)` that returns an array of n booleans "
        "(whether each apple is rotten). End with the Infer(...) over `Math.sum(barrel(10))`."
    ),
    "groundtruth_code": (
        "var makeBarrel = mem(function(barrelName) {\n"
        "  var pRotten = beta({a: .1, b: .2});\n"
        "  var barrel = function(n) {\n"
        "    return repeat(n, function() { flip(pRotten) });\n"
        "  };\n"
        "  return barrel;\n"
        "});\n\n"
        "Infer({method: 'forward'}, function() {\n"
        "  var barrel = makeBarrel('barrel');\n"
        "  return Math.sum(barrel(10));\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-hierarchical-models/ex2.2",
    "source": "exercises/hierarchical-models.md",
    "task_type": "fill_in_blank",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "sameStore": "distribution",
        "differentStore": "distribution",
    }},
    "prompt": (
        "Extend the barrel model: each store has its own Beta hyperparameters drawn "
        "from a prior. Use a simple two-class store prior: `flip() ? {a:.1, b:.3} : "
        "{a:.3, b:.1}`. Then visualize differences in rotten counts between barrels:\n"
        "  - `sameStore`: |B1 - B2| where B1, B2 from same store\n"
        "  - `differentStore`: |B1 - B2| where B1 from store 1, B2 from store 2\n\n"
        "Both barrels of size 10. Use forward sampling with 10000 samples. Return "
        "`{sameStore, differentStore}` as the final expression."
    ),
    "groundtruth_code": (
        "var makeStore = mem(function(storeName) {\n"
        "  var storePrior = flip() ? {a: .1, b: .3} : {a: .3, b: .1};\n"
        "  var makeBarrel = mem(function(barrelName) {\n"
        "    var pRotten = beta(storePrior);\n"
        "    var barrel = function(n) {\n"
        "      return repeat(n, function() { flip(pRotten) });\n"
        "    };\n"
        "    return barrel;\n"
        "  });\n"
        "  return makeBarrel;\n"
        "});\n\n"
        "({\n"
        "  sameStore: Infer({method: 'forward', samples: 10000}, function() {\n"
        "    var S = makeStore('S');\n"
        "    var B1 = S('B1');\n"
        "    var B2 = S('B2');\n"
        "    return Math.abs(Math.sum(B1(10)) - Math.sum(B2(10)));\n"
        "  }),\n"
        "  differentStore: Infer({method: 'forward', samples: 10000}, function() {\n"
        "    var S1 = makeStore('S1');\n"
        "    var S2 = makeStore('S2');\n"
        "    var B1 = S1('B1');\n"
        "    var B2 = S2('B2');\n"
        "    return Math.abs(Math.sum(B1(10)) - Math.sum(B2(10)));\n"
        "  })\n"
        "})\n"
    ),
})

ATOMS.append({
    "id": "probmods2-hierarchical-models/ex2.3",
    "source": "exercises/hierarchical-models.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Add another level: cities. Each city has a `cityPrior = beta({a:.25, b:.25})` "
        "that determines whether stores tend to be good ({a:.1, b:.3}) or bad ({a:.3, "
        "b:.1}). Use `flip(cityPrior)` per store to pick. Then for city C1, store S1, "
        "barrel B1, end with the Infer(...) over `Math.sum(B1(20))` (forward sampling)."
    ),
    "groundtruth_code": (
        "var makeCity = mem(function(cityName){\n"
        "  var cityPrior = beta({a: .25, b: .25});\n"
        "  var makeStore = mem(function(storeName) {\n"
        "    var storePrior = flip(cityPrior) ? {a: .1, b: .3} : {a: .3, b: .1};\n"
        "    var makeBarrel = mem(function(barrelName) {\n"
        "      var pRotten = beta(storePrior);\n"
        "      var barrel = function(n) {\n"
        "        return repeat(n, function() { flip(pRotten) });\n"
        "      };\n"
        "      return barrel;\n"
        "    });\n"
        "    return makeBarrel;\n"
        "  });\n"
        "  return makeStore;\n"
        "});\n\n"
        "var C1 = makeCity(\"C1\");\n"
        "var S1 = C1(\"S1\");\n"
        "var B1 = S1(\"B1\");\n\n"
        "Infer({method: 'forward'}, function(){\n"
        "    return Math.sum(B1(20));\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-hierarchical-models/ex2.4",
    "source": "exercises/hierarchical-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Using the hierarchical city/store/barrel model (city: Beta(.25,.25) prior on "
        "store quality; store: prior is {a:.1,b:.3} or {a:.3,b:.1} based on flip(cityPrior); "
        "barrel: pRotten = beta(storePrior)), answer this question via MCMC: you visit "
        "a store in a city and observe a barrel of 10 apples, 7 of which are rotten. "
        "You then visit another store in the same city. End with the Infer(...) "
        "returning the posterior over the number of rotten apples in the second store's "
        "barrel of 10. Use MCMC with 5000 samples, lag 100."
    ),
    "groundtruth_code": (
        "var makeCity = mem(function(cityName){\n"
        "    var cityPrior = beta({a: .25, b: .25});\n\n"
        "    var makeStore = mem(function(storeName) {\n"
        "        var storePrior = flip(cityPrior) ? {a: .1, b: .3} : {a: .3, b: .1};\n\n"
        "        var makeBarrel = mem(function(barrelName) {\n"
        "            var pRotten = beta(storePrior);\n"
        "            var barrel = function(n) {\n"
        "                return repeat(n, function() { flip(pRotten) });\n"
        "            };\n"
        "            return barrel;\n"
        "        });\n\n"
        "        return makeBarrel;\n"
        "    });\n\n"
        "    return makeStore;\n"
        "});\n\n"
        "Infer({method: 'MCMC', samples:5000, lag: 100}, function(){\n"
        "    var C = makeCity(\"C\");\n"
        "    var S1 = C(\"S1\");\n"
        "    var B1 = S1(\"B1\");\n"
        "    var S2 = C(\"S2\");\n"
        "    var B2 = S2(\"B2\");\n\n"
        "    condition(Math.sum(B1(10)) == 7);\n\n"
        "    return Math.sum(B2(10));\n"
        "});\n"
    ),
})

# Word reading time data
WORD_DATA = (
    "var data = [{group: \"vowel\", word: \"abacus\", id: 1, rt: 210},\n"
    "            {group: \"vowel\", word: \"abacus\", id: 2, rt: 212},\n"
    "            {group: \"vowel\", word: \"abacus\", id: 3, rt: 209},\n"
    "            {group: \"vowel\", word: \"aardvark\", id: 1, rt: 200},\n"
    "            {group: \"vowel\", word: \"aardvark\", id: 2, rt: 201},\n"
    "            {group: \"vowel\", word: \"aardvark\", id: 3, rt: 198},\n"
    "            {group: \"vowel\", word: \"ellipse\", id: 1, rt: 220},\n"
    "            {group: \"vowel\", word: \"ellipse\", id: 2, rt: 222},\n"
    "            {group: \"vowel\", word: \"ellipse\", id: 3, rt: 219},\n"
    "            {group: \"consonant\", word: \"proton\", id: 1, rt: 190},\n"
    "            {group: \"consonant\", word: \"proton\", id: 2, rt: 191},\n"
    "            {group: \"consonant\", word: \"proton\", id: 3, rt: 189},\n"
    "            {group: \"consonant\", word: \"folder\", id: 1, rt: 180},\n"
    "            {group: \"consonant\", word: \"folder\", id: 2, rt: 182},\n"
    "            {group: \"consonant\", word: \"folder\", id: 3, rt: 178},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 1, rt: 230},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 2, rt: 231},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 3, rt: 228},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 1, rt: 231},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 2, rt: 233},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 3, rt: 230},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 1, rt: 230},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 2, rt: 232},\n"
    "            {group: \"consonant\", word: \"fedora\", id: 3, rt: 228}];\n\n"
    "var opts = {method: \"MCMC\", burn: 10000, lag: 5, samples: 5000};\n\n"
)

ATOMS.append({
    "id": "probmods2-hierarchical-models/ex3.1",
    "source": "exercises/hierarchical-models.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given a simple BDA on word reading times (data points have group, word, id, rt; "
        "groups are 'vowel' vs 'consonant'):\n\n"
        "```\n"
        "Infer(opts, function() {\n"
        "  var groupMeans = {vowel: gaussian(200, 100),\n"
        "                    consonant: gaussian(200, 100)};\n"
        "  var obsFn = function(d) {\n"
        "    observe(Gaussian({mu: groupMeans[d.group], sigma: 10}), d.rt);\n"
        "  }\n"
        "  mapData({data: data}, obsFn);\n"
        "  return groupMeans['vowel'] - groupMeans['consonant'];\n"
        "})\n"
        "```\n\n"
        "Adjust the model so each *word* has its own mean reading time drawn from its "
        "group's mean (use `mem` keyed on word, sigma=20 for the word-level Gaussian). "
        "End with the Infer(...) returning the posterior over `vowel - consonant` "
        "group mean diff."
    ),
    "groundtruth_code": (
        WORD_DATA +
        "Infer(opts, function() {\n"
        "  var groupMeans = {vowel: gaussian(200, 100),\n"
        "                    consonant: gaussian(200, 100)};\n\n"
        "  var wordMean = mem(function(word, group) {\n"
        "    return gaussian(groupMeans[group], 20);\n"
        "  });\n\n"
        "  var obsFn = function(d) {\n"
        "    observe(Gaussian({mu: wordMean(d.word, d.group),\n"
        "                      sigma: 10}), d.rt);\n"
        "  };\n\n"
        "  mapData({data: data}, obsFn);\n\n"
        "  return groupMeans['vowel'] - groupMeans['consonant'];\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-hierarchical-models/ex3.2",
    "source": "exercises/hierarchical-models.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Extend the hierarchical word-reading-time model (with per-word random effects "
        "via `mem(function(word, group){ gaussian(groupMeans[group], 20) })`) to also "
        "include a per-participant random effect: an unknown additive influence on "
        "reading time per participant id. Use a Gaussian(0, 2) prior on the participant "
        "offset. End with the Infer(...) returning the joint posterior "
        "`{diff, p1, p2, p3}` where diff = vowel - consonant group mean and p1..p3 are "
        "participant offsets."
    ),
    "groundtruth_code": (
        WORD_DATA +
        "Infer(opts, function() {\n"
        "  var groupMeans = {vowel: gaussian(200, 100),\n"
        "                    consonant: gaussian(200, 100)};\n\n"
        "  var participantMean = mem(function(pid) {\n"
        "    return gaussian(0, 2);\n"
        "  });\n\n"
        "  var wordMean = mem(function(word, group) {\n"
        "    return gaussian(groupMeans[group], 20);\n"
        "  });\n\n"
        "  var obsFn = function(d) {\n"
        "    observe(Gaussian({mu: wordMean(d.word, d.group) + participantMean(d.id),\n"
        "                      sigma: 10}), d.rt);\n"
        "  };\n\n"
        "  mapData({data: data}, obsFn);\n\n"
        "  return {diff: groupMeans['vowel'] - groupMeans['consonant'],\n"
        "          p1: participantMean(1),\n"
        "          p2: participantMean(2),\n"
        "          p3: participantMean(3)};\n"
        "});\n"
    ),
})

# ─── 05-observing-sequences.md ───────────────────────────────────────
SEQ_HELPER = (
    "var comparray = function(arr1,arr2){\n"
    "  return (JSON.stringify(arr1) === JSON.stringify(arr2));\n"
    "};\n\n"
)

ATOMS.append({
    "id": "probmods2-observing-sequences/ex1.a",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "We have a vocabulary {dogs, cats, chase, sleep, stop} and a bigram model with "
        "Dirichlet priors over per-word transition distributions.\n\n"
        "Given the scaffold:\n"
        "```\n"
        "Infer({method:'MCMC', burn:10000, samples: 50000, onlyMAP:false}, function() {\n"
        "  let vocab = ['dogs', 'cats', 'chase', 'sleep', 'stop'];\n"
        "  var wordToDistribution = mem(function(word) {\n"
        "    return dirichletDrift({alpha:ones([vocab.length,1]), concentration:10})\n"
        "  })\n"
        "  var transition = function(word) {\n"
        "    return categorical({ps: wordToDistribution(word), vs: vocab})\n"
        "  }\n"
        "  // ...your code here...\n"
        "})\n"
        "```\n"
        "Someone says 'dogs chase cats'. Determine how likely 'chase' is to be followed "
        "by each word. Use a recursive `generateSentence` that emits until 'stop'. "
        "Condition on the observation. End with the Infer(...) returning the "
        "posterior over `transition('chase')`."
    ),
    "groundtruth_code": (
        SEQ_HELPER +
        "Infer({method:'MCMC', burn:10000, samples: 50000, onlyMAP:false}, function() {\n"
        "  let vocab = ['dogs', 'cats', 'chase', 'sleep', 'stop'];\n"
        "  var wordToDistribution = mem(function(word) {\n"
        "    return dirichletDrift({alpha:ones([vocab.length,1]), concentration:10});\n"
        "  });\n"
        "  var transition = function(word) {\n"
        "    return categorical({ps: wordToDistribution(word), vs: vocab});\n"
        "  };\n"
        "  let obs = ['dogs', 'chase', 'cats'];\n"
        "  let generateSentence = function(lastState, sentence) {\n"
        "    let word = transition(lastState);\n"
        "    if (word == 'stop') return [];\n"
        "    return [word].concat(generateSentence(word, sentence));\n"
        "  };\n"
        "  condition(comparray(obs, generateSentence('start')));\n"
        "  return transition('chase');\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-observing-sequences/ex1.b",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Using the same bigram model with Dirichlet-drift priors over transitions and "
        "vocabulary {dogs, cats, chase, sleep, stop}: the speaker said 'dogs chase cats'. "
        "Now they say a *second* sentence whose first word is 'dogs'. End with the "
        "Infer(...) returning the distribution over the second word. Append 'stop' to "
        "the observed sentence so 'undefined' doesn't appear."
    ),
    "groundtruth_code": (
        SEQ_HELPER +
        "Infer({method:'MCMC', burn:10000, samples: 50000, onlyMAP: false}, function() {\n"
        "  let vocab = ['dogs', 'cats', 'chase', 'sleep', 'stop'];\n"
        "  var wordToDistribution = mem(function(word) {\n"
        "    return dirichletDrift({alpha:ones([vocab.length,1]), concentration:10});\n"
        "  });\n"
        "  var transition = function(word) {\n"
        "    return categorical({ps: wordToDistribution(word), vs: vocab});\n"
        "  };\n"
        "  let generateSentence = function(lastState, sentence) {\n"
        "    let word = transition(lastState);\n"
        "    if (word == 'stop') return ['stop'];\n"
        "    return [word].concat(generateSentence(word, sentence));\n"
        "  };\n"
        "  let obs = ['dogs', 'chase', 'cats', 'stop'];\n"
        "  condition(comparray(obs, generateSentence('start')));\n"
        "  let newSentence = generateSentence('start');\n"
        "  condition(newSentence[0] == 'dogs');\n"
        "  return newSentence[1];\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-observing-sequences/ex1.c",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Using the bigram model (vocab = {dogs, cats, chase, sleep, stop}, "
        "Dirichlet-drift priors): observe 'dogs chase cats'. In a *second* sentence, "
        "the second word is 'chase'. Show that 'dogs' is the most likely first word. "
        "End with the Infer(...) returning the distribution over the first word."
    ),
    "groundtruth_code": (
        SEQ_HELPER +
        "Infer({method:'MCMC', burn:10000, samples: 50000, onlyMAP: false}, function() {\n"
        "  let vocab = ['dogs', 'cats', 'chase', 'sleep', 'stop'];\n"
        "  var wordToDistribution = mem(function(word) {\n"
        "    return dirichletDrift({alpha:ones([vocab.length,1]), concentration:10});\n"
        "  });\n"
        "  var transition = function(word) {\n"
        "    return categorical({ps: wordToDistribution(word), vs: vocab});\n"
        "  };\n"
        "  let generateSentence = function(lastState, sentence) {\n"
        "    let word = transition(lastState);\n"
        "    if (word == 'stop') return ['stop'];\n"
        "    return [word].concat(generateSentence(word, sentence));\n"
        "  };\n"
        "  let obs = ['dogs', 'chase', 'cats', 'stop'];\n"
        "  condition(comparray(obs, generateSentence('start')));\n"
        "  let newSentence = generateSentence('start');\n"
        "  condition(newSentence[1] == 'chase');\n"
        "  return newSentence[0];\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-observing-sequences/ex2.a",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Using the bigram model from Ex 1.b (observe 'dogs chase cats stop'): in a "
        "second sentence, the first word is now 'cats' instead of 'dogs'. End with "
        "the Infer(...) returning the marginal distribution over the second word."
    ),
    "groundtruth_code": (
        SEQ_HELPER +
        "Infer({method:'MCMC', burn:10000, samples: 50000, onlyMAP: false}, function() {\n"
        "  let vocab = ['dogs', 'cats', 'chase', 'sleep', 'stop'];\n"
        "  var wordToDistribution = mem(function(word) {\n"
        "    return dirichletDrift({alpha:ones([vocab.length,1]), concentration:10});\n"
        "  });\n"
        "  var transition = function(word) {\n"
        "    return categorical({ps: wordToDistribution(word), vs: vocab});\n"
        "  };\n"
        "  let generateSentence = function(lastState, sentence) {\n"
        "    let word = transition(lastState);\n"
        "    if (word == 'stop') return ['stop'];\n"
        "    return [word].concat(generateSentence(word, sentence));\n"
        "  };\n"
        "  let obs = ['dogs', 'chase', 'cats', 'stop'];\n"
        "  condition(comparray(obs, generateSentence('start')));\n"
        "  let newSentence = generateSentence('start');\n"
        "  condition(newSentence[0] == 'cats');\n"
        "  return newSentence[1];\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-observing-sequences/ex2.c",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "write_from_scratch",
    "eval_mode": "samples",
    "answer_shape": "samples",
    "prompt": (
        "Define a hidden Markov model for sentence generation: words have parts of "
        "speech (N for nouns 'dogs','cats'; V for verbs 'chase','sleep'; 'stop'). "
        "Markov transitions are between POS, not words; words are then drawn given POS. "
        "Use Dirichlet-drift priors for the per-POS transition distributions. End your "
        "program with `generateSentence('start')` (a single sample - the harness will "
        "rerun your program multiple times to estimate the distribution)."
    ),
    "groundtruth_code": (
        "var drawWord = function(pos){\n"
        "  return (pos==\"N\") ? uniformDraw(['dogs','cats']) :\n"
        "         (pos==\"V\") ? uniformDraw(['chase','sleep']) : \n"
        "         'stop';\n"
        "};\n"
        "var POS = [\"N\", \"V\", \"stop\"];\n\n"
        "var posToDistribution = mem(function(pos) {\n"
        "  return dirichletDrift({alpha:ones([POS.length,1]), concentration:10});\n"
        "});\n\n"
        "var transition = function(pos) {\n"
        "  return categorical({ps: posToDistribution(pos), vs: POS});\n"
        "};\n\n"
        "var generateSentence = function(lastPOS) {\n"
        "  var nextPOS = transition(lastPOS);\n"
        "  var word = drawWord(nextPOS);\n"
        "  return (word == 'stop') ? [word] : [word].concat(generateSentence(nextPOS));\n"
        "};\n\n"
        "generateSentence(\"start\");\n"
    ),
})

ATOMS.append({
    "id": "probmods2-observing-sequences/ex2.d",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Using a hidden Markov model for sentence generation (N for 'dogs','cats'; V "
        "for 'chase','sleep'; 'stop'; Dirichlet-drift over POS transitions): observe "
        "the sentence 'dogs chase cats stop'. Then in a new sentence whose first word "
        "is 'cats', end with the Infer(...) returning the distribution over the "
        "second word."
    ),
    "groundtruth_code": (
        SEQ_HELPER +
        "var drawWord = function(pos){\n"
        "  return (pos==\"N\") ? uniformDraw(['dogs','cats']) :\n"
        "         (pos==\"V\") ? uniformDraw(['chase','sleep']) : \n"
        "         'stop';\n"
        "};\n"
        "var POS = [\"N\", \"V\", \"stop\"];\n\n"
        "Infer({method:'MCMC', burn:10000, samples: 1000, lag:10, onlyMAP: false}, function() {\n"
        "  var posToDistribution = mem(function(pos) {\n"
        "    return dirichletDrift({alpha:ones([POS.length,1]), concentration:10});\n"
        "  });\n\n"
        "  var transition = function(pos) {\n"
        "    return categorical({ps: posToDistribution(pos), vs: POS});\n"
        "  };\n\n"
        "  let generateSentence = function(lastPOS) {\n"
        "    let nextPOS = transition(lastPOS);\n"
        "    let word = drawWord(nextPOS);\n"
        "    return (word == 'stop') ? [word] : [word].concat(generateSentence(nextPOS));\n"
        "  };\n"
        "  let obs = ['dogs', 'chase', 'cats', 'stop'];\n"
        "  condition(comparray(obs, generateSentence('start')));\n\n"
        "  let newSentence = generateSentence('start');\n"
        "  condition(newSentence[0] == 'cats');\n"
        "  return newSentence[1];\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-observing-sequences/ex3.a",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Extend the hidden Markov model with determiners (D for 'the','a') and adverb "
        "('A' for 'dilligently'). Use singular forms 'dog','cat','chases','sleeps'. "
        "Condition on the sentence 'the dog chases a cat stop' having been generated, "
        "then sample 5 new sentences. Use `factor(comparray(obs, generateSentence('start'))*5)` "
        "and `onlyMAP: true` for tractability. End with the Infer(...) returning the joint "
        "distribution `{sent1, sent2, sent3, sent4, sent5}`."
    ),
    "groundtruth_code": (
        SEQ_HELPER +
        "var drawWord = function(pos){\n"
        "  return (pos==\"N\") ? uniformDraw(['dog','cat']) :\n"
        "         (pos==\"V\") ? uniformDraw(['chases','sleeps']) : \n"
        "         (pos==\"D\") ? uniformDraw(['the','a']) :\n"
        "         (pos==\"A\") ? 'dilligently' : \n"
        "         'stop';\n"
        "};\n"
        "var POS = [\"N\", \"V\", \"D\", \"A\", \"stop\"];\n\n"
        "Infer({method:'MCMC', burn:10000, samples: 1000, lag:10, onlyMAP: true}, function() {\n"
        "  var posToDistribution = mem(function(pos) {\n"
        "    return dirichletDrift({alpha:ones([POS.length,1]), concentration:10});\n"
        "  });\n\n"
        "  var transition = function(pos) {\n"
        "    return categorical({ps: posToDistribution(pos), vs: POS});\n"
        "  };\n\n"
        "  let generateSentence = function(lastPOS) {\n"
        "    let nextPOS = transition(lastPOS);\n"
        "    let word = drawWord(nextPOS);\n"
        "    return (word == 'stop') ? [word] : [word].concat(generateSentence(nextPOS));\n"
        "  };\n"
        "  let obs = ['the', 'dog', 'chases', 'a', 'cat', 'stop'];\n\n"
        "  factor(comparray(obs, generateSentence('start'))*5);\n\n"
        "  var sent1 = generateSentence('start');\n"
        "  var sent2 = generateSentence('start');\n"
        "  var sent3 = generateSentence('start');\n"
        "  var sent4 = generateSentence('start');\n"
        "  var sent5 = generateSentence('start');\n"
        "  return {sent1: sent1, sent2: sent2, sent3: sent3, sent4: sent4, sent5: sent5};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-observing-sequences/ex3.b",
    "source": "exercises/05-observing-sequences.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Use a phrase structure grammar instead of an HMM. Vocabulary: D='the','a'; "
        "N='cat','dog'; V='chases','sleeps'; A='diligently'. Production rules: "
        "AP -> A; NP -> [D, N]; VP -> [V, AP] | [V, NP]; S -> [NP, VP]. Condition on "
        "`[['the', 'dog'], ['chases', ['a', 'cat']]]` being a sentence (using comparray "
        "and S()), then sample 5 new sentences. Use MCMC with onlyMAP=true. End with "
        "the Infer(...) returning `{sent1...sent5}`."
    ),
    "groundtruth_code": (
        SEQ_HELPER +
        "var uniformDraw = function (xs) {return xs[randomInteger(xs.length)]};\n\n"
        "var D  = function() {return uniformDraw(['the', 'a'])};\n"
        "var N  = function() {return uniformDraw(['cat', 'dog'])};\n"
        "var V  = function() {return uniformDraw(['chases', 'sleeps'])};\n"
        "var A  = function() {return uniformDraw(['diligently'])};\n"
        "var AP = function() {return uniformDraw([A()])};\n"
        "var NP = function() {return [D(), N()]};\n"
        "var VP = function() {return uniformDraw([[V(), AP()],\n"
        "                                         [V(), NP()]])};\n"
        "var S  = function() {return [NP(), VP()]};\n\n"
        "Infer({method:'MCMC', burn:10000, samples: 1000, onlyMAP: true}, function() {\n"
        "  let obs = [['the', 'dog'], ['chases', ['a', 'cat']]];\n"
        "  condition(comparray(obs, S()));\n\n"
        "  var sent1 = S();\n"
        "  var sent2 = S();\n"
        "  var sent3 = S();\n"
        "  var sent4 = S();\n"
        "  var sent5 = S();\n"
        "  return {sent1: sent1, sent2: sent2, sent3: sent3, sent4: sent4, sent5: sent5};\n"
        "});\n"
    ),
})


if __name__ == "__main__":
    write_atoms(ATOMS, append=True)

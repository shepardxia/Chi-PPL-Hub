"""Batch D1: agents-as-programs (9 atoms)."""
from atom_writer import write_atoms

ATOMS = []

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex1.a",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Modify the `factor(...)` line in the coin-flipping model so that the soft "
        "condition makes heads happen with approximately 95% probability:\n\n"
        "```\n"
        "Infer({method: 'enumerate'}, function () {\n"
        "  var A = flip()\n"
        "  factor(A) // edit this line\n"
        "  return A\n"
        "});\n"
        "```\n\n"
        "Hint: a factor of `c` gives weight exp(c). End your program with the Infer(...)."
    ),
    "groundtruth_code": (
        "Infer({method: 'enumerate'}, function () {\n"
        "  var A = flip();\n"
        "  factor(A*3);\n"
        "  return A;\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex1.b",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Three coins are flipped. Use `factor` to favor outcomes where the number of "
        "heads (true's) equals 2:\n\n"
        "```\n"
        "Infer({}, function() {\n"
        "    var a = flip(0.5);\n"
        "    var b = flip(0.5);\n"
        "    var c = flip(0.5);\n"
        "    factor(...);  // fill in\n"
        "    return a;\n"
        "})\n"
        "```\n\n"
        "End your program with the Infer(...)."
    ),
    "groundtruth_code": (
        "Infer({}, function() {\n"
        "    var a = flip(0.5);\n"
        "    var b = flip(0.5);\n"
        "    var c = flip(0.5);\n"
        "    factor(1*((a+b+c)==2));\n"
        "    return a;\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex2.a",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Ultimatum game: a proposer allocates $10 between themselves and a responder in "
        "$1 increments. The responder accepts or rejects. If accepted, both get the "
        "split; if rejected, both get $0.\n\n"
        "Assume the responder is a strict utilitarian (accepts any offer >= $1). The "
        "proposer is a soft maximizer who wants to keep as much of the $10 as possible. "
        "End your program with the Infer(...) over the proposer's offer.\n\n"
        "```\n"
        "var responder = function(offer) { /* fill in */ }\n"
        "Infer({method: \"enumerate\"}, function(){\n"
        "  // sample offer, compute reward = responder(offer) ? (10-offer) : 0\n"
        "  // factor(reward)\n"
        "  return offer\n"
        "})\n"
        "```"
    ),
    "groundtruth_code": (
        "var responder = function(offer) {    \n"
        "    return (offer>0 ? true : false);\n"
        "};\n\n"
        "Infer({method: \"enumerate\"}, function(){\n"
        "    var offer = uniformDraw([0,1,2,3,4,5,6,7,8,9,10]);\n"
        "    var reward = responder(offer) ? (10 - offer) : 0;\n"
        "    factor(reward);\n"
        "    return offer;\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex2.b",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "fill_in_blank",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Modified ultimatum game: the responder accepts in proportion to the fraction "
        "of $10 allocated to her, raised to a power `alpha = 2` (a spitefulness "
        "parameter):\n\n"
        "```\n"
        "var alpha = 2;\n"
        "var responder = function(offer, alpha) {\n"
        "  var p = Math.pow(offer/10, alpha);\n"
        "  return flip(p);\n"
        "}\n"
        "```\n\n"
        "Use offers 0..10 in $1 increments. End with the Infer(...) over the proposer's "
        "offer (factor on reward)."
    ),
    "groundtruth_code": (
        "var alpha = 2;\n\n"
        "var responder = function(offer, alpha) {    \n"
        "    var p = Math.pow(offer/10,alpha);\n"
        "    return flip(p);\n"
        "};\n\n"
        "Infer({method: \"enumerate\"}, function(){\n"
        "    var offer = uniformDraw([0,1,2,3,4,5,6,7,8,9,10]);\n"
        "    var reward = responder(offer,alpha) ? (10 - offer) : 0;\n"
        "    factor(reward);\n"
        "    return offer;\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex2.d",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "In the ultimatum game (responder accepts with probability (offer/10)^alpha), "
        "the proposer doesn't know alpha but believes it is uniformly distributed on "
        "[0.5, 5]. The proposer offered $2 and the responder rejected it. End with "
        "the Infer(...) returning the posterior over alpha. Use MCMC with 50000 samples."
    ),
    "groundtruth_code": (
        "var responder = function(offer, alpha) {    \n"
        "    var p = Math.pow(offer/10,alpha);\n"
        "    return flip(p);\n"
        "};\n\n"
        "Infer({method: \"MCMC\", samples:50000}, function(){\n"
        "    var alpha = uniform(0.5,5);\n"
        "    var offer = 2;\n"
        "    var reward = responder(offer, alpha) ? (10 - offer) : 0;\n"
        "    condition(reward==0);\n"
        "    return alpha;\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex2.e",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Continuing the ultimatum game with uncertain alpha (uniform [0.5, 5]): the "
        "proposer offered $2 and the responder rejected. They will play a *second* round. "
        "What should the proposer offer in round 2?\n\n"
        "Two-stage inference: first compute the posterior over alpha given round 1 "
        "rejection (MCMC, 50000 samples). Then use that posterior to choose a round-2 "
        "offer maximizing expected payoff (forward sample 1000 alphas; for each, MCMC "
        "5000 to find best offer). End with the Infer(...) over round-2 offers."
    ),
    "groundtruth_code": (
        "var responder = function(offer, alpha) {    \n"
        "    var p = Math.pow(offer/10,alpha);\n"
        "    return flip(p);\n"
        "};\n\n"
        "var proposer1 = Infer({method: \"MCMC\", samples:50000}, function(){\n"
        "    var alpha = uniform(0.5,5);\n"
        "    var offer1 = 2;\n"
        "    var reward1 = responder(offer1, alpha) ? (10 - offer1) : 0;\n"
        "    condition(reward1==0);\n"
        "    return alpha;\n"
        "});\n\n"
        "Infer({method: \"forward\", samples:1000}, function(){\n"
        "     var alpha2 = sample(proposer1);\n"
        "     var proposer2 = Infer({method: \"MCMC\", samples:5000}, function(){\n"
        "       var offer2 = uniformDraw([0,1,2,3,4,5,6,7,8,9,10]);\n"
        "       var reward2 = responder(offer2, alpha2) ? (10 - offer2) : 0;\n"
        "       factor(reward2);\n"
        "       return offer2;\n"
        "      });\n"
        "      return sample(proposer2);\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex3",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Prisoner's Dilemma. Two thieves are interrogated separately. If a thief "
        "confesses she gets a lenient sentence (`lenient = 6` years). If she doesn't "
        "but the other does, she gets 10 years. If neither confesses, both go free. "
        "With `lenient = 6`, use `factor(percentYearsFreedom)` (where "
        "percentYearsFreedom = (10 - years_in_jail) / 10). The other thief flips "
        "uniformly to decide whether to rat. End your program with the Infer(...) "
        "returning the posterior over the focal thief's rat decision."
    ),
    "groundtruth_code": (
        "var thiefRats = function(){\n"
        "  return flip();\n"
        "};\n\n"
        "var lenient = 6;\n\n"
        "Infer({}, function(){\n"
        "  var otherThiefRats = thiefRats();\n"
        "  var IRat = thiefRats();\n"
        "  var years = (otherThiefRats? \n"
        "              (IRat? lenient : 10) : \n"
        "              (IRat? lenient : 0));\n"
        "  var percentYearsFreedom = (10-years)/10;\n"
        "  factor(percentYearsFreedom);\n"
        "  return IRat;\n"
        "});\n"
    ),
})

# RSA priors used in 4a, 4b
RSA_PRIORS = (
    "var meaningPrior = function() {\n"
    "  uniformDraw([\n"
    "    {shape: \"square\", color: \"blue\"},\n"
    "    {shape: \"circle\", color: \"blue\"},\n"
    "    {shape: \"square\", color: \"green\"}\n"
    "  ])\n"
    "};\n\n"
    "var utterances = [\"blue\",\"green\",\"square\",\"circle\"];\n\n"
    "var meaning = function(utterance, obj){\n"
    "  (utterance === \"blue\" || utterance === \"green\") ? utterance === obj.color :\n"
    "  (utterance === \"circle\" || utterance === \"square\") ? utterance === obj.shape :\n"
    "  true\n"
    "};\n\n"
)

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex4.a",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "write_from_scratch",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "alpha_001": "distribution",
        "alpha_1": "distribution",
        "alpha_4": "distribution",
        "alpha_10": "distribution",
    }},
    "prompt": (
        "Implement the Frank & Goodman RSA model. The world has three objects: "
        "{blue square, blue circle, green square}, drawn uniformly. Possible "
        "utterances: 'blue', 'green', 'square', 'circle'. Truth function: "
        "color/shape utterances must match the corresponding attribute, otherwise true.\n\n"
        "Build:\n"
        "- `literalListener(utterance)`: Infer over meaningPrior conditional on truth\n"
        "- `speaker(obj, alpha)`: uniformDraw utterance, factor alpha * literalListener(utterance).score(obj)\n"
        "- `pragmaticListener(utterance, alpha)`: Infer over meaningPrior conditional on speaker(obj, alpha)\n\n"
        "Return an object literal with `pragmaticListener('blue', alpha)` for four "
        "alphas, keyed as:\n"
        "  - `alpha_001` (alpha=0.01)\n"
        "  - `alpha_1` (alpha=1)\n"
        "  - `alpha_4` (alpha=4)\n"
        "  - `alpha_10` (alpha=10)"
    ),
    "groundtruth_code": (
        RSA_PRIORS +
        "var literalListener = function(utterance){\n"
        "  return Infer({model: function(){\n"
        "    var obj = meaningPrior();\n"
        "    condition(meaning(utterance, obj));\n"
        "    return obj;\n"
        "  }});\n"
        "};\n\n"
        "var speaker = function(obj,alpha){\n"
        "  return Infer({model: function(){\n"
        "    var utterance = uniformDraw(utterances);\n"
        "    factor(alpha * literalListener(utterance).score(obj));\n"
        "    return utterance;\n"
        "  }});\n"
        "};\n\n"
        "var pragmaticListener = function(utterance,alpha){\n"
        "  return Infer({model: function(){\n"
        "    var obj = meaningPrior();\n"
        "    observe(speaker(obj,alpha),utterance);\n"
        "    return obj;\n"
        "  }});\n"
        "};\n\n"
        "({\n"
        "  alpha_001: pragmaticListener(\"blue\", 0.01),\n"
        "  alpha_1: pragmaticListener(\"blue\", 1),\n"
        "  alpha_4: pragmaticListener(\"blue\", 4),\n"
        "  alpha_10: pragmaticListener(\"blue\", 10)\n"
        "})\n"
    ),
})

ATOMS.append({
    "id": "probmods2-agents-as-programs/ex4.b",
    "source": "exercises/04.1-agents-as-programs.md",
    "task_type": "modify_given",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "L1": "distribution",
        "L2": "distribution",
    }},
    "prompt": (
        "Extend the RSA model with a level-2 listener: a `speaker2` who reasons about "
        "the *pragmatic* listener (instead of the literal listener), and a `listener3` "
        "(L2) who reasons about speaker2. Use alpha = 1.\n\n"
        "World: {blue square, blue circle, green square}. Utterances: blue, green, "
        "square, circle. Return an object literal `{L1, L2}` where L1 is "
        "`pragmaticListener('blue')` and L2 is `listener3('blue')`."
    ),
    "groundtruth_code": (
        RSA_PRIORS +
        "var alpha = 1;\n\n"
        "var literalListener = function(utterance){\n"
        "  return Infer({model: function(){\n"
        "    var obj = meaningPrior();\n"
        "    condition(meaning(utterance, obj));\n"
        "    return obj;\n"
        "  }});\n"
        "};\n\n"
        "var speaker = function(obj){\n"
        "  return Infer({model: function(){\n"
        "    var utterance = uniformDraw(utterances);\n"
        "    factor(alpha * literalListener(utterance).score(obj));\n"
        "    return utterance;\n"
        "  }});\n"
        "};\n\n"
        "var pragmaticListener = function(utterance){\n"
        "  return Infer({model: function(){\n"
        "    var obj = meaningPrior();\n"
        "    observe(speaker(obj),utterance);\n"
        "    return obj;\n"
        "  }});\n"
        "};\n\n"
        "var speaker2 = function(obj){\n"
        "  return Infer({model: function(){\n"
        "    var utterance = uniformDraw(utterances);\n"
        "    factor(alpha * pragmaticListener(utterance).score(obj));\n"
        "    return utterance;\n"
        "  }});\n"
        "};\n\n"
        "var listener3 = function(utterance){\n"
        "  return Infer({model: function(){\n"
        "    var obj = meaningPrior();\n"
        "    observe(speaker2(obj),utterance);\n"
        "    return obj;\n"
        "  }});\n"
        "};\n\n"
        "({\n"
        "  L1: pragmaticListener(\"blue\"),\n"
        "  L2: listener3(\"blue\")\n"
        "})\n"
    ),
})


if __name__ == "__main__":
    write_atoms(ATOMS, append=True)

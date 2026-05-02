"""probmods2-conditioning atoms (13). Truncates atomized.jsonl, then writes."""
from atom_writer import write_atoms

# Schema additions in this rewrite:
#   - groundtruth_code now ends with the answer expression (no viz/display/print).
#   - "answer_shape" field describes how to interpret the program's return value:
#       "value"        -> a scalar (number/string/list); compare with eval metric
#       "distribution" -> a Distribution object (from Infer); compare KL/TV
#       "samples"      -> a stochastic expression; harness re-runs N seeded times
#       {"record": {field: shape, ...}} -> object literal; recurse per field

ATOMS = [
    {
        "id": "probmods2-conditioning/ex1.a",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "value",
        "answer_shape": "value",
        "prompt": (
            "I flip a fair coin. Compute the probability that it lands heads. "
            "End your program with the numeric answer."
        ),
        "groundtruth_code": (
            "var model = function() { return flip() ? \"H\" : \"T\" };\n"
            "Math.exp(Infer({method:'enumerate'}, model).score('H'));\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex1.b",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "I have a biased coin with P(heads) = 0.9, and a fair coin. Someone hands me "
            "one of the two coins, chosen uniformly at random, without telling me which. "
            "I flip it three times. Given that the first two flips landed heads, what is "
            "the posterior distribution over the third flip? End your program with the "
            "Infer(...) call returning that distribution."
        ),
        "groundtruth_code": (
            "var flipCoin = function(coinType) {\n"
            "  return coinType == \"fair\" ? flip() : flip(0.9);\n"
            "}\n"
            "var model = function() {\n"
            "  var coinType = flip() ? \"fair\" : \"biased\";\n"
            "  var flip1 = flipCoin(coinType);\n"
            "  var flip2 = flipCoin(coinType);\n"
            "  var flip3 = flipCoin(coinType);\n"
            "  condition(flip1 && flip2);\n"
            "  return flip3;\n"
            "};\n"
            "Infer({method:'enumerate'}, model);\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex1.c",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "I have a biased coin with P(heads) = 0.9, and a fair coin. Someone hands me "
            "one of the two coins, chosen uniformly at random, without telling me which. "
            "I flip it three times. Given that all three flips landed heads, what is the "
            "posterior distribution over the coin type (\"fair\" or \"biased\")? End your "
            "program with the Infer(...) call."
        ),
        "groundtruth_code": (
            "var flipCoin = function(coinType) {\n"
            "  return coinType == \"fair\" ? flip() : flip(0.9);\n"
            "}\n"
            "var model = function() {\n"
            "  var coinType = flip() ? \"fair\" : \"biased\";\n"
            "  var flip1 = flipCoin(coinType);\n"
            "  var flip2 = flipCoin(coinType);\n"
            "  var flip3 = flipCoin(coinType);\n"
            "  condition(flip1 && flip2 && flip3);\n"
            "  return coinType;\n"
            "};\n"
            "Infer({method:'enumerate'}, model);\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex1.d",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "I have a biased coin with P(heads) = 0.9, and a fair coin. Someone hands me "
            "one of the two coins, chosen uniformly at random, without telling me which. "
            "I flip it three times. Given that the first two flips were different, what is "
            "the posterior distribution over the third flip? End your program with the "
            "Infer(...) call."
        ),
        "groundtruth_code": (
            "var flipCoin = function(coinType) {\n"
            "  return coinType == \"fair\" ? flip() : flip(0.9);\n"
            "}\n"
            "var model = function() {\n"
            "  var coinType = flip() ? \"fair\" : \"biased\";\n"
            "  var flip1 = flipCoin(coinType);\n"
            "  var flip2 = flipCoin(coinType);\n"
            "  var flip3 = flipCoin(coinType);\n"
            "  condition(flip1 != flip2);\n"
            "  return flip3;\n"
            "};\n"
            "Infer({method:'enumerate'}, model);\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex2.a",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "record",
        "answer_shape": {"record": {
            "original": "distribution",
            "intervention": "distribution",
            "conditioning": "distribution",
        }},
        "prompt": (
            "Consider this generative model of a cough symptom:\n\n"
            "```\n"
            "var lungCancer = flip(0.01);\n"
            "var cold = flip(0.2);\n"
            "var cough = (cold && flip(0.5)) || (lungCancer && flip(0.3));\n"
            "```\n\n"
            "Show that *intervening* on `lungCancer` (setting it to true directly) "
            "produces the same marginal of `cough` as *conditioning* on `lungCancer`. "
            "Return an object literal with three keys, each a distribution over `cough`:\n"
            "  - `original`: the unconditional marginal\n"
            "  - `intervention`: the marginal under `lungCancer = true`\n"
            "  - `conditioning`: the marginal under `condition(lungCancer)`"
        ),
        "groundtruth_code": (
            "({\n"
            "  original: Infer({method: \"enumerate\"}, function() {\n"
            "    var lungCancer = flip(0.01);\n"
            "    var cold = flip(0.2);\n"
            "    var cough = (cold && flip(0.5)) || (lungCancer && flip(0.3));\n"
            "    return cough;\n"
            "  }),\n"
            "  intervention: Infer({method: \"enumerate\"}, function() {\n"
            "    var lungCancer = true;\n"
            "    var cold = flip(0.2);\n"
            "    var cough = (cold && flip(0.5)) || (lungCancer && flip(0.3));\n"
            "    return cough;\n"
            "  }),\n"
            "  conditioning: Infer({method: \"enumerate\"}, function() {\n"
            "    var lungCancer = flip(0.01);\n"
            "    condition(lungCancer);\n"
            "    var cold = flip(0.2);\n"
            "    var cough = (cold && flip(0.5)) || (lungCancer && flip(0.3));\n"
            "    return cough;\n"
            "  })\n"
            "})\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex2.b",
        "source": "exercises/conditioning.md",
        "task_type": "modify_given",
        "eval_mode": "record",
        "answer_shape": {"record": {
            "original": "distribution",
            "intervention": "distribution",
            "conditioning": "distribution",
        }},
        "prompt": (
            "Using the same lungCancer/cold/cough generative model:\n\n"
            "```\n"
            "var lungCancer = flip(0.01);\n"
            "var cold = flip(0.2);\n"
            "var cough = (cold && flip(0.5)) || (lungCancer && flip(0.3));\n"
            "```\n\n"
            "Construct a case where intervening produces a *different* result from "
            "conditioning. You don't need to introduce new variables - think about what "
            "other queries you can ask. Return an object literal with three distributions "
            "(over the *queried* variable, not necessarily `cough`):\n"
            "  - `original`: unconditional marginal\n"
            "  - `intervention`: under intervention on the chosen variable\n"
            "  - `conditioning`: under conditioning on the chosen variable"
        ),
        "groundtruth_code": (
            "({\n"
            "  original: Infer({method: \"enumerate\"}, function() {\n"
            "    var lungCancer = flip(0.01);\n"
            "    var cold = flip(0.2);\n"
            "    var cough = (cold && flip(0.5)) || (lungCancer && flip(0.3));\n"
            "    return lungCancer;\n"
            "  }),\n"
            "  intervention: Infer({method: \"enumerate\"}, function() {\n"
            "    var lungCancer = flip(0.01);\n"
            "    var cold = flip(0.2);\n"
            "    var cough = true;\n"
            "    return lungCancer;\n"
            "  }),\n"
            "  conditioning: Infer({method: \"enumerate\"}, function() {\n"
            "    var lungCancer = flip(0.01);\n"
            "    var cold = flip(0.2);\n"
            "    var cough = (cold && flip(0.5)) || (lungCancer && flip(0.3));\n"
            "    condition(cough);\n"
            "    return lungCancer;\n"
            "  })\n"
            "})\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex4.b",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "Here is a model of how niceness affects whether a person smiles:\n\n"
            "```\n"
            "var smilesModel = function() {\n"
            "  var nice = mem(function(person) { flip(.7) });\n"
            "  var smiles = function(person) {\n"
            "    return nice(person) ? flip(.8) : flip(.5);\n"
            "  }\n"
            "  condition(smiles('alice') && smiles('bob') && smiles('alice'));\n"
            "  return nice('alice');\n"
            "}\n"
            "```\n\n"
            "Extend this into a function `extendedSmilesModel` that captures two "
            "additional factors:\n"
            "1. People smile 80% of the time if they want something from you, and 50% otherwise.\n"
            "2. Nice people want something from you 20% of the time; non-nice people 50%.\n\n"
            "Nice people should still smile more often regardless of whether they want "
            "something. Niceness is a stable property of a person; whether they want "
            "something can vary. Have `extendedSmilesModel` return whether Alice smiles "
            "today, and end your program with `Infer({method: 'enumerate'}, "
            "extendedSmilesModel)` returning that distribution."
        ),
        "groundtruth_code": (
            "var extendedSmilesModel = function() {\n"
            "  var nice = mem(function(person) { flip(.7) });\n"
            "  var wantsSomething = function(person) {\n"
            "    return flip(nice(person) ? .2 : .5);\n"
            "  }\n"
            "  var smiles = function(person, wants) {\n"
            "    return (wants ? flip(.8) : flip(.5))\n"
            "            || (nice(person) ? flip(.8) : flip(.5));\n"
            "  }\n"
            "  var wants = wantsSomething('alice');\n"
            "  return smiles('alice', wants);\n"
            "};\n"
            "Infer({method: \"enumerate\"}, extendedSmilesModel);\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex4.c",
        "source": "exercises/conditioning.md",
        "task_type": "modify_given",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "Given this `extendedSmilesModel` skeleton:\n\n"
            "```\n"
            "var extendedSmilesModel = function() {\n"
            "  var nice = mem(function(person) { flip(.7) });\n"
            "  var wantsSomething = function(person) {\n"
            "    return flip(nice(person) ? .2 : .5);\n"
            "  }\n"
            "  var smiles = function(person, wants) {\n"
            "    return (wants ? flip(.8) : flip(.5))\n"
            "            || (nice(person) ? flip(.8) : flip(.5));\n"
            "  }\n"
            "  // ... your code here ...\n"
            "}\n"
            "```\n\n"
            "Suppose you've seen Bob five times this week and each time he was *not* "
            "smiling. Today, you see him smiling. Modify the model body to compute the "
            "posterior probability that Bob wants something from you today. Niceness is "
            "stable (use `mem` as given); whether he wants something varies day-to-day. "
            "End your program with `Infer({method: 'enumerate'}, extendedSmilesModel)`."
        ),
        "groundtruth_code": (
            "var extendedSmilesModel = function() {\n"
            "  var nice = mem(function(person) { flip(.7) });\n"
            "  var wantsSomething = function(person) {\n"
            "    return flip(nice(person) ? .2 : .5);\n"
            "  }\n"
            "  var smiles = function(person, wants) {\n"
            "    return (wants ? flip(.8) : flip(.5))\n"
            "            || (nice(person) ? flip(.8) : flip(.5));\n"
            "  }\n"
            "  var wantsToday = wantsSomething('bob');\n"
            "  condition(!smiles('bob', wantsSomething('bob')));\n"
            "  condition(!smiles('bob', wantsSomething('bob')));\n"
            "  condition(!smiles('bob', wantsSomething('bob')));\n"
            "  condition(!smiles('bob', wantsSomething('bob')));\n"
            "  condition(!smiles('bob', wantsSomething('bob')));\n"
            "  condition(smiles('bob', wantsToday));\n"
            "  return wantsToday;\n"
            "};\n"
            "Infer({method: \"enumerate\"}, extendedSmilesModel);\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex5.a",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "record",
        "answer_shape": {"record": {
            "rain": "distribution",
            "sprinkler": "distribution",
        }},
        "prompt": (
            "I have a sprinkler in my garden that turns on each morning at random - half "
            "the time, independently each day. I live in a city where it rains on 30% of "
            "mornings. The lawn gets wet whenever the sprinkler turns on, it rains, or "
            "both. One morning, I notice my lawn is wet. Return an object literal with "
            "two keys:\n"
            "  - `rain`: posterior distribution over whether it rained\n"
            "  - `sprinkler`: posterior distribution over whether the sprinkler turned on"
        ),
        "groundtruth_code": (
            "({\n"
            "  rain: Infer({method: \"enumerate\"}, function() {\n"
            "    var sprinkler = flip();\n"
            "    var rain = flip(0.3);\n"
            "    var wetLawn = sprinkler || rain;\n"
            "    condition(wetLawn);\n"
            "    return rain;\n"
            "  }),\n"
            "  sprinkler: Infer({method: \"enumerate\"}, function() {\n"
            "    var sprinkler = flip();\n"
            "    var rain = flip(0.3);\n"
            "    var wetLawn = sprinkler || rain;\n"
            "    condition(wetLawn);\n"
            "    return sprinkler;\n"
            "  })\n"
            "})\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex5.b",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "Same setup as before: each person's sprinkler turns on with probability 0.5 "
            "each morning, it rains with probability 0.3, and lawns are wet whenever "
            "either occurs. My neighbor Kelsey has the same kind of sprinkler. One "
            "morning, both my lawn and Kelsey's lawn are wet. End your program with "
            "an `Infer(...)` returning the posterior distribution over whether it rained."
        ),
        "groundtruth_code": (
            "Infer({method: \"enumerate\"}, function() {\n"
            "  var rain = flip(0.3);\n"
            "  var mySprinkler = flip();\n"
            "  var herSprinkler = flip();\n"
            "  var myLawnIsWet = mySprinkler || rain;\n"
            "  var herLawnIsWet = herSprinkler || rain;\n"
            "  condition(myLawnIsWet && herLawnIsWet);\n"
            "  return rain;\n"
            "});\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex5.c",
        "source": "exercises/conditioning.md",
        "task_type": "write_from_scratch",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "Same setup as before: each person's sprinkler turns on with probability 0.5 "
            "each morning, it rains with probability 0.3, and lawns are wet whenever "
            "either occurs. Five people in the area - me, Kelsey, Kevin, Manu, and Josh - "
            "all observe their lawns wet on the same morning. Use `mem` so each person's "
            "sprinkler is modeled independently. End your program with `Infer(...)` "
            "returning the posterior over whether it rained."
        ),
        "groundtruth_code": (
            "Infer({method: \"enumerate\"}, function() {\n"
            "  var rain = flip(0.3);\n"
            "  var sprinkler = mem(function(person) { return flip() });\n"
            "  var wetLawn = function(person) { return rain || sprinkler(person) };\n"
            "  condition(wetLawn(\"me\"));\n"
            "  condition(wetLawn(\"Kelsey\"));\n"
            "  condition(wetLawn(\"Kevin\"));\n"
            "  condition(wetLawn(\"Manu\"));\n"
            "  condition(wetLawn(\"Josh\"));\n"
            "  return rain;\n"
            "});\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex6.c",
        "source": "exercises/conditioning.md",
        "task_type": "fill_in_blank",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "A machine randomly draws a letter of the word \"game\" with probabilities "
            "{g: 0.05, a: 0.45, m: 0.05, e: 0.45}. Bob's probability of winning given "
            "letter h is 1/k^2 where k is the position of that letter in the word "
            "\"game\" (so g=1, a=2, m=3, e=4). We observe that Bob won, but don't know "
            "which letter he drew. Fill in the `...`'s in the program below to compute "
            "p(letter | win), and end your program with the resulting distribution:\n\n"
            "```\n"
            "var checkVowel = function(letter) { _.includes(['a', 'e', 'i', 'o', 'u'], letter) };\n"
            "var letterVals = ['g', 'a', 'm', 'e'];\n"
            "var letterProbs = map(function(letter) { checkVowel(letter) ? 0.45 : 0.05 }, letterVals);\n"
            "var letters = Categorical({vs: letterVals, ps: letterProbs});\n\n"
            "Infer({method: 'enumerate'}, function() {\n"
            "  var letter = sample(letters);\n"
            "  var position = letterVals.indexOf(letter) + 1;\n"
            "  var winProb = 1 / Math.pow(position, 2);\n"
            "  condition(...);\n"
            "  return ...;\n"
            "});\n"
            "```"
        ),
        "groundtruth_code": (
            "var checkVowel = function(letter) { _.includes(['a', 'e', 'i', 'o', 'u'], letter) };\n"
            "var letterVals = ['g', 'a', 'm', 'e'];\n"
            "var letterProbs = map(function(letter) { checkVowel(letter) ? 0.45 : 0.05 }, letterVals);\n"
            "var letters = Categorical({vs: letterVals, ps: letterProbs});\n\n"
            "Infer({method: 'enumerate'}, function() {\n"
            "  var letter = sample(letters);\n"
            "  var position = letterVals.indexOf(letter) + 1;\n"
            "  var winProb = 1 / Math.pow(position, 2);\n"
            "  condition(flip(winProb));\n"
            "  return letter;\n"
            "});\n"
        ),
    },
    {
        "id": "probmods2-conditioning/ex6.d",
        "source": "exercises/conditioning.md",
        "task_type": "fill_in_blank",
        "eval_mode": "distribution",
        "answer_shape": "distribution",
        "prompt": (
            "Same casino-game setup: letters {g, a, m, e} drawn with probs "
            "{0.05, 0.45, 0.05, 0.45}, win prob = 1/k^2 with k = position in \"game\". "
            "Fill in the program below so the posterior distribution is over `vowel` vs "
            "`consonant` (instead of over the letter itself), given that Bob won:\n\n"
            "```\n"
            "var checkVowel = function(letter) { _.includes(['a', 'e', 'i', 'o', 'u'], letter) };\n"
            "var letterVals = ['g', 'a', 'm', 'e'];\n"
            "var letterProbs = map(function(letter) { checkVowel(letter) ? 0.45 : 0.05 }, letterVals);\n"
            "var letters = Categorical({vs: letterVals, ps: letterProbs});\n\n"
            "Infer({method: 'enumerate'}, function() {\n"
            "  var letter = sample(letters);\n"
            "  var position = letterVals.indexOf(letter) + 1;\n"
            "  var winProb = 1 / Math.pow(position, 2);\n"
            "  condition(...);\n"
            "  return ...;\n"
            "});\n"
            "```\n\n"
            "End your program with the resulting `Infer(...)` distribution."
        ),
        "groundtruth_code": (
            "var checkVowel = function(letter) { _.includes(['a', 'e', 'i', 'o', 'u'], letter) };\n"
            "var letterVals = ['g', 'a', 'm', 'e'];\n"
            "var letterProbs = map(function(letter) { checkVowel(letter) ? 0.45 : 0.05 }, letterVals);\n"
            "var letters = Categorical({vs: letterVals, ps: letterProbs});\n\n"
            "Infer({method: 'enumerate'}, function() {\n"
            "  var letter = sample(letters);\n"
            "  var position = letterVals.indexOf(letter) + 1;\n"
            "  var winProb = 1 / Math.pow(position, 2);\n"
            "  condition(flip(winProb));\n"
            "  return checkVowel(letter);\n"
            "});\n"
        ),
    },
]


if __name__ == "__main__":
    write_atoms(ATOMS, append=False)

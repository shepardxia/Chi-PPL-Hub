"""Build data/atomized_v2.jsonl with self-contained ('atomic') prompts.

Reads data/atomized.jsonl, applies per-id prompt rewrites for atoms that
referenced earlier subparts ("Same setup as before...") or assumed shared
helpers ("Helpers X, Y, Z are available"), and drops the 4 process-models
atoms whose groundtruth depends on wall-clock timing (`_.now()`) and is
therefore not reproducible.

groundtruth_code is unchanged. groundtruth_output (cached values) is also
unchanged - prompts are presentation-only, not part of the answer.

After running this:
    PYTHONPATH=. .venv/bin/python scripts/cache_groundtruth_outputs.py \
        --dataset data/atomized_v2.jsonl
to re-cache (a no-op since gt_code is unchanged, but useful as a
verification that the v2 file parses).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.io import load_jsonl, write_jsonl


# Atoms whose groundtruth is not reproducible (wall-clock observations).
DROPPED = {
    "probmods2-process-models/ex1",
    "probmods2-process-models/ex3",
    "probmods2-process-models/ex4",
    "probmods2-process-models/ex5",
}


# id -> new prompt. Atoms not in this dict keep their v1 prompt.
REWRITES: dict[str, str] = {}


# ─── conditioning ───────────────────────────────────────────────────
REWRITES["probmods2-conditioning/ex5.b"] = (
    "I have a sprinkler in my garden that turns on each morning at random - half "
    "the time, independently each day. I live in a city where it rains on 30% of "
    "mornings. The lawn gets wet whenever the sprinkler turns on, it rains, or "
    "both. My neighbor Kelsey has the same kind of sprinkler (independent of mine, "
    "also turning on with probability 0.5 each morning). One morning, both my lawn "
    "and Kelsey's lawn are wet. End your program with `Infer(...)` returning the "
    "posterior distribution over whether it rained."
)
REWRITES["probmods2-conditioning/ex5.c"] = (
    "I have a sprinkler in my garden that turns on each morning at random - half "
    "the time, independently each day. I live in a city where it rains on 30% of "
    "mornings. Lawns are wet whenever the sprinkler turns on, it rains, or both. "
    "Five people in the area - me, Kelsey, Kevin, Manu, and Josh - all have the "
    "same kind of sprinkler (each independent, p=0.5). One morning all five lawns "
    "are wet. Use `mem` so each person's sprinkler is modeled independently. End "
    "your program with `Infer(...)` returning the posterior over whether it rained."
)


# ─── mixture-models ─────────────────────────────────────────────────
ALIEN_DATA = (
    "var data = [\n"
    "  {antennae: false, green: false, blarghNoise: false},\n"
    "  {antennae: true,  green: true,  blarghNoise: true},\n"
    "  {antennae: true,  green: true,  blarghNoise: true},\n"
    "  {antennae: true,  green: true,  blarghNoise: true},\n"
    "  {antennae: false, green: false, blarghNoise: false},\n"
    "  {antennae: true,  green: true,  blarghNoise: true},\n"
    "  {antennae: false, green: false, blarghNoise: false},\n"
    "  {antennae: true,  green: true,  blarghNoise: true},\n"
    "  {antennae: false, green: false, blarghNoise: false},\n"
    "  {antennae: false, green: false, blarghNoise: false}\n"
    "];"
)
REWRITES["probmods2-mixture-models/ex1.a"] = (
    "You visit an alien planet and observe 10 aliens, each with three binary "
    "properties (antennae, green, blarghNoise). Build a mixture model with two "
    "latent kinds of aliens, each with its own per-property probability. Use "
    "`Beta(.5, .5)` priors independently on each of the three probabilities for "
    "each group, and a 50/50 prior over which group each alien belongs to. "
    "Memoize the per-group prototype with `mem` (so within one inference run, "
    "the group's prototype is consistent across aliens).\n\n"
    "Data:\n```\n"
    + ALIEN_DATA +
    "\n```\n\n"
    "End your program with `Infer({method: 'MCMC', kernel: {HMC: {steps: 10, "
    "stepSize: .01}}, samples: 3000}, ...)` returning the joint posterior "
    "`{group1, group2}` where each is the per-property probability object "
    "`{antennae, green, blarghNoise}`."
)
REWRITES["probmods2-mixture-models/ex1.b"] = (
    "Alien-planet setup with 10 aliens (three binary properties: antennae, green, "
    "blarghNoise). Two latent kinds of aliens with Beta(.5, .5) priors per "
    "property, 50/50 prior on group membership, mem'd group prototypes, "
    "MCMC + HMC inference (steps:10, stepSize:.01, samples:3000).\n\n"
    "Data:\n```\n"
    + ALIEN_DATA +
    "\n```\n\n"
    "Extend the model with a new observation: you hear a 'blargh' sound from a "
    "crater but cannot see the alien. Add a latent `mysteryGroup` (50/50 over "
    "the two groups), sample its prototype, and observe `blarghNoise=true` for "
    "that prototype. End your program with the Infer(...) returning "
    "`{group1, group2, mysteryGroup}`."
)


# ─── occams-razor ────────────────────────────────────────────────────
NUMGAME_HELPERS_INLINE = (
    "var maxNumber = 20;\n"
    "var filterByInRange = function(set) {\n"
    "  var inRange = function(v) { v <= maxNumber && v >= 0 };\n"
    "  return _.uniq(filter(inRange, set));\n"
    "};\n"
    "var genEvens = function() {\n"
    "  return filter(function(v) { return v % 2 == 0 }, _.range(1, maxNumber));\n"
    "};\n"
    "var genOdds = function() {\n"
    "  return filter(function(v) { return (v + 1) % 2 == 0 }, _.range(1, maxNumber));\n"
    "};\n"
    "var genMultiples = function(base) {\n"
    "  var multiples = map(function(v) { return base * v }, _.range(maxNumber));\n"
    "  return filterByInRange(multiples);\n"
    "};\n"
    "var genPowers = function(base) {\n"
    "  var powers = map(function(v) { return Math.pow(base, v) }, _.range(maxNumber));\n"
    "  return filterByInRange(powers);\n"
    "};\n"
    "var inSet = function(val, set) { return _.includes(set, val); };\n"
    "var makeRuleHypothesisSpace = function() {\n"
    "  var multipleRules = map(function(b) { return 'multiples_of_' + b }, _.range(1, 12));\n"
    "  var powerRules    = map(function(b) { return 'powers_of_' + b },    _.range(1, 12));\n"
    "  return multipleRules.concat(powerRules).concat(['evens', 'odds']);\n"
    "};"
)
REWRITES["probmods2-occams-razor/ex1.2"] = (
    "Number-game model. The hypothesis space contains rule-based concepts "
    "(`multiples_of_N`, `powers_of_N` for N=1..11; `evens`; `odds`) over the "
    "integers in [1, 20]. Implement similarity-based interval hypotheses "
    "(integers in [a, b]) so that:\n"
    "  - `genSetFromInterval(a, b)` returns all integers from a through b inclusive,\n"
    "  - `makeIntervalHypothesisSpace(start, end)` returns names "
    "`'interval_a_b'` for every (a, b) with start <= a < b <= end,\n"
    "  - `getSetFromHypothesis(rule)` dispatches on the rule's prefix to "
    "produce the set of integers it denotes (handling multiples_, powers_, "
    "evens, odds, and interval_).\n\n"
    "These helpers are given (include them or equivalent in your program):\n"
    "```\n" + NUMGAME_HELPERS_INLINE + "\n```\n\n"
    "Build `learnConcept(examples, testQuery)` that mixes rule and interval "
    "hypotheses 50/50 in the prior, draws a hypothesis, observes each example "
    "via `Categorical({vs: set})`, and returns "
    "`{hypothesis, testQueryResponse: inSet(testQuery, set)}`. End your program "
    "with `learnConcept([3, 10], 12)` returning the joint posterior."
)
REWRITES["probmods2-occams-razor/ex1.3"] = (
    "Using the number-game model with rule + interval hypotheses (50/50 mix of "
    "rule-based and `interval_a_b` hypotheses) over integers [1, 20]: compute "
    "for each query in [1, 20] the expected probability that the query is in "
    "the inferred concept, given examples = [3, 6, 9].\n\n"
    "Helpers (include them in your program):\n"
    "```\n" + NUMGAME_HELPERS_INLINE + "\n"
    "var genSetFromInterval = function(a, b) { return _.range(a, b+1); };\n"
    "var makeIntervalHypothesisSpace = function(start, end) {\n"
    "  var allIntervals = _.flatten(map(function(s) {\n"
    "    return map(function(e) { [s, e] }, genSetFromInterval(s+1, end));\n"
    "  }, genSetFromInterval(start, end)));\n"
    "  return map(function(x) { 'interval_' + x[0] + '_' + x[1] }, allIntervals);\n"
    "};\n"
    "var getSetFromHypothesis = function(rule) {\n"
    "  var parts = rule.split('_');\n"
    "  return parts[0] == 'multiples' ? genMultiples(_.parseInt(parts[2])) :\n"
    "         parts[0] == 'powers'    ? genPowers(_.parseInt(parts[2])) :\n"
    "         parts[0] == 'evens'     ? genEvens() :\n"
    "         parts[0] == 'odds'      ? genOdds() :\n"
    "         parts[0] == 'interval'  ? genSetFromInterval(_.parseInt(parts[1]), _.parseInt(parts[2])) :\n"
    "         null;\n"
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
    "    return {hypothesis: hypothesis, testQueryResponse: inSet(testQuery, set)};\n"
    "  });\n"
    "};\n"
    "```\n\n"
    "End your program with the array `pQueries` of length 20: for each query "
    "1..20, compute `expectation(marginalize(learnConcept([3, 6, 9], query), "
    "function(x) { x.testQueryResponse }))`."
)


# ─── social-cognition (Monty Hall variants) ──────────────────────────
MONTY_HELPERS = (
    "var removeBadItems = function(l, badItems) {\n"
    "  return reduce(function(badItem, remainingL) {\n"
    "    return remove(badItem, remainingL);\n"
    "  }, l, badItems);\n"
    "};\n"
    "var doors = [1, 2, 3];"
)
def _monty_prompt(monty_descr: str, monty_func_name: str) -> str:
    return (
        "Monty Hall variant. Alice picks a door uniformly from {1, 2, 3} (one "
        "hides a prize, the others are empty). Monty opens a different door, "
        "and we condition on his door being neither Alice's nor the prize. "
        f"In this variant, {monty_descr}\n\n"
        "Helpers (include them in your program):\n"
        "```\n" + MONTY_HELPERS + "\n```\n\n"
        f"Build `{monty_func_name}(aliceDoor, prizeDoor)` returning a Distribution "
        "over Monty's door under this variant. Build `model(switches)` that draws "
        f"`aliceDoor` and `prizeDoor` uniformly, samples `montyDoor` from `{monty_func_name}`, "
        "conditions on `montyDoor != prizeDoor && montyDoor != aliceDoor`, and "
        "returns `aliceDoor == prizeDoor` (whether Alice wins) under her strategy "
        "(`switches` true means switch to the remaining unopened door, false means "
        "stay). End your program with an object literal:\n"
        "  `stay`: P(win) when Alice doesn't switch\n"
        "  `switch`: P(win) when Alice switches\n"
        "Each value should be `Infer({method: 'enumerate'}, function() { return model(<flag>); })`."
    )

REWRITES["probmods2-social-cognition/ex2.1"] = _monty_prompt(
    "Monty picks a door uniformly at random from {1, 2, 3}, including possibly "
    "Alice's or the prize door (we filter those via the outer `condition`).",
    "montyRandom",
)
REWRITES["probmods2-social-cognition/ex2.2"] = _monty_prompt(
    "Monty deliberately picks a door that is neither Alice's nor the prize "
    "(this is the original Monty Hall setup). Inside `montyAvoidBoth`, sample "
    "Monty's door uniformly and condition that it is not Alice's and not the prize.",
    "montyAvoidBoth",
)
REWRITES["probmods2-social-cognition/ex2.4"] = _monty_prompt(
    "Monty picks a door uniformly at random but only avoids Alice's door (he "
    "may inadvertently reveal the prize). Inside `montyAvoidAlice`, sample "
    "Monty's door uniformly and condition only that it is not Alice's.",
    "montyAvoidAlice",
)
REWRITES["probmods2-social-cognition/ex2.5"] = _monty_prompt(
    "Monty picks a door uniformly at random but only avoids the prize door "
    "(he may inadvertently pick Alice's). Inside `montyAvoidPrize`, sample "
    "Monty's door uniformly and condition only that it is not the prize.",
    "montyAvoidPrize",
)


# ─── hierarchical-models ────────────────────────────────────────────
ROTTEN_BARREL_BASE = (
    "var makeBarrel = mem(function(barrelName) {\n"
    "  var pRotten = beta({a: .1, b: .2});\n"
    "  var barrel = function(n) {\n"
    "    return repeat(n, function() { flip(pRotten); });\n"
    "  };\n"
    "  return barrel;\n"
    "});"
)
REWRITES["probmods2-hierarchical-models/ex2.2"] = (
    "Apples in a barrel: each apple is rotten with probability `pRotten`, and "
    "`pRotten` is itself drawn from a Beta distribution. Different *stores* "
    "tend to be either mostly-rotten or mostly-fresh, captured by drawing the "
    "Beta hyperparameters from a discrete prior.\n\n"
    "Implement `makeStore(storeName)` that returns a `makeBarrel(barrelName)` "
    "function. Each store's Beta hyperparameters come from a 50/50 mix: "
    "`{a: .1, b: .3}` (mostly fresh) vs `{a: .3, b: .1}` (mostly rotten). "
    "Within a store, all `makeBarrel(...)` calls share that store's Beta. "
    "Memoize at both store and barrel level. The barrel function takes `n` "
    "and returns an array of `n` booleans (rotten or not).\n\n"
    "End your program with an object literal `{sameStore, differentStore}`, "
    "each a forward-sample distribution over the absolute difference in "
    "rotten counts between two barrels of size 10:\n"
    "  - `sameStore`: both barrels from the same store\n"
    "  - `differentStore`: barrels from two different stores\n"
    "Use `Infer({method: 'forward', samples: 10000}, ...)` for each."
)
REWRITES["probmods2-hierarchical-models/ex2.3"] = (
    "Three-level rotten-apple hierarchy: cities → stores → barrels. Each city "
    "has a `cityPrior = beta({a: .25, b: .25})` controlling the probability "
    "that a store in that city is the 'fresh' kind (Beta(.1, .3) for pRotten) "
    "vs the 'rotten' kind (Beta(.3, .1)). Within a store, each barrel draws "
    "`pRotten = beta(storePrior)`, and `barrel(n)` returns an array of n "
    "booleans (rotten or not).\n\n"
    "Implement `makeCity(cityName)` returning a `makeStore` function returning "
    "a `makeBarrel` function as described. Memoize at all three levels.\n\n"
    "For city C1, store S1, barrel B1, end your program with the Infer over "
    "`Math.sum(B1(20))` using forward sampling."
)
REWRITES["probmods2-hierarchical-models/ex2.4"] = (
    "Three-level rotten-apple hierarchy (cities → stores → barrels), as in "
    "the previous exercise: city has `cityPrior = beta({a: .25, b: .25})`; "
    "stores within a city pick Beta(.1, .3) (fresh) with prob `cityPrior`, "
    "else Beta(.3, .1) (rotten); barrels in a store sample `pRotten = "
    "beta(storePrior)`, and `barrel(n)` returns n booleans. Memoized at all "
    "levels.\n\n"
    "You visit a store in a city and observe a barrel of 10 apples, 7 of "
    "which are rotten. You then visit a *different* store in the *same* "
    "city. End your program with `Infer({method: 'MCMC', samples: 5000, lag: "
    "100}, ...)` returning the posterior over the number of rotten apples in "
    "a 10-apple barrel from this second store."
)
REWRITES["probmods2-hierarchical-models/ex3.2"] = (
    "Hierarchical Bayesian data analysis on word reading times. Each data "
    "point is `{group: 'vowel'|'consonant', word: str, id: int, rt: number}`. "
    "Group means are drawn from `Gaussian(200, 100)`; each word has its own "
    "mean `mem(function(word, group) { gaussian(groupMeans[group], 20) })`; "
    "individual reading times are `Gaussian({mu: wordMean(d.word, d.group), "
    "sigma: 10})`.\n\n"
    "Extend this model with a per-participant random effect: a Gaussian(0, 2) "
    "additive offset for each participant id. Each `rt` is now `Gaussian({mu: "
    "wordMean(d.word, d.group) + participantMean(d.id), sigma: 10})`.\n\n"
    "Data:\n```\n"
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
    "            {group: \"consonant\", word: \"fedora\", id: 3, rt: 228}];\n"
    "```\n\n"
    "Use `Infer({method: 'MCMC', burn: 10000, lag: 5, samples: 5000}, ...)`. "
    "End your program with the joint posterior `{diff, p1, p2, p3}` where "
    "`diff = groupMeans['vowel'] - groupMeans['consonant']` and `p1, p2, p3` "
    "are `participantMean(1), participantMean(2), participantMean(3)`."
)


# ─── observing-sequences ────────────────────────────────────────────
BIGRAM_INTRO = (
    "Bigram sentence model over vocabulary `['dogs', 'cats', 'chase', 'sleep', "
    "'stop']`. Each word has its own transition distribution drawn from a "
    "`dirichletDrift({alpha: ones([5,1]), concentration: 10})`. A sentence is "
    "generated by starting at the special token `'start'`, repeatedly applying "
    "`transition(prevWord)` to draw the next word, terminating when `'stop'` "
    "is drawn (and emitting `'stop'` so that the observed sentence's length "
    "matches without `undefined`).\n\n"
    "Helper:\n"
    "```\n"
    "var comparray = function(arr1, arr2) {\n"
    "  return JSON.stringify(arr1) === JSON.stringify(arr2);\n"
    "};\n"
    "```\n\n"
    "Use `Infer({method: 'MCMC', burn: 10000, samples: 50000, onlyMAP: false}, ...)`. "
)
REWRITES["probmods2-observing-sequences/ex1.b"] = BIGRAM_INTRO + (
    "Observe the sentence `['dogs', 'chase', 'cats', 'stop']`. Then, in a "
    "second sentence, the first word is `'dogs'`. End your program with the "
    "Infer(...) returning the marginal distribution over the second word of "
    "this new sentence."
)
REWRITES["probmods2-observing-sequences/ex1.c"] = BIGRAM_INTRO + (
    "Observe the sentence `['dogs', 'chase', 'cats', 'stop']`. Then, in a "
    "second sentence, the second word is `'chase'`. End your program with the "
    "Infer(...) returning the marginal distribution over the FIRST word of "
    "this new sentence."
)
REWRITES["probmods2-observing-sequences/ex2.a"] = BIGRAM_INTRO + (
    "Observe the sentence `['dogs', 'chase', 'cats', 'stop']`. Then, in a "
    "second sentence, the first word is `'cats'`. End your program with the "
    "Infer(...) returning the marginal distribution over the second word of "
    "this new sentence."
)
HMM_INTRO = (
    "Hidden Markov sentence model. Words have parts of speech: N for nouns "
    "{'dogs', 'cats'}; V for verbs {'chase', 'sleep'}; plus `'stop'`. Each "
    "POS has its own transition distribution drawn from "
    "`dirichletDrift({alpha: ones([3,1]), concentration: 10})`, memoized. A "
    "sentence is generated by starting at `'start'` POS, transitioning to "
    "successive POS tags, drawing a word given each POS via `drawWord(pos)` "
    "(which returns `uniformDraw([...])` for N or V, else `'stop'`), and "
    "appending `'stop'` when reached.\n\n"
    "Helper:\n"
    "```\n"
    "var comparray = function(arr1, arr2) {\n"
    "  return JSON.stringify(arr1) === JSON.stringify(arr2);\n"
    "};\n"
    "```\n\n"
)
REWRITES["probmods2-observing-sequences/ex2.d"] = HMM_INTRO + (
    "Observe the sentence `['dogs', 'chase', 'cats', 'stop']`. Then, in a "
    "second sentence whose first word is `'cats'`, end your program with the "
    "Infer(...) (MCMC, burn: 10000, samples: 1000, lag: 10, onlyMAP: false) "
    "returning the marginal distribution over the second word."
)
REWRITES["probmods2-observing-sequences/ex3.a"] = (
    "Hidden Markov sentence model with extended vocabulary. POS tags: N for "
    "nouns {'dog', 'cat'}, V for verbs {'chases', 'sleeps'}, D for determiners "
    "{'the', 'a'}, A for adverbs ({'dilligently'}), plus `'stop'`. Per-POS "
    "transitions drawn from `dirichletDrift({alpha: ones([5,1]), concentration: "
    "10})`, memoized. `drawWord(pos)` returns `uniformDraw(...)` for the "
    "respective list, or `'dilligently'` for A, or `'stop'` for stop.\n\n"
    "Helper:\n"
    "```\n"
    "var comparray = function(arr1, arr2) {\n"
    "  return JSON.stringify(arr1) === JSON.stringify(arr2);\n"
    "};\n"
    "```\n\n"
    "Use `factor(comparray(['the', 'dog', 'chases', 'a', 'cat', 'stop'], "
    "generateSentence('start')) * 5)` to softly condition on the observed "
    "sentence. Then sample five new sentences from `generateSentence('start')`. "
    "End your program with `Infer({method: 'MCMC', burn: 10000, samples: 1000, "
    "lag: 10, onlyMAP: true}, ...)` returning a record `{sent1, sent2, sent3, "
    "sent4, sent5}`."
)


# ─── agents-as-programs ─────────────────────────────────────────────
REWRITES["probmods2-agents-as-programs/ex2.e"] = (
    "Ultimatum game with uncertain alpha. Responder accepts with probability "
    "`Math.pow(offer/10, alpha)`; the proposer doesn't know alpha but believes "
    "it is uniform on [0.5, 5].\n\n"
    "Setup: in round 1, the proposer offered $2 and the responder rejected. "
    "In round 2, what should the proposer offer to maximize expected payoff?\n\n"
    "Two-stage inference:\n"
    "  1. `proposer1`: Infer (MCMC, 50000 samples) the posterior over alpha "
    "given that round 1's offer of $2 was rejected.\n"
    "  2. End your program with an outer Infer (forward, 1000 samples) that "
    "samples an `alpha2` from `proposer1`, then runs an inner Infer (MCMC, "
    "5000 samples) over offers 0..10 with `factor(reward2)` where "
    "`reward2 = responder(offer2, alpha2) ? (10 - offer2) : 0`. Sample one "
    "round-2 offer from that inner posterior and return it. The outer Infer "
    "is the distribution over the chosen round-2 offer."
)
REWRITES["probmods2-agents-as-programs/ex4.b"] = (
    "Frank & Goodman pragmatic listener / pragmatic speaker / RSA model. The "
    "world has three objects: {blue square, blue circle, green square}, drawn "
    "uniformly. Possible utterances: 'blue', 'green', 'square', 'circle'. "
    "Truth function: a color/shape utterance must match the corresponding "
    "attribute, otherwise the utterance is true.\n\n"
    "Build the level-1 stack with alpha = 1:\n"
    "  - `literalListener(utterance)`: Infer over `meaningPrior` conditioning "
    "on `meaning(utterance, obj)` being true.\n"
    "  - `speaker(obj)`: Infer over utterances drawn uniformly with "
    "`factor(alpha * literalListener(utterance).score(obj))`.\n"
    "  - `pragmaticListener(utterance)` = L1: Infer over `meaningPrior` "
    "conditioning on `observe(speaker(obj), utterance)`.\n\n"
    "Build the level-2 stack with alpha = 1:\n"
    "  - `speaker2(obj)`: Infer over utterances drawn uniformly with "
    "`factor(alpha * pragmaticListener(utterance).score(obj))`.\n"
    "  - `listener3(utterance)` = L2: Infer over `meaningPrior` conditioning "
    "on `observe(speaker2(obj), utterance)`.\n\n"
    "End your program with an object literal `{L1, L2}` where `L1 = "
    "pragmaticListener('blue')` and `L2 = listener3('blue')`."
)


# ─── inference-algorithms ───────────────────────────────────────────
HEART_HELPERS_INLINE = (
    "var onCurve = function(x, y) {\n"
    "  var x2 = x*x;\n"
    "  var term1 = y - Math.pow(x2, 1/3);\n"
    "  var crossSection = x2 + term1*term1 - 1;\n"
    "  return Math.abs(crossSection) < 0.01;\n"
    "};\n"
    "var xbounds = [-1, 1];\n"
    "var ybounds = [-1, 1.6];\n"
    "var xmu = 0.5 * (xbounds[0] + xbounds[1]);\n"
    "var ymu = 0.5 * (ybounds[0] + ybounds[1]);\n"
    "var xsigma = 0.5 * (xbounds[1] - xbounds[0]);\n"
    "var ysigma = 0.5 * (ybounds[1] - ybounds[0]);"
)
HEART_INTRO = (
    "Heart-shaped implicit curve: a point (x, y) is *on the curve* if "
    "`x^2 + (y - x^(2/3))^2 - 1` is within 0.01 of 0. The reference model "
    "draws x and y from independent Gaussians around the bounding box "
    "center, and conditions on `onCurve(x, y)`.\n\n"
    "Helpers (include them or equivalent in your program):\n"
    "```\n" + HEART_HELPERS_INLINE + "\n```\n\n"
)
REWRITES["probmods2-inference-algorithms/ex1.1"] = HEART_INTRO + (
    "Use Metropolis-Hastings MCMC instead of rejection sampling on the "
    "independent-Gaussians model `var x = gaussian(xmu, xsigma); var y = "
    "gaussian(ymu, ysigma); condition(onCurve(x, y))`. End your program with "
    "`Infer({method: 'MCMC', samples: 10000, lag: 10}, model)` returning the "
    "posterior over `{x, y}`."
)
REWRITES["probmods2-inference-algorithms/ex1.2"] = HEART_INTRO + (
    "Change the model to draw x and y *jointly* via a `diagCovGaussian` "
    "centered at `(xmu, ymu)` with diagonal covariance `(xsigma, ysigma)`, "
    "so MH MCMC successfully traces the curve. Use `T.get` to extract the "
    "x and y components of the sample. End your program with `Infer({method: "
    "'MCMC', samples: 1000, lag: 100}, model)` returning the posterior over "
    "`{x, y}`."
)
REWRITES["probmods2-inference-algorithms/ex1.3"] = HEART_INTRO + (
    "Using the original independent-Gaussians model, use HMC instead of MH. "
    "End your program with `Infer({method: 'MCMC', samples: 10000, kernel: "
    "{HMC: {steps: 10, stepSize: .5}}}, model)` returning the posterior over "
    "`{x, y}`."
)
INTERP_INTRO = (
    "Two-endpoint interpolation: `point1 = -10` is fixed; `point2` is uniform "
    "on [-100, 100]; `interpolationWeight` is uniform on [0, 1]; "
    "`pointInMiddle = point1 * interpolationWeight + point2 * (1 - "
    "interpolationWeight)`. We `observe(Gaussian({mu: 0, sigma: 0.1}), "
    "pointInMiddle)`.\n\n"
    "Helper:\n"
    "```\n"
    "var interpolate = function(point1, point2, interpolationWeight) {\n"
    "  return point1 * interpolationWeight + point2 * (1 - interpolationWeight);\n"
    "};\n"
    "```\n\n"
)
REWRITES["probmods2-inference-algorithms/ex2.1"] = INTERP_INTRO + (
    "Run MCMC with `samples: 5000, lag: 100`. End your program with an object "
    "literal `{point2, interpolationWeight}` where each value is the marginal "
    "distribution of the corresponding latent (use `marginalize`)."
)
REWRITES["probmods2-inference-algorithms/ex2.2"] = INTERP_INTRO + (
    "Run MCMC with `samples: 5000, lag: 100`. End your program with the joint "
    "marginal distribution over `(point2, interpolationWeight)` as a Distribution "
    "(use `marginalize` returning an object `{point2, inter}` for each sample)."
)
REWRITES["probmods2-inference-algorithms/ex2.3"] = INTERP_INTRO + (
    "Run MCMC with `samples: 100, lag: 0`. From `posterior.samples`, extract "
    "the array of `pointInMiddle` values in order. End your program with that "
    "array of length 100."
)
REWRITES["probmods2-inference-algorithms/ex2.4"] = INTERP_INTRO + (
    "Rewrite this as rejection sampling. Convert the `observe` into a "
    "`condition(Math.abs(pointInMiddle) < 0.01)`. End your program with "
    "`Infer({method: 'rejection', samples: 1000}, model)` returning the "
    "posterior over `{point2, interpolationWeight, pointInMiddle}`."
)
REWRITES["probmods2-inference-algorithms/ex2.5"] = INTERP_INTRO + (
    "Replace `point2`'s prior with a drift kernel: `uniformDrift({a: -100, "
    "b: 100, width: 0.1})`. End your program with `Infer({method: 'MCMC', "
    "samples: 500}, model)` returning the posterior over "
    "`{point2, interpolationWeight, pointInMiddle}`."
)


def main():
    in_path = Path("data/atomized.jsonl")
    out_path = Path("data/atomized_v2.jsonl")

    atoms = load_jsonl(in_path)
    out_atoms = []
    n_rewritten = 0
    n_dropped = 0
    for a in atoms:
        if a["id"] in DROPPED:
            n_dropped += 1
            continue
        if a["id"] in REWRITES:
            a = {**a, "prompt": REWRITES[a["id"]]}
            n_rewritten += 1
        out_atoms.append(a)

    write_jsonl(out_path, out_atoms)
    print(f"v1: {len(atoms)} atoms -> v2: {len(out_atoms)} atoms")
    print(f"  dropped: {n_dropped}  (process-models, wall-clock-broken)")
    print(f"  rewritten prompts: {n_rewritten}")
    print(f"  unchanged: {len(atoms) - n_dropped - n_rewritten}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

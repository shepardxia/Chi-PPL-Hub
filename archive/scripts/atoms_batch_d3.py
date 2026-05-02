"""Batch D3: generative-models (10 atoms)."""
from atom_writer import write_atoms

ATOMS = []

ATOMS.append({
    "id": "probmods2-generative-models/ex1.b",
    "source": "exercises/generative-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "record",
    "answer_shape": {"record": {
        "p1": "samples",
        "p2": "samples",
        "p3": "samples",
    }},
    "prompt": (
        "Verify by sampling 1000 times each that the following three programs have the "
        "same marginal distribution:\n\n"
        "Program 1: `flip() ? flip(.7) : flip(.1)`\n"
        "Program 2: `flip(flip() ? .7 : .1)`\n"
        "Program 3: `flip(.4)`\n\n"
        "Return an object literal `{p1, p2, p3}` where each value is an array of 1000 "
        "samples from the corresponding program (use `repeat(1000, function() {...})`)."
    ),
    "groundtruth_code": (
        "({\n"
        "  p1: repeat(1000, function() { return flip() ? flip(.7) : flip(.1); }),\n"
        "  p2: repeat(1000, function() { return flip(flip() ? .7 : .1); }),\n"
        "  p3: repeat(1000, function() { return flip(.4); })\n"
        "})\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex1.c",
    "source": "exercises/generative-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "samples",
    "answer_shape": "samples",
    "prompt": (
        "Write a *new* WebPPL expression with the same marginal distribution as "
        "`flip(.4)` (i.e., true with probability 0.4) that looks structurally different "
        "from `flip(.4)`, `flip() ? flip(.7) : flip(.1)`, and `flip(flip() ? .7 : .1)`. "
        "End with that expression - the harness will rerun your program multiple times "
        "to estimate its marginal."
    ),
    "groundtruth_code": (
        "flip() ? false : flip(.8);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex2.b",
    "source": "exercises/generative-models.md",
    "task_type": "modify_given",
    "eval_mode": "samples",
    "answer_shape": "samples",
    "prompt": (
        "Given:\n"
        "```\n"
        "var foo = function() { return flip() }\n"
        "[foo(), foo(), foo()]\n"
        "```\n\n"
        "Each call to `foo()` independently flips, so the list can have any combination "
        "of trues and falses. Modify the program using `mem` so that `[foo(), foo(), "
        "foo()]` is always either `[true, true, true]` or `[false, false, false]`. End "
        "with the list expression - the harness will rerun your program multiple times."
    ),
    "groundtruth_code": (
        "var foo = mem(function() { return flip(); });\n"
        "[foo(), foo(), foo()];\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex2.c",
    "source": "exercises/generative-models.md",
    "task_type": "modify_given",
    "eval_mode": "samples",
    "answer_shape": "samples",
    "prompt": (
        "Given the memoized program:\n"
        "```\n"
        "var foo = mem(function() { return flip() })\n"
        "[foo(), foo(), foo()]\n"
        "```\n\n"
        "Modify it so that the first two elements are always equal but the third can "
        "differ. Hint: pass an argument to `foo` that distinguishes the two calls you "
        "want to be the same. End with the list expression."
    ),
    "groundtruth_code": (
        "var foo = mem(function(x) { return flip(); });\n"
        "[foo(0), foo(0), foo(1)];\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex4.b",
    "source": "exercises/generative-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given a simple symptoms model:\n"
        "```\n"
        "var allergies = flip(0.3)\n"
        "var cold = flip(0.2)\n"
        "var sneeze = cold || allergies\n"
        "var fever = cold\n"
        "```\n\n"
        "End with `Infer({method: 'forward', samples: 1000}, ...)` returning the joint "
        "distribution over `{sneeze, fever}`."
    ),
    "groundtruth_code": (
        "Infer({method: \"forward\", samples: 1000}, function() {\n"
        "  var allergies = flip(0.3);\n"
        "  var cold = flip(0.2);\n\n"
        "  var sneeze = cold || allergies;\n"
        "  var fever = cold;\n\n"
        "  return {\"sneeze\": sneeze, \"fever\": fever};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex4.c",
    "source": "exercises/generative-models.md",
    "task_type": "modify_given",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given a multi-patient symptoms model attempted via per-person functions:\n"
        "```\n"
        "var allergies = function(person) { return flip(.3) }\n"
        "var cold = function(person) { return flip(.2) }\n"
        "var sneeze = function(person) { return cold(person) || allergies(person) }\n"
        "[sneeze('bob'), sneeze('alice')]\n"
        "```\n\n"
        "Add `fever`, then end with `Infer({method: 'forward', samples: 1000}, ...)` "
        "returning the joint distribution over Bob's symptoms `{sneeze, fever}`. The "
        "raw program double-counts: `sneeze('bob')` flips a fresh cold, and so does "
        "`fever('bob')`. Fix this by using `mem` on `cold` and `allergies` so the same "
        "person has consistent disease state within a sample."
    ),
    "groundtruth_code": (
        "Infer({method: \"forward\", samples: 1000}, function() {\n"
        "  var allergies = mem(function(person) { return flip(.3); });\n"
        "  var cold = mem(function(person) { return flip(.2); });\n\n"
        "  var sneeze = function(person) { return cold(person) || allergies(person); };\n"
        "  var fever = function(person) { return cold(person); };\n\n"
        "  return {\"sneeze\": sneeze('bob'), \"fever\": fever('bob')};\n"
        "});\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex5.b",
    "source": "exercises/generative-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Given:\n"
        "```\n"
        "var makeCoin = function(weight) {\n"
        "  return function() { return flip(weight) ? 'h' : 't' }\n"
        "}\n"
        "var bend = function(coin) {\n"
        "  return function() {\n"
        "    return coin() == 'h' ? makeCoin(.7)() : makeCoin(.1)()\n"
        "  }\n"
        "}\n"
        "var fairCoin = makeCoin(.5)\n"
        "var bentCoin = bend(fairCoin)\n"
        "```\n\n"
        "End with `Infer({method: 'forward', samples: 10000}, bentCoin)` returning "
        "the bent coin's distribution."
    ),
    "groundtruth_code": (
        "var makeCoin = function(weight) {\n"
        "  return function() {\n"
        "    return flip(weight) ? 'h' : 't';\n"
        "  };\n"
        "};\n"
        "var bend = function(coin) {\n"
        "  return function() {\n"
        "    return coin() == 'h' ? makeCoin(.7)() : makeCoin(.1)();\n"
        "  };\n"
        "};\n\n"
        "var fairCoin = makeCoin(.5);\n"
        "var bentCoin = bend(fairCoin);\n\n"
        "Infer({method: 'forward', samples: 10000}, bentCoin);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex6.b",
    "source": "exercises/generative-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "The recursive geometric distribution is defined as:\n"
        "```\n"
        "var geometric = function() {\n"
        "  return flip() ? 0 : 1 + geometric()\n"
        "}\n"
        "```\n\n"
        "End with `Infer({method: 'forward', samples: 10000}, geometric)` returning "
        "its distribution."
    ),
    "groundtruth_code": (
        "var geometric = function() {\n"
        "  return flip() ? 0 : 1 + geometric();\n"
        "};\n"
        "Infer({method: \"forward\", samples:10000}, geometric);\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex7.a",
    "source": "exercises/generative-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "samples",
    "answer_shape": "samples",
    "prompt": (
        "Convert this joint probability table into a compact WebPPL program:\n\n"
        "| A | B | P(A,B) |\n"
        "|---|---|--------|\n"
        "| F | F | 0.14   |\n"
        "| F | T | 0.06   |\n"
        "| T | F | 0.4    |\n"
        "| T | T | 0.4    |\n\n"
        "Requirement: fix P(A) first, then define the probability of B as a function of "
        "A. Use `flip(...)` for both. End with the expression `[a, b]` - the harness "
        "will rerun your program multiple times."
    ),
    "groundtruth_code": (
        "var a = flip(0.8);\n"
        "var b = flip(a ? 0.5 : 0.3);\n"
        "[a, b];\n"
    ),
})

ATOMS.append({
    "id": "probmods2-generative-models/ex7.b",
    "source": "exercises/generative-models.md",
    "task_type": "write_from_scratch",
    "eval_mode": "distribution",
    "answer_shape": "distribution",
    "prompt": (
        "Wrap the joint probability program for (A, B) - where P(A=T)=0.8 and P(B=T|A) "
        "is 0.5 if A=T else 0.3 - in `Infer({method: 'forward', samples: 10000}, ...)` "
        "to verify the joint distribution. End with the Infer(...) returning [a, b]."
    ),
    "groundtruth_code": (
        "Infer({method: \"forward\", samples: 10000}, function() {\n"
        "  var a = flip(0.8);\n"
        "  var b = flip(a ? 0.5 : 0.3);\n"
        "  return [a, b];\n"
        "});\n"
    ),
})


if __name__ == "__main__":
    write_atoms(ATOMS, append=True)

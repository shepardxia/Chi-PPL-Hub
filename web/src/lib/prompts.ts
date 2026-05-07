// Mirrors eval/prompt.py — kept verbatim so the site shows exactly what
// the LM saw. If eval/prompt.py changes, update both halves.

export const PROMPT_VERSION = 'v2-atom';

export const SYSTEM_PROMPT_BASE = `You are a WebPPL code generator. Given an exercise, produce a single WebPPL program that binds the answer to a top-level variable named \`ANSWER\`.

Answer format (strict): emit exactly one fenced code block.

\`\`\`js
<your WebPPL program ending with: var ANSWER = <expression>;>
\`\`\`

The last statement of your program MUST be \`var ANSWER = <expression>;\` where \`<expression>\` is the answer the prompt asks for - typically an \`Infer({...}, model)\` for a distribution, a numeric/array value, or an object literal \`{key: value, ...}\` for a record of multiple sub-answers.

Do not write prose, explanations, or multiple code blocks. Do not use \`return\` at the top level - WebPPL doesn't allow it (return is only for function bodies).`;

export const WEBPPL_PRIMER = `WebPPL is a probabilistic programming language with JavaScript-like syntax. A few quirks to remember:

Control flow: WebPPL is a functional subset of JavaScript. There are no \`for\` or \`while\` loops. Iterate with \`map(fn, list)\`, \`mapData({data: list}, fn)\`, \`repeat(n, fn)\`, or recursion. \`_.range(start, end)\` produces integer ranges.

Functions: define with \`var f = function(args) { ...; return value; }\`. The last expression of a function is auto-returned only if it is a bare expression statement; otherwise use explicit \`return\`. Memoize with \`mem(fn)\` so repeated calls with the same arguments return the same value within an inference run.

Random primitives - lowercase samples directly, uppercase constructs a Distribution object. They are NOT interchangeable:

  flip(p)            -> boolean             (sample, no \`sample()\` needed)
  uniform(a, b)      -> number              (sample)
  gaussian(mu, sg)   -> number              (sample)
  beta(a, b)         -> number              (sample)
  dirichlet(alpha)   -> tensor              (sample; alpha must be a Vector)
  randomInteger(n)   -> int 0..n-1          (sample)
  uniformDrift({a, b, width})        -> sample  (drift kernel; do NOT wrap in sample())
  dirichletDrift({alpha, conc.})     -> sample  (drift kernel; do NOT wrap in sample())

Distribution *constructors* (used with \`sample(D)\`, \`observe(D, val)\`, or as return value of inference):
  Bernoulli({p})              Beta({a, b})              Gaussian({mu, sigma})
  Uniform({a, b})             Categorical({vs, ps})     Binomial({p, n})
  Dirichlet({alpha})          Multinomial({ps, n})      Poisson({mu})

Common gotchas:
- \`Dirichlet({alpha: ...})\` requires a Vector, not a JS array. Use \`ones([n, 1])\` or \`Vector([1, 1, ...])\`.
- WebPPL only supports \`var\`, not \`let\` / \`const\`.
- WebPPL is *single-assignment*: \`var X = ...;\` only. You can't declare \`var X;\` and assign later, and you can't reassign \`X = ...\` after the declaration. Use ternaries or recursion to express conditional bindings.
- Array methods like \`.fill\`, \`.indexOf\`, \`.map\`, \`.forEach\`, \`.concat\` may fail. Prefer \`repeat(n, fn)\`, \`_.indexOf(arr, x)\`, \`map(fn, arr)\`, \`mapData({data: arr}, fn)\`, and \`arr1.concat(arr2)\` only at the top of a returned expression.
- Always end top-level statements with \`;\`. WebPPL inherits JS ASI rules, so \`var x = f()\\n[a, b]\` parses as \`var x = f()[a, b]\` (subscript), not two statements.

Inference: \`Infer({method: ..., samples: N, ...}, modelFn)\` runs \`modelFn\` under the chosen method and returns a Distribution over its return value. Methods: \`'enumerate'\`, \`'rejection'\`, \`'forward'\`, \`'MCMC'\`, \`'SMC'\`. For MCMC, optional kernels: \`kernel: {HMC: {steps, stepSize}}\`; the drift kernels above can replace \`uniform\`/\`dirichlet\` calls inside the model.

Conditioning: \`condition(bool)\` zeros out worlds where bool is false. \`observe(dist, value)\` factors in \`dist.score(value)\`. \`factor(score)\` adds \`score\` to the log-probability directly.

Tensors / utilities: \`Vector([a, b, ...])\`, \`T.get(vec, i)\`, \`ones([rows, cols])\`, \`_.range\`, \`_.flatten\`, \`_.zipObject\`, \`_.fromPairs\`, \`_.includes\`, \`_.parseInt\`, \`_.uniq\`, \`_.merge\`. Arrays use \`Array.isArray\`, \`Object.keys\` works on plain objects.

Display: there's no \`viz\`, \`print\`, \`display\`, etc. that affects the answer - those are browser-only. The answer is the value of your program's last expression.`;

export function systemPrompt(withPrimer: boolean): string {
  return withPrimer ? `${SYSTEM_PROMPT_BASE}\n\n${WEBPPL_PRIMER}` : SYSTEM_PROMPT_BASE;
}

/** Run names follow `<model>-<primer-flag>-...` convention; "noprimer" → false. */
export function runHasPrimer(runName: string): boolean {
  return !runName.toLowerCase().includes('noprimer');
}

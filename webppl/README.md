# WebPPL

[Official website](http://webppl.org/) | [Documentation](https://webppl.readthedocs.io/en/master/) | [GitHub](https://github.com/orgs/probmods/repositories)

WebPPL (pronounced ‘web people’), is a small probabilistic programming language built on top of a (purely functional) subset of Javascript. 

## Tutorials

- [Probabilistic Models of Cognition (probmods)](http://probmods.org/) — An introduction to computational cognitive science and the probabilistic programming language WebPPL. 
- [DIPPL](http://dippl.org/) — Language design & implementation of PPL with WebPPL with examples.
- [Probabilistic Language Understanding](https://www.problang.org/chapters/app-06-intro-to-webppl.html) - Rational Speech Act framework's WebPPL intro appendix.
- [Modeling Agents with Probabilistic Programs](https://agentmodels.org/) — Modeling Agents with Probabilistic Programs.
- [Gerstenberg & Smith tutorial](https://github.com/tobiasgerstenberg/webppl_tutorial) — Slides and notes on WebPPL basics, generative models, inference.

## Existing PPL Code

- [Forest](https://stuhlmueller.org/) — Repository for generative models in WebPPL
- [probmods/webppl/examples](https://github.com/probmods/webppl/tree/master/examples) - Probmods examples
- [stuhlmueller/neural-nets](https://github.com/stuhlmueller/neural-nets) — Neural nets in WebPPL
- [agentmodels/webppl-agents](https://github.com/agentmodels/webppl-agents) - Library for modeling MDP and POMDP agents in WebPPL

## Known LLM issues

- WebPPL has no assignment expressions or looping constructs (for, while, do), and it’s a purely functional subset of JS. 
- LLMs can hallucinate up functions that don't exist in webppl. Kaya has a webppl library which actually implements these commonly hallucinated functions.

## Publications

- List on [website](http://webppl.org/)
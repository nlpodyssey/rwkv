# RWKV

RWKV (**R**eceptance **W**eighted **K**ey **V**alue) is a RNN with Transformer-level performance without the quadratic attention mechanism: only the hidden state at the current position is needed to calculate the state at the next position.

RWKV is designed to perform inference efficiently, even on CPUs, so it is well-suited to run LLM (Large Language Model) on normal consumer hardware at decent speed.

This implementation is written in Go and utilizes the [Spago](https://github.com/nlpodyssey/spago) machine learning framework.

# How it works

Currently, there are no research papers that describe this neural architecture. The majority of the information can be found in the [original codebase](https://github.com/BlinkDL/RWKV-LM) of RWKV's author, [PENG Bo (BlinkDL on GitHub)](https://github.com/BlinkDL).

Roughly speaking, 

- it uses a method similar to an "exponential moving average" to gather contextual information by alternating `time-mix` and `channel-mix` layers. The layers decay at different rates, which helps the network remember important information for longer periods of time as it processes the input sequence.
- the time-mix is inspired by [Apple's AFT](https://arxiv.org/abs/2105.14103). The channel-mix is inspired by [GeGLU](https://arxiv.org/abs/2002.05202).
- it uses careful parameters initialization to get fast convergence (orthogonal matrices with proper scaling and special time curves).
 
# Installation

Requirements:

* [Go 1.19](https://golang.org/dl/)

Clone this repo or get the library:

```console
go get -u github.com/nlpodyssey/rwkv
```

The library is optimized to run in x86-64 CPUs. If you want to run it on a different architecture, you can use the `GOARCH=amd64` environment variable.

# Roadmap

- [ ] Parameters initialization (**essential**)
- [ ] Unit tests
- [ ] Documentation
- [ ] Gob serialization for large models
- [ ] Model optimization

# Credits

- RWKV is a research project by [PENG Bo](https://github.com/BlinkDL) and this implementation is a Go port of the [original codebase](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo).

# References

```
@software{peng_bo_2021_5196578,
  author       = {PENG Bo},
  title        = {BlinkDL/RWKV-LM: 0.01},
  month        = aug,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.01},
  doi          = {10.5281/zenodo.5196577},
  url          = {https://doi.org/10.5281/zenodo.5196577}
}
```
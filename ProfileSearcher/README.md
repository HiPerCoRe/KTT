# Profile-based searcher

This searcher uses profiling counters along with ML models to more effectively
navigate through the tuning space. See `./ExampleUsage.py` on how
to use the searcher. It's possible to compare it with the results
of the random search with `./GraphResults.py [output-graph]`.

## Installing dependencies

> Before installing it is recommended to set up a virtual enviroment.

Dependencies for the newer GPUs (starting from Turing):
```bash
pip isntall -r requirements/cu13-cupy.txt
pip install -r requirements/common.txt
```


### Older GPUs (Volta and older)

> For these GPUs, the installed NVIDIA driver on linux **must** be
> of version `<= 575xx` for profiling to work. Latest driver supporting
> these architectures (`580xx`) can run CUDA, but cannot collect the
> profiling counters. On modern systems it's next to impossible
> to use the older driver..

Dependencies for Volta and older architectures:
```bash
pip install -r requirements/cu12-cupy.txt
pip install -r requirements/common.txt
```

### Pytorch

Pytorch is *not required* to run the code, buuut if you want to use the neural
models, uncomment code in `modules/model.py` and install the library:

```bash
pip install -r requirements/cu130-torch.txt # new GPUs
pip install -r requirements/cu126-torch.txt # old GPUs (Volta and older)
```

## Batching

Profiling is expensive, so a batching system is implemented.
The searcher takes in a `BatchInfo` object, with 3 parameters:
`batchSize`, `randomSize`, `neighborSize`. `randomSize + neighborSize`
is the amount of configurations scored every profiling pass. For
the profiling to happen this value **must** be larger than `batchSize`.
It can be relatively large, since the scoring process is a lot cheaper
in comparison to profiling itself.

The code does the profiling every `batchSize`
configurations, smaller value introduces lots of profiling overhead,
but makes use of the profiling  more, which might lead to faster 
convergence (lower amount of iterations).

## Models

Generate the model files via the code from [this repo](https://gitlab.fi.muni.cz/xratushn/ktt-model).
Link them in the searcher (it takes a `ModelInfo` object in with the paths to the models).


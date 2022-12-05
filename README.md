# STEAL
Simultaneous Trajectory Estimation and Learning

## Install

```
pip install poetry
poetry install

# For Linux/Windows/Intel Macs
poetry run pip install -U torch gpytorch pytorch-lightning

# For Apple Silicon
poetry run pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
poetry run pip install -U gpytorch pytorch-lightning
```

## Datasets

- [LASA Dataset](https://cs.stanford.edu/people/khansari/download.html)

#### Resources

- [RMPFlow](https://github.com/mrana6/hgrmpflow)
- [Euclideanizing Flow](https://github.com/mrana6/euclideanizing_flows)
- [Alternate Euclideanizing Flow](https://github.com/nash169/learn-diffeomorphism)

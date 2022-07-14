# Structure Representation Generation

<div align="center">

**Atomic Structure Generation from Reconstructing Structural Fingerprints**

</div>

## Requirements

Git clone this repo and in the main directory do

```python
pip install -r requirements.txt
pip install -e .
```

The package has been tested on

- Python 3.6.10 / 3.9.12
- PyTorch 1.10.2 / 1.12.0
- Torch.cuda 11.3

## How to Use

Most configurations of the system/code are done through `.yaml` files under `configs/`. 

To load a configuration:

```python
# load our config yaml file
stream = open('./configs/example/example_extract_reconstruct.yaml')
CONFIG = yaml.safe_load(stream)
stream.close()
# dotdict for dot operations on Python dict 
# e.g., CONFIG.cutoff == CONFIG['cutoff']
CONFIG = dotdict(CONFIG)
```

### Representation Extraction

Extract representations for training generative model using selected descriptor.

See [`examples/example_extract.py`](https://github.com/shuyijia/srg/blob/main/examples/example_extract.py)

### Generative Model

Train a CVAE (conditional variational auto-encoder) as the generative model. 

See [`examples/example_cvae_trainer.py`](https://github.com/shuyijia/srg/blob/main/examples/example_cvae_trainer.py).

### Generation

Generate representation from a given target value using the decoder part of the CVAE. 

See [`tests/test_generator.py`](https://github.com/shuyijia/srg/blob/main/tests/test_generator.py).

### Reconstruction

Generate atomic structures from generated representation.

See [`tests/test_reconstruction.py`](https://github.com/shuyijia/srg/blob/main/tests/test_reconstruction.py).

______________________________________________________________________

## Results

<div align="center">
    <img src="https://images.unsplash.com/photo-1494256997604-768d1f608cac?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1858&q=80" width="65%">
</div>

______________________________________________________________________

## Citation

```bash
@article{XXXX,
  title={Atomic structure generation from reconstructing structural fingerprints},
  author = {XXXX},
  journal={XXXX},
  year={2022}
}
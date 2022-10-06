import yaml, os, unittest, torch
import numpy as np
from structrepgen.extraction.representation_extraction import *
from structrepgen.utils.dotdict import dotdict
from structrepgen.reconstruction.reconstruction import Reconstruction

stream = open('./configs/example/example_extract_reconstruct.yaml')
CONFIG = yaml.safe_load(stream)
stream.close()
CONFIG = dotdict(CONFIG)

constructor = Reconstruction(CONFIG)

constructor.main()
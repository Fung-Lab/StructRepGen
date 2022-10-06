import yaml, os, unittest, torch
import numpy as np
from structrepgen.generators.generator import Generator
from structrepgen.utils.dotdict import dotdict

stream = open('./configs/example/example_generator.yaml')
CONFIG = yaml.safe_load(stream)
stream.close()
CONFIG = dotdict(CONFIG)

gen = Generator(CONFIG)

gen.generate()
gen.range_check()
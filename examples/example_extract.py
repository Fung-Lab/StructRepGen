'''
    Use Behler ACSF descriptor to extract features.

    To customize parameters, make a yaml file in configs/
    In this example, we will use configs/example_extract_reconstruct.yaml

    Output csv files are saved under data/representation/
'''

import yaml
from structrepgen.extraction.representation_extraction import *
from structrepgen.utils.dotdict import dotdict

# load our config yaml file
stream = open('./configs/example/example_extract_reconstruct.yaml')
CONFIG = yaml.safe_load(stream)
stream.close()
# dotdict for dot operations on Python dict e.g., CONFIG.cutoff
CONFIG = dotdict(CONFIG)

# create extractor instance
extractor = RepresentationExtraction(CONFIG)
# extract
extractor.extract()
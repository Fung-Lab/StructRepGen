import numpy as np
import pandas as pd
import ase, ase.db, csv, yaml, torch

from structrepgen.descriptors.behler import Behler
from structrepgen.descriptors.generic import *

class RepresentationExtraction:
    def __init__(self, CONFIG) -> None:
        '''
        Extract representation R from raw data.

            Parameters:
                CONFIG (dict): descriptor configurations for extraction 

            Returns:
                NIL
        '''
        self.CONFIG = CONFIG

    def extract(self, fname=None):
        descriptor = self.CONFIG.descriptor

        if descriptor == 'behler':
            self.behler(fname)
        else:
            raise Exception("Descriptor currently not supported.")

    def behler(self, fname=None):
        behler = Behler(self.CONFIG)

        data_x=[]
        data_y=[]

        db = ase.db.connect(self.CONFIG.data)
        for row in db.select():
            atoms = row.toatoms(add_additional_information=False)
            positions = atoms.get_positions()
            offsets = get_pbc_offsets(np.array(atoms.get_cell()), 0, behler.device)
            distances, rij = get_distances(positions, offsets, behler.device)

            features = behler.get_features(distances, rij, atoms.get_atomic_numbers())
            features = features.cpu().numpy()
            
            data_x.append(list(features))
            data_y.append([row.get('target')])
            break # should be removed in production
        
        with open(self.CONFIG.x_fname, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(data_x)

        with open(self.CONFIG.y_fname, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(data_y)
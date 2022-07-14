import yaml, os, unittest, torch
import numpy as np
from src.extraction.representation_extraction import *
from src.utils.dotdict import dotdict
from original.db_to_data import db_to_data

class TestBehler(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # original implementation
        db_to_data()

        # SRG
        stream = open('./configs/unittest/unittest_behler.yaml')
        CONFIG = yaml.safe_load(stream)
        stream.close()
        self.CONFIG = dotdict(CONFIG)
        extractor = RepresentationExtraction(self.CONFIG)
        extractor.extract()

        self.original_x = './data/unittest/unittest_original_x.csv'
        self.original_y = './data/unittest/unittest_original_y.csv'
        self.srg_x = self.CONFIG.x_fname
        self.srg_y = self.CONFIG.y_fname
    
    def test_file_existence(self):
        '''
        Test if output files are successfully generated
        '''

        for file in [self.original_x, self.original_y, self.srg_x, self.srg_y]:
            self.assertTrue(os.path.exists(file))
    
    def test_x(self):
        '''
        Test if output x files are the same
        '''

        original_x_tensor = torch.from_numpy(np.genfromtxt(self.original_x, delimiter=','))
        srg_x_tensor = torch.from_numpy(np.genfromtxt(self.srg_x, delimiter=','))

        same = torch.all(torch.eq(original_x_tensor, srg_x_tensor))
        self.assertTrue(same.item())

    def test_y(self):
        '''
        Test if output y files are the same
        '''

        original_y_tensor = torch.from_numpy(np.genfromtxt(self.original_y, delimiter=','))
        srg_y_tensor = torch.from_numpy(np.genfromtxt(self.srg_y, delimiter=','))

        same = torch.all(torch.eq(original_y_tensor, srg_y_tensor))
        self.assertTrue(same.item())
    
    @classmethod
    def tearDownClass(self):
        for file in [self.original_x, self.original_y, self.srg_x, self.srg_y]:
            os.remove(file)

if __name__ == "__main__":
    unittest.main(verbosity=2)
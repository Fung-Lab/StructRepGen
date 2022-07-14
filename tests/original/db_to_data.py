import numpy as np
import ase, csv
import ase.db
from .original_descriptor import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def db_to_data():
    processing_arguments = {}
    g2_params = {'eta': torch.tensor([0.01, 0.06, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 3.5, 5.0], device=device), 'Rs': torch.tensor([
        0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10], device=device)}
    g5_params = {'lambda': torch.tensor([-1, 1], device=device), 'zeta': torch.tensor([1, 2, 4, 16, 64], device=device),
                'eta': torch.tensor([0.06, 0.1, 0.2, 0.4, 1.0], device=device), 'Rs': torch.tensor([0], device=device)}
    processing_arguments['cutoff'] = 20
    processing_arguments['mode'] = 'features'
    processing_arguments['g2_params'] = g2_params
    processing_arguments['g5_params'] = g5_params
    processing_arguments['g5'] = True
    processing_arguments['average'] = True

    db = ase.db.connect('./data/raw/data.db')
    data_x=[]
    data_y=[]

    for row in db.select():
        ase_structure = row.toatoms(add_additional_information=False)
        positions = torch.tensor(
            ase_structure.get_positions(), device=device, dtype=torch.float)
        distances, rij = get_distances(positions, get_pbc_offsets(torch.tensor(
            np.array(ase_structure.get_cell()), device=device, dtype=torch.float), 0))
        features = get_ACSF_features(
            distances, rij, ase_structure.get_atomic_numbers(), processing_arguments)
        features = torch.squeeze(features).cpu().numpy()

        data_x.append(list(features))
        data_y.append([row.get('target')])
        break
    
    with open("./data/unittest/unittest_original_x.csv", 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(data_x)

    with open("./data/unittest/unittest_original_y.csv", 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(data_y)

if __name__ == "__main__":
    db_to_data()
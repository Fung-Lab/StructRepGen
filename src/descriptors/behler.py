import numpy as np
import itertools, torch, warnings
from itertools import combinations, combinations_with_replacement
from src.utils.utils import torch_device_select

class Behler:
    '''
    Behler-Parinello Atomic-centered symmetry functions (ACSF) for representation extraction
    See
        1. Behler, J. Chem. Phys. (2011)
    '''
    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)

    def get_features(self, distances, rij, atomic_numbers):
        '''
        wrapper method for self.get_ACSF_features()
        '''
        return self.get_ACSF_features(distances, rij, atomic_numbers)
    
    def get_ACSF_features(self, distances, rij, atomic_numbers):
        '''
        Get Atomic-centered symmetry functions (ACSF) features (representation R)

            Parameters:
                distances ():
                rij ():
                atomic_numbers ():

            Returns:
                #TODO
        '''

        n_atoms = distances.shape[0]
        type_set1 = self.create_type_set(atomic_numbers, 1)
        type_set2 = self.create_type_set(atomic_numbers, 2)

        atomic_numbers = torch.tensor(atomic_numbers, dtype=int)
        atomic_numbers = atomic_numbers.view(1, -1).expand(n_atoms, -1)
        atomic_numbers = atomic_numbers.reshape(n_atoms, -1).numpy()

        # calculate size of all_features tensor
        g2p = self.CONFIG.g2_params
        g5p = self.CONFIG.g5_params
        G2_out_size = len(g2p.Rs) * len(g2p.eta) * len(type_set1)
        G5_out_size = len(g5p.Rs) * len(g5p.eta) * len(g5p.zeta) * len(g5p.lambdas) * len(type_set2) if self.CONFIG.g5 else 0
        all_features = torch.zeros((n_atoms, G2_out_size+G5_out_size), device=self.device, dtype=torch.float)

        for i in range(n_atoms):
            Rc = self.CONFIG.cutoff
            mask = (distances[i, :] <= Rc) & (distances[i, :] != 0.0)
            Dij = distances[i, mask]
            Rij = rij[i, mask]
            IDs = atomic_numbers[i, mask.cpu().numpy()]
            jks = np.array(list(combinations(range(len(IDs)), 2)))

            # get G2
            G2_i = self.get_G2(Dij, IDs, type_set1)

            all_features[i, :G2_out_size] = G2_i
            # get G5
            if self.CONFIG.g5:
                G5_i = self.get_G5(Rij, IDs, jks, type_set2)
                all_features[i, G2_out_size:] = G5_i
            
        if self.CONFIG.average:
            length = all_features.shape[1]
            out = torch.zeros(length * 2, device=self.device, dtype=torch.float)
            out[:length] = torch.max(all_features, dim=0)[0]
            out[length:] = torch.min(all_features, dim=0)[0]
            return out
        
        return all_features

    def get_G2(self, Dij, IDs, type_set):
        '''
        Get G2 symmetry function (radial):
        G_{i}^{2}=\sum_{j} e^{-\eta\left(R_{i j}-R_{s}\right)^{2}} \cdot f_{c}\left(R_{i j}\right)
        See Behler, J. Chem. Phys. (2011) Eqn (6)

            Parameters:
                Dij ():
                IDs ():
                type_set ():
        
            Returns:
                TODO
        '''
        Rc = self.CONFIG.cutoff
        Rs = torch.tensor(self.CONFIG.g2_params.Rs, device=self.device)
        eta = torch.tensor(self.CONFIG.g2_params.eta, device=self.device)

        n1, n2, m, l = len(Rs), len(eta), len(Dij), len(type_set)

        d20 = (Dij - Rs.view(-1, 1)) ** 2
        term = torch.exp(torch.einsum('i,jk->ijk', -eta, d20))
        results = torch.einsum('ijk,k->ijk', term, self.cosine_cutoff(Dij, Rc))
        results = results.view(-1, m)

        G2 = torch.zeros([n1*n2*l], device=self.device)

        for id, j_type in enumerate(type_set):
            ids = self.select_rows(IDs, j_type)
            G2[id::l] = torch.sum(results[:, ids], axis=1)
        
        return G2
    
    def get_G5(self, Rij, IDs, jks, type_set):
        '''
        Get G5 symmetry function (angular)
        See Behler, J. Chem. Phys. (2011) Eqn (9)

            Parameters:
                TODO
        
            Returns:
                TODO
        '''

        Rc = self.CONFIG.cutoff
        Rs = torch.tensor(self.CONFIG.g5_params.Rs, device=self.device)
        eta = torch.tensor(self.CONFIG.g5_params.eta, device=self.device)
        lambdas = torch.tensor(self.CONFIG.g5_params.lambdas, device=self.device)
        zeta = torch.tensor(self.CONFIG.g5_params.zeta, device=self.device)

        n1, n2, n3, n4, l = len(Rs), len(eta), len(lambdas), len(zeta), len(type_set)
        jk = len(jks)  # m1
        if jk == 0:  # For dimer
            return torch.zeros([n1*n2*n3*n4*l], device=self.device)

        rij = Rij[jks[:, 0]]  # [m1, 3]
        rik = Rij[jks[:, 1]]  # [m1, 3]
        R2ij0 = torch.sum(rij**2., axis=1)
        R2ik0 = torch.sum(rik**2., axis=1)
        R1ij0 = torch.sqrt(R2ij0)  # m1
        R1ik0 = torch.sqrt(R2ik0)  # m1
        R2ij = R2ij0 - Rs.view(-1, 1)**2  # n1*m1
        R2ik = R2ik0 - Rs.view(-1, 1)**2  # n1*m1

        R1ij = R1ij0 - Rs.view(-1, 1)  # n1*m1
        R1ik = R1ik0 - Rs.view(-1, 1)  # n1*m1

        powers = 2. ** (1.-zeta)  # n4
        cos_ijk = torch.sum(rij*rik, axis=1)/R1ij0/R1ik0  # m1 array
        term1 = 1. + torch.einsum('i,j->ij', lambdas, cos_ijk)  # n3*m1

        zetas1 = zeta.repeat_interleave(n3*jk).reshape([n4, n3, jk])  # n4*n3*m1
        term2 = torch.pow(term1, zetas1)  # n4*n3*m1
        term3 = torch.exp(torch.einsum('i,jk->ijk', -eta, (R2ij+R2ik)))  # n2*n1*m1
        # * Cosine(R1jk0, Rc) # m1
        term4 = self.cosine_cutoff(R1ij0, Rc) * self.cosine_cutoff(R1ik0, Rc)
        term5 = torch.einsum('ijk,lmk->ijlmk', term2, term3)  # n4*n3*n2*n1*m1
        term6 = torch.einsum('ijkml,l->ijkml', term5, term4)  # n4*n3*n2*n1*m1
        results = torch.einsum('i,ijkml->ijkml', powers, term6)  # n4*n3*n2*n1*m1
        results = results.reshape([n1*n2*n3*n4, jk])

        G5 = torch.zeros([n1*n2*n3*n4*l], device=self.device)
        jk_ids = IDs[jks]
        for id, jk_type in enumerate(type_set):
            ids = self.select_rows(jk_ids, jk_type)
            G5[id::l] = torch.sum(results[:, ids], axis=1)

        return G5

    def create_type_set(self, number_set, order):
        '''
        TODO
        '''
        types = list(set(number_set))
        return np.array(list(combinations_with_replacement(types, order)))
    
    def select_rows(self, data, row_pattern):
        '''
        TODO
        '''
        if len(row_pattern) == 1:
            ids = (data == row_pattern)
        elif len(row_pattern) == 2:
            a, b = row_pattern
            if a == b:
                ids = [id for id, d in enumerate(data) if d[0] == a and d[1] == a]
            else:
                ids = [id for id, d in enumerate(data) if (d[0] == a and d[1] == b) or (d[0] == b and d[1] == a)]
        return ids

    def cosine_cutoff(self, Rij, Rc):
        '''
        Cosine cutoff function
        See Behler, J. Chem. Phys. (2011) Eqn (4)

            Parameters:
                Rij (torch.Tensor): distance between atom i and j
                Rc (float): cutoff radius
            
            Returns:
                out (torch.Tensor): cosine cutoff
        '''
        
        out = 0.5 * (torch.cos(np.pi * Rij / Rc) + 1.)
        out[out > Rc] = 0.
        return out
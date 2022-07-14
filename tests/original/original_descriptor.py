import numpy as np
import itertools
import math
from itertools import combinations, combinations_with_replacement

import torch
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

###obtains distance with pytorch tensors
def get_distances(positions, pbc_offsets):

    num_atoms = len(positions)
    num_cells = len(pbc_offsets)     

    pos1 = positions.view(-1, 1, 1, 3).expand(-1, num_atoms, 1, 3)
    pos1 = pos1.expand(-1, num_atoms, num_cells, 3)
    pos2 = positions.view(1, -1, 1, 3).expand(num_atoms, -1, 1, 3)
    pos2 = pos2.expand(num_atoms, -1, num_cells, 3)    
                     
    pbc_offsets = pbc_offsets.view(-1, num_cells, 3).expand(pos2.shape[0], num_cells, 3)
    pos2 = pos2 + pbc_offsets        
    ###calculates distance between target atom and the periodic images of the other atom, then gets the minimum distance
    atom_distance_sqr = torch.linalg.norm(pos1 - pos2, dim=-1)      
    atom_distance_sqr_min, min_indices = torch.min(atom_distance_sqr, dim=-1)
    #atom_distance_sqr_min = torch.amin(atom_distance_sqr, dim=-1)
    
    atom_rij = pos1 - pos2
    min_indices = min_indices[..., None, None].expand(-1, -1, 1, atom_rij.size(3))
    atom_rij = torch.gather(atom_rij, dim=2, index=min_indices).squeeze()
                
    return atom_distance_sqr_min, atom_rij
 
        
###Obtain unit cell offsets for distance calculation
class PBC_offsets():
    def __init__(self,cell, supercell_max=4):
        #set up pbc offsets for minimum distance in pbc
        self.pbc_offsets=[]      
        
        for offset_num in range(0, supercell_max):
            unit_cell=[]
            offset_range = np.arange(-offset_num, offset_num+1)
        
            for prod in itertools.product(offset_range, offset_range, offset_range):
                unit_cell.append(list(prod))
        
            unit_cell = torch.tensor(unit_cell, dtype=torch.float, device=device)
            self.pbc_offsets.append(torch.mm(unit_cell, cell.to(device)))

    def get_offset(self, offset_num):                
        return self.pbc_offsets[offset_num]
###functional form
def get_pbc_offsets(cell, offset_num):
    
    unit_cell=[]
    offset_range = np.arange(-offset_num, offset_num+1)
    
    ##put this out of loop
    for prod in itertools.product(offset_range, offset_range, offset_range):
        unit_cell.append(list(prod))

    unit_cell = torch.tensor(unit_cell, dtype=torch.float, device=device)
           
    return torch.mm(unit_cell, cell.to(device))


###ACSF evaluation function
def get_ACSF_features(distances, rij, atomic_numbers, parameters):

    num_atoms = distances.shape[0]
        
    #offsets=get_pbc_offsets(cell, 2)
    #distances, rij = get_distances(positions, offsets)  
    #print(distances.shape, rij.shape)
    
    type_set1 = create_type_set(atomic_numbers, 1)
    type_set2 = create_type_set(atomic_numbers, 2)     
    
    atomic_numbers = torch.tensor(atomic_numbers, dtype=int)  
    atomic_numbers = atomic_numbers.view(1, -1).expand(num_atoms, -1)  
    atomic_numbers = atomic_numbers.reshape(num_atoms, -1).numpy()  
 
    for i in range(0, num_atoms):

        mask = (distances[i, :] <= parameters['cutoff']) & (distances[i, :] != 0.0)
        Dij = distances[i, mask]    
        Rij = rij[i, mask] 
        IDs = atomic_numbers[i, mask.cpu().numpy()] 
        jks = np.array(list(combinations(range(len(IDs)), 2)))
                       
        G2i = calculate_G2_pt(Dij, IDs, type_set1, parameters['cutoff'], parameters['g2_params']['Rs'], parameters['g2_params']['eta'])       
        if parameters['g5'] == True:
            G5i = calculate_G5_pt(Rij, IDs, jks, type_set2, parameters['cutoff'], parameters['g5_params']['Rs'], parameters['g5_params']['eta'], parameters['g5_params']['zeta'], parameters['g5_params']['lambda'])
            G_comb=torch.cat((G2i, G5i), dim=0).view(1,-1)
        else:
            G_comb=G2i.view(1,-1)
        
        if i == 0:
            all_G=G_comb
        else:
            all_G=torch.cat((all_G, G_comb), dim=0)
    
    if parameters['average']==True:
        all_G_max = torch.max(all_G, dim=0)[0]
        all_G_min = torch.min(all_G, dim=0)[0]
        all_G = torch.cat((all_G_max, all_G_min)).unsqueeze(0)
    
    return all_G    
        
        
### ACSF sub-functions
###G2 symmetry function (radial)
def calculate_G2_pt(Dij, IDs, type_set, Rc, Rs, etas):

    n1, n2, m, l = len(Rs), len(etas), len(Dij), len(type_set)
    d20 = (Dij - Rs.view(-1,1)) ** 2  # n1*m
    term = torch.exp(torch.einsum('i,jk->ijk', -etas, d20)) # n2*n1*m
    results = torch.einsum('ijk,k->ijk', term, cosine_cutoff_pt(Dij, Rc)) # n2*n1*m
    results = results.reshape([n1*n2, m])

    G2 = torch.zeros([n1*n2*l], device=device)
    for id, j_type in enumerate(type_set):
        ids = select_rows(IDs, j_type)
        G2[id::l] = torch.sum(results[:, ids], axis=1)

    return G2


###G5 symmetry function (angular)
def calculate_G5_pt(Rij, IDs, jks, type_set, Rc, Rs, etas, zetas, lambdas):

    n1, n2, n3, n4, l = len(Rs), len(etas), len(lambdas), len(zetas), len(type_set)
    jk = len(jks)  # m1
    if jk == 0: # For dimer
        return torch.zeros([n1*n2*n3*n4*l], device=device)
    
    rij = Rij[jks[:,0]] # [m1, 3]
    rik = Rij[jks[:,1]] # [m1, 3]
    R2ij0 = torch.sum(rij**2., axis=1) 
    R2ik0 = torch.sum(rik**2., axis=1) 
    R1ij0 = torch.sqrt(R2ij0) # m1
    R1ik0 = torch.sqrt(R2ik0) # m1
    R2ij = R2ij0 - Rs.view(-1,1)**2  # n1*m1
    R2ik = R2ik0 - Rs.view(-1,1)**2  # n1*m1
   
    R1ij = R1ij0 - Rs.view(-1,1) # n1*m1
    R1ik = R1ik0 - Rs.view(-1,1) # n1*m1
    
    powers = 2. ** (1.-zetas) #n4
    cos_ijk = torch.sum(rij*rik, axis=1)/R1ij0/R1ik0 # m1 array
    term1 = 1. + torch.einsum('i,j->ij', lambdas, cos_ijk) # n3*m1

    zetas1 = zetas.repeat_interleave(n3*jk).reshape([n4, n3, jk])  # n4*n3*m1
    term2 = torch.pow(term1, zetas1) # n4*n3*m1
    term3 = torch.exp(torch.einsum('i,jk->ijk', -etas, (R2ij+R2ik))) # n2*n1*m1
    term4 = cosine_cutoff_pt(R1ij0, Rc) * cosine_cutoff_pt(R1ik0, Rc) #* Cosine(R1jk0, Rc) # m1
    term5 = torch.einsum('ijk,lmk->ijlmk', term2, term3) #n4*n3*n2*n1*m1
    term6 = torch.einsum('ijkml,l->ijkml', term5, term4) #n4*n3*n2*n1*m1
    results = torch.einsum('i,ijkml->ijkml', powers, term6) #n4*n3*n2*n1*m1
    results = results.reshape([n1*n2*n3*n4, jk])

    G5 = torch.zeros([n1*n2*n3*n4*l], device=device)
    jk_ids = IDs[jks]
    for id, jk_type in enumerate(type_set):
        ids = select_rows(jk_ids, jk_type)
        G5[id::l] = torch.sum(results[:, ids], axis=1) 

    return G5
 

###
def create_type_set(number_set, order):
    types = list(set(number_set))
    return np.array(list(combinations_with_replacement(types, order)))


###Aggregation function
#slow
def select_rows(data, row_pattern):
    if len(row_pattern) == 1:
        ids = (data==row_pattern)
    elif len(row_pattern) == 2:
        a, b = row_pattern
        if a==b:        
            ids = [id for id, d in enumerate(data) if d[0]==a and d[1]==a]
        else:
            ids = [id for id, d in enumerate(data) if (d[0] == a and d[1]==b) or (d[0] == b and d[1]==a)]
    return ids


###Cosine cutoff function
def cosine_cutoff_pt(Rij, Rc):
    mask = (Rij > Rc)
    result = 0.5 * (torch.cos(torch.tensor(np.pi, device=device) * Rij / Rc) + 1.)
    result[mask] = 0
    return result    
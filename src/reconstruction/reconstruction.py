import time, ase, torch, os, pickle
from turtle import pos
import numpy as np
import torch.nn.functional as F
from ase import io, Atoms
from skopt.space import Space
from skopt.sampler import Lhs, Halton, Grid
from src.models.models import *
from src.descriptors.behler import Behler
from src.descriptors.generic import *
from src.models.CVAE import *
from src.utils.utils import torch_device_select

class Reconstruction:
    '''
    Reconstruction
    '''
    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG

        # select gpu or cpu for pytorch
        self.device = torch_device_select(self.CONFIG.gpu)

        # initialize descriptor
        if self.CONFIG.descriptor == 'behler':
            self.descriptor = Behler(CONFIG)
        else:
            raise ValueError('Unrecognised descriptor method: {}.'.format(self.CONFIG.descriptor))

        self.model_ff = torch.load(self.CONFIG.ff_model_path)
        self.model_ff.to(self.device)

        torch.backends.cudnn.benchmark = True
        
    def main(self):
        '''

        '''
        tic = time.time()

        data_list = self.create_datalist()
        self.model_ff.eval()

        for i in range(len(data_list)):
            best_positions, _ = self.basin_hopping(
                data_list[i],
                total_trials=2,
                max_hops=10,
                lr=0.2,
                displacement_factor=2,
                max_loss=0.0001,
                verbose=True
            )

            optimized_structure = Atoms(
                numbers=data_list[i]['atomic_numbers'],
                positions=best_positions.detach().cpu().numpy(),
                cell=data_list[i]['cell'].cpu().numpy(),
                pbc=(True, True, True)
            )

            # save reconstructed cifs
            filename = self.CONFIG.reconstructed_file_path + str(i) + '_reconstructed.cif'
            ase.io.write(filename, optimized_structure)

            pbc_offsets = get_pbc_offsets(data_list[i]['cell'], 0, self.device)
            distances, rij = get_distances(best_positions, pbc_offsets, self.device)
            features = self.descriptor.get_features(distances, rij, data_list[i]['atomic_numbers'])

            # inference on FF model
            y0 = self.model_ff(data_list[i]['representation'])
            y1 = self.model_ff(features)
            print(y0, y1)
        
        elapsed_time = time.time() - tic
        print('Total elapsed time: {:.3f}'.format(elapsed_time))

    def basin_hopping(
        self,
        data,
        total_trials = 20,
        max_hops = 500,
        lr = 0.05,
        displacement_factor = 2,
        max_loss = 0.01,
        write = False,
        verbose = False
    ):
        '''
            
        '''
        # setting for self.optimize()
        max_iter = 140

        offset_count = 0
        offsets = PBC_offsets(data['cell'], self.device, supercell_max=1)

        best_global_loss = float('inf')
        best_global_positions = None
        converged = False

        for _ in range(total_trials):
            # initialize random structure
            generated_pos = self.initialize(
                len(data['atomic_numbers']),
                data['cell'],
                data['atomic_numbers'],
                sampling_method='random'
            )
            
            tic = time.time()
            best_local_loss = float('inf')
            best_local_positions = None

            for hop in range(max_hops):
                # TODO: parse loss function, optimizer and scheduler as args
                loss_func = F.l1_loss
                optimizer = torch.optim.Adam([generated_pos], lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.8,
                    patience=15,
                    min_lr=0.0001,
                    threshold=0.0001
                )

                best_positions, _, loss = self.optimize(
                    generated_pos,
                    data['cell'],
                    offsets.get_offset(offset_count),
                    data['atomic_numbers'],
                    data['representation'],
                    loss_func,
                    optimizer,
                    scheduler,
                    max_iter,
                    verbosity=999,
                    early_stopping=100,
                    threshold=0.0001,
                    convergence=0.00005
                )

                elapsed_time = time.time() - tic
                tic = time.time()

                if verbose:
                    print("BH Hop: {:04d}, Loss: {:.5f}, Time (s): {:.4f}".format(hop, loss.item(), elapsed_time)) 
                
                if loss.item() < best_local_loss:
                    # TODO: implement metropolis criterion
                    best_local_positions = best_positions.detach().clone()
                    best_local_loss = loss.item()

                if loss.item() < best_global_loss:
                    best_global_positions = best_local_positions
                    best_global_loss = loss.item()

                if best_local_loss < max_loss:
                    print("Convergence criterion met.")
                    converged = True
                    break

                # apply random shift in positions
                # TODO: change hardcoded value of 0.5
                displacement = (np.random.random_sample(best_positions.shape) - 0.5) * displacement_factor
                displacement = torch.tensor(displacement, device=self.device, dtype=torch.float)

                generated_pos = best_local_positions.detach().clone() + displacement
                generated_pos.requires_grad_()

            if converged:
                break
        
        if verbose:
            print('Ending Basin Hopping, fine optimization of best structure. Best global loss: {:5.3f}'.format(best_global_loss))
        
        best_global_positions.requires_grad_()
        optimizer = torch.optim.Adam([best_global_positions], lr=lr*0.2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=20,
            min_lr=0.0001,
            threshold=0.0001
        )

        best_global_positions, _, loss = self.optimize(
            best_global_positions,
            data['cell'],
            offsets.get_offset(offset_count),
            data['atomic_numbers'],
            data['representation'],
            loss_func,
            optimizer,
            scheduler,
            900,
            verbosity=(10 if verbose == True else 999),
            early_stopping=20,
            threshold=0.00001,
            convergence=0.00005
        )

        if verbose:
            print('Best Basin Hopping loss: {:.5f}'.format(loss.item()))
        return best_global_positions, loss

    def optimize(
        self,
        positions,
        cell,
        pbc_offsets,
        atomic_numbers,
        target_representation,
        loss_function,
        optimizer,
        scheduler,
        max_iterations = 100,
        verbosity = 5,
        early_stopping = 1,
        threshold = 0.01,
        convergence = 0.001
    ):
        '''
        
        '''
        tic = time.time()


        all_positions = [positions.detach().clone()]
        count, loss_history = 0, []
        best_positions = None
        best_loss = float('inf')
        min_dist = 1.5

        while count < max_iterations:
            optimizer.zero_grad()
            lr = scheduler.optimizer.param_groups[0]['lr']

            # get representation / reconstruction loss
            distances, rij = get_distances(positions, pbc_offsets, self.device)
            features = self.descriptor.get_features(distances, rij, atomic_numbers)

            features = torch.unsqueeze(features, 0)

            loss = loss_function(features, target_representation)
            loss.backward(retain_graph=False)
            loss_history.append(loss)

            optimizer.step()
            scheduler.step(loss)

            #print loss and time taken   
            if (count + 1) % verbosity == 0:          
                elapsed_time = time.time() - tic
                tic = time.time()
                print("System Size: {:04d}, Step: {:04d}, Loss: {:.5f}, LR: {:.5f}, Time/step (s): {:.4f}".format(len(positions), count+1, loss.item(), lr, elapsed_time/verbosity))                                  

            # save best structure
            if loss < best_loss:
                best_positions = positions.detach().clone()
                best_loss = loss
            
            count += 1

            if best_loss < convergence:
                break

            if early_stopping > 1 and count > early_stopping:
                if abs((loss.item() - loss_history[-early_stopping]) / loss_history[-early_stopping]) < threshold:                    
                    break
            
            all_positions.append(positions.detach().clone())
        
        return best_positions, all_positions, best_loss

    def initialize(self, structure_len, cell, atomic_numbers, sampling_method='random', write=False):
        '''
        Initialize structure

            Parameters:

            Returns:

        '''
        space = Space([(0., 1.)] * 3)
        if sampling_method == 'random':
            positions = np.array(space.rvs(structure_len))
        else:
            raise ValueError("Unrecognised sampling method.")
        
        generated_structure = Atoms(
            numbers=atomic_numbers,
            scaled_positions=positions,
            cell=cell.cpu().numpy(),
            pbc=(True, True, True)
        )

        positions = generated_structure.positions
        if write == True:
            # TODO
            print("Implement write")
            pass

        return torch.tensor(positions, requires_grad=True, device=self.device, dtype=torch.float)

    def create_datalist(self):
        '''
        Load crystal structure(s) data from directory

            Parameters:
                NIL
            
            Returns:
                TODO

        '''
        representation = np.loadtxt(self.CONFIG.structure_file_path, dtype='float', delimiter=',')
        cell = self.CONFIG.cell
        data_list = []

        for i in range(len(representation)):
            data = {}

            data['atomic_numbers'] = self.CONFIG.atoms
            data['cell'] = torch.tensor(np.array(cell), dtype=torch.float)
            data['representation'] = torch.unsqueeze(torch.tensor(representation[i], dtype=torch.float, device=self.device), 0)
            data_list.append(data)
        
        return data_list
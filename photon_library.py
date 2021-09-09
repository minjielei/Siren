import h5py  as h5
import numpy as np
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PhotonLibrary(object):
    def __init__(self, fname='plib.h5'):
        if not os.path.isfile(fname):
            print('Downloading photon library file... (>300MByte, may take minutes')
            os.system('curl -O https://www.nevis.columbia.edu/~kazuhiro/plib.h5 ./')
        if not os.path.isfile(fname):
            print('Error: failed to download the photon library file...')
            raise Exception

        with h5.File(fname,'r') as f:
            self._vis  = np.array(f['vis'])
            self._min  = np.array(f['min'])
            self._max  = np.array(f['max'])
            self.shape = np.array(f['numvox'])

    def numpy(self):
        x = np.linspace(0, 1, self.shape[0])
        y = np.linspace(0, 1, self.shape[1])
        z = np.linspace(0, 1, self.shape[2])

        coordx, coordy, coordz = np.meshgrid(x, y, z)
        coord = np.reshape(np.stack([coordx, coordy, coordz], -1), (-1, 3))
        axis = (coord * (self.shape-1)).astype(int)
        data = self.VisibilityFromAxisID(axis)

        return coord, data

    def UniformSample(self,num_points=32,use_numpy=True,use_world_coordinate=False):
        '''
        Samples visibility for a specified number of points uniformly sampled within the voxelized volume
        INPUT
          num_points - number of points to be sampled
          use_numpy - if True, the return is in numpy array. If False, the return is in torch Tensor
          use_world_coordinate - if True, returns absolute (x,y,z) position. Else fractional position is returned.
        RETURN
          An array of position, shape (num_points,3)
          An array of visibility, shape (num_points,180)
        '''
        
        array_ctor = np.array if use_numpy else torch.Tensor
        
        pos = np.random.uniform(size=num_points*3).reshape(num_points,3)
        axis_id = (pos[:] * (self.shape-1)).astype(np.int32)
        
        if use_world_coordinate:
            pos = array_ctor(self.AxisID2Position(axis_id))
        else:
            pos = array_ctor(pos)
            
        vis = array_ctor(self.VisibilityFromAxisID(axis_id))

        return pos,vis

    def VisibilityFromAxisID(self, axis_id, ch=None):
        return self.Visibility(self.AxisID2VoxID(axis_id),ch)

    def VisibilityFromXYZ(self, pos, ch=None):
        if not torch.is_tensor(pos):
          pos = torch.tensor(pos, device=device)
        return self.Visibility(self.Position2VoxID(pos), ch)

    def Visibility(self, vids, ch=None):
        '''
        Returns a probability for a detector to observe a photon.
        If ch (=detector ID) is unspecified, returns an array of probability for all detectors
        INPUT
          vids - Tensor of integer voxel IDs
          ch  - Integer (valid range 0 to N-1) to specify an optical detector (optional)
        RETURN
          Probability(ies) in FP32 to observe a photon at a specified location for each vid
        '''
        if ch is None:
            return self._vis[vids]
        return self._vis[vids][ch]

    def AxisID2VoxID(self, axis_id):
        '''
        Takes an integer ID for voxels along xyz axis (ix, iy, iz) and converts to a voxel ID
        INPUT
          axis_id - Length 3 integer array noting the position in discretized index along xyz axis
        RETURN
          The voxel ID (single integer)          
        '''
        return axis_id[:, 0] + axis_id[:, 1]*self.shape[0] + axis_id[:, 2]*(self.shape[0] * self.shape[1])

    def AxisID2Position(self, axis_id):
        '''
        Takes a axis ID (discretized location along xyz axis) and converts to a xyz position (x,y,z)
        INPUT
          axis_id - The axis ID in an integer array (ix,iy,iz)
        RETURN
          Length 3 floating point array noting the position along xyz axis
        '''    
        return self._min + (self._max - self._min) / self.shape * (axis_id + 0.5)

    def Position2VoxID(self, pos):
        '''
        Takes a tensor of xyz position (x,y,z) and converts to a tensor of voxel IDs
        INPUT
          pos - Tensor of length 3 floating point array noting the position along xyz axis
        RETURN
          Tensor of sigle integer voxel IDs       
        '''
        axis_ids = ((pos - self._min) / (self._max - self._min) * self.shape).int()

        return (axis_ids[:, 0] + axis_ids[:, 1] * self.shape[0] +  axis_ids[:, 2]*(self.shape[0] * self.shape[1])).long()
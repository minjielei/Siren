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
            
        pmt_data = np.loadtxt('pmt_loc.csv',skiprows=1,delimiter=',')
        if not (pmt_data[:,0].astype(np.int32) == np.arange(pmt_data.shape[0])).all():
            raise Exception('pmt_loc.csv contains optical channel data not in order of channel numbers')
        self._pmt_pos = pmt_data[:,1:4]
        self._pmt_dir = pmt_data[:,4:7]
        if not self._pmt_pos.shape[0] == self._vis.shape[1]:
            raise Exception('Optical detector count mismatch: photon library %d v.s. pmt_loc.csv %d' % (self._vis.shape[1],
                                                                                                        self._pmt_pos.shape[0])
                           )
        if ((self._pmt_pos < self._min) | (self._max < self._pmt_pos)).any():
            raise Exception('Some PMT positions are out of the volume bounds')
        # Convert the PMT positions in a normalized coordinate (fractional position within the voxelized volume)
#         self._pmt_pos = (self._pmt_pos - self._min) / (self._max - self._min)

    def numpy(self):
        axis = self.VoxID2AxisID(np.arange(self._vis.shape[0]))
        coord = (axis + 0.5) / self.shape
        data = self.VisibilityFromAxisID(axis.astype(int))

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
        axis_id = (pos[:] * self.shape).astype(np.int32)
        
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
    
    def VoxID2AxisID(self, vid):
        '''
        Takes a voxel ID and converts to discretized index along xyz axis
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 integer array noting the position in discretized index along xyz axis
        '''
        xid = vid.astype(int) % self.shape[0]
        yid = ((vid - xid) / self.shape[0]).astype(int) % self.shape[1]
        zid = ((vid - xid - (yid * self.shape[0])) / (self.shape[0] * self.shape[1])).astype(int) % self.shape[2]
        
        return np.reshape(np.stack([xid,yid,zid], -1), (-1, 3)).astype(np.float32) 
    
    def Visibility2D(self, axis, frac, gt, pred, ch=None):
        '''
        Provides a 2D slice of a visibility map at a fractional location along the specified axis
        INPUT
          axis - One of three cartesian axis 'x', 'y', or 'z'
          frac - A floating point value in the range [0,1] to specify the location, in fraction, along the axis
          ch   - An integer or a list of integers specifying a (set of) optical detectors (if not provided, visibility for all detectors are summed)
        RETURN
          2D (XY) slice of a visibility map
        '''
        axis_labels = ['x','y','z']
        ia, ib, itarget = 0,0,0
        if   axis == 'x': itarget, ia, ib = 0,1,2
        elif axis == 'y': itarget, ia, ib = 1,2,0
        elif axis == 'z': itarget, ia, ib = 2,0,1
        else:
            print('axis must be x, y, or z')
            raise ValueError
        
        if frac < 0 or 1.0 < frac:
            print('frac must be between 0.0 and 1.0')
            raise ValueError
            
        loc_target = int(float(frac) * self.shape[itarget] + 0.5)
        result_gt = np.zeros(shape=[self.shape[ia],self.shape[ib]],dtype=np.float32)
        result_pred = np.zeros(shape=[self.shape[ia],self.shape[ib]],dtype=np.float32)
        axis_id = [0,0,0]
        chs = np.arange(len(self._vis[0]))
        if ch is not None:
            chs = [ch] if type(ch) == type(int()) else ch
        for loc_a in range(self.shape[ia]):
            for loc_b in range(self.shape[ib]):
                axis_id[itarget] = loc_target
                axis_id[ia]      = loc_a
                axis_id[ib]      = loc_b
                vid = self.AxisID2VoxID(np.array([axis_id]))[0]
                for ch in chs:
                    result_gt[loc_a][loc_b] += gt[vid][ch]
                    result_pred[loc_a][loc_b] += pred[vid][ch]
        return [result_gt, result_pred]
    
    def PlotVisibility2D(self,axis,frac,gt,pred,ch=None):
        '''
        Visualize a 2D slice of a visibility map at a fractional location along the specified axis
        INPUT
          axis - One of three cartesian axis 'x', 'y', or 'z'
          frac - A floating point value in the range [0,1] to specify the location, in fraction, along the axis
          ch   - An integer or a list of integers specifying a (set of) optical detectors (if not provided, visibility for all detectors are summed)
        RETURN
          figure object
        '''
        axis_labels = ['x','y','z']
        ia, ib = 0,0
        if   axis == 'x': ia, ib = 1,2
        elif axis == 'y': ia, ib = 2,0
        elif axis == 'z': ia, ib = 0,1
        else:
            print('axis must be x, y, or z')
            raise ValueError
            
        ar=self.Visibility2D(axis,frac,gt,pred,ch)
        pos_range=np.column_stack([self._min,self._max])
        extent = np.concatenate([pos_range[ib], pos_range[ia]])
        
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        fig,ax=plt.subplots(2, 1, figsize=(16,6), sharex=True,sharey=True,facecolor='w', constrained_layout=True)
        color_map = plt.cm.get_cmap('viridis')
        reversed_color_map = color_map.reversed()
        for i in range(2):
            ar[i] = - np.log(ar[i] + 1e-7)
            im = ax[i].matshow(ar[i],extent=extent, cmap=reversed_color_map, vmin = np.min(ar[0]), vmax = np.max(ar[0]))
            ax[i].tick_params(labelsize=16,bottom=True,top=False,left=True,right=False,labelleft=True,labelbottom=True)
            ax[i].set_ylabel('%s [cm]' % axis_labels[ia].upper(),fontsize=20)
            
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), pad=0.02)
        cbar.ax.tick_params(labelsize=14)
       
        ax[1].set_xlabel('%s [cm]' % axis_labels[ib].upper(),fontsize=20)
        return fig 
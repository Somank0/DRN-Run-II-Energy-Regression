from torch_geometric.data import InMemoryDataset, Data
import torch
import pickle
import Extract
import numpy as np
import awkward as ak

class ECALDataset(InMemoryDataset):
    def __init__(self, path):
        self.path = path
        print("Loading dataset from",path)
        super().__init__()

        self.data, self.slices = self.read()

    @property
    def variables(self):
        raise NotImplementedError

    @property
    def raw_dir(self):
        return self.path

    @property
    def processed_dir(self):
        return self.path

    @property
    def raw_file_names(self):
        return ['%s.pickle'%(var) for var in self.variables]

    @property
    def processed_file_names(self):
        raise NotImplementedError

    @property
    def minima(self):
        raise NotImplementedError

    @property
    def ranges(self):
        raise NotImplementedError

    @property
    def eventlevel(self):
        raise NotImplementedError

    def dump(self, data, slices):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def download(self):
        raise Exception("Can't find feature files. Run extract first")

    def process(self):
        print("processing...")
        data = []
        for var in self.variables:
            print("Reading",var)
            with open("%s/%s.pickle"%(self.path, var), 'rb') as f:
                data.append(pickle.load(f))
        print("Making feats")
        print("Rescaling")

        slices = []

        for i in range(len(data)):
            if i not in self.eventlevel:
                slices.append(np.concatenate(([0], np.cumsum(ak.to_numpy(ak.num(data[i])), dtype=np.int64))))
                data[i] = ak.to_numpy(ak.flatten((data[i]-self.minima[i])/self.ranges[i])).astype(np.float32)
            else:
                slices.append(np.arange(len(data[i])+1))
                data[i] = ak.to_numpy((data[i]-self.minima[i])/self.ranges[i]).astype(np.float32)

            data[i] = torch.from_numpy(data[i])
            slices[i] = torch.from_numpy(slices[i])

        self.dump(data,slices)
    
class ObjectDataset(ECALDataset):
    def __init__(self, path, nonoise=False, noflags=False, pho=None):
        self.nonoise = nonoise
        self.noflags = noflags
        if pho is None:
            self.pho = 'pho' in path or "Pho" in path
        else:
            self.pho = pho

        super().__init__(path)

    @property
    def variables(self):
        variables = ['Hit_X', 'Hit_Y', 'Hit_Z', 'RecHitEn', 'RecHitFrac']
        if not self.nonoise:
            variables += ['HitNoise']
        if not self.noflags:
            variables += ['RecHitFlag_kGood', 'RecHitFlag_kOutOfTime', 'RecHitFlag_kPoorCalib',
                          'RecHitGain']
        variables += ['Hit_ES_X', 'Hit_ES_Y', 'Hit_ES_Z', 'ES_RecHitEn']
        if not self.noflags:
            variables += ['RecHitFlag_kESGood']
        variables += ['rho']
        if self.pho:
            variables += ['Pho_HadOverEm']
        else:
            variables += ['Ele_HadOverEm']
        return variables

    @property
    def processed_file_names(self):
        fnames = ['%s_scaled.pt'%var for var in self.variables]
        fnames += ['slices_xECAL.pt']
        if not self.noflags:
            fnames += ['slices_fECAL.pt']
        fnames += ['slices_xES.pt']
        if not self.noflags:
            fnames += ['slices_fES.pt']
        fnames += ['slices_gx.pt']
        return fnames

    @property
    def eventlevel(self):
        return [len(self.variables)-2, len(self.variables)-1]

    def dump(self, data, slices):
        for i in range(len(data)):
            print("saving",self.variables[i],"to",self.processed_paths[i])
            torch.save(data[i], self.processed_paths[i], pickle_protocol=4)

        i+=1
        print("saving slices",self.variables[0],"to",self.processed_paths[i])
        torch.save(slices[0], self.processed_paths[i], pickle_protocol=4)
        if not self.noflags:
            i+=1
            print("saving slices",self.variables[7],"to",self.processed_paths[i])
            torch.save(slices[7], self.processed_paths[i], pickle_protocol=4)
            i+=1
            print("saving slices",self.variables[10],"to",self.processed_paths[i])
            torch.save(slices[10], self.processed_paths[i], pickle_protocol=4)
            i+=1
            print("saving slices",self.variables[14],"to",self.processed_paths[i])
            torch.save(slices[14], self.processed_paths[i], pickle_protocol=4)
        else:
            i+=1
            print("saving slices",self.variables[6],"to",self.processed_paths[i])
            torch.save(slices[6], self.processed_paths[i], pickle_protocol=4)
        i+=1
        print("saving slices",self.variables[-1],"to",self.processed_paths[i])
        torch.save(slices[-1], self.processed_paths[-1], pickle_protocol=4)

    def read(self):
        datalist = []
        sliceslist = []
        for path in self.processed_paths:
            if 'slice' in path:
                sliceslist.append(torch.load(path))
            else:
                datalist.append(torch.load(path))

        data = Data()
        slices = {}

        xECALlist = [ datalist[0], datalist[1], datalist[2], datalist[3]*datalist[4] ]
        if not self.nonoise:
            xECALlist += [datalist[5]]
        data.xECAL = torch.stack( xECALlist, 1)
        slices['xECAL'] = sliceslist[0]

        if not self.noflags:
            i = 5 if self.nonoise else 6
            flagslist = [datalist[i], datalist[i+1], datalist[i+2]]
            flagsint = torch.zeros(len(flagslist[0]), dtype=torch.int64)
            for j in range(len(flagslist)):
                flagsint += torch.round(flagslist[j]).int() * (2**j)
            data.fECAL = flagsint 
            slices['fECAL'] = sliceslist[1]

            gain = datalist[i+3]
            data.gainECAL = torch.round(2*gain).int()
            slices['gainECAL'] = sliceslist[1]

        i = 5 if self.noflags else 9
        if not self.nonoise:
            i+=1

        xESlist = [ datalist[i], datalist[i+1], datalist[i+2], datalist[i+3] ]
        data.xES = torch.stack( xESlist, 1)
        slices['xES'] = sliceslist[1 if self.noflags else 2]

        if not self.noflags:
            fECALlist = [datalist[i+4]]
            fESint = torch.zeros(len(fECALlist[0]), dtype=torch.int64)
            for j in range(len(fECALlist)):
                fESint += torch.round(fECALlist[j]).int() * (2**j)
            data.fES = fESint
            slices['fES'] = sliceslist[3]

        data.gx = torch.stack( (datalist[-2], datalist[-1]), 1)
        slices['gx'] = sliceslist[-1]

        return data, slices

    def process(self):
        super().process()

    @property
    def minima(self):
        minima = [-150., -150., -330., 0.0, 0.0,]
        if not self.nonoise:
            minima += [0.9]
        if not self.noflags:
            minima += [0, 0, 0, 0]
        minima += [-150, -150, -150, 0]
        if not self.noflags:
            minima += [0]
        minima += [0, 0]
        return minima

    @property
    def ranges(self):
        ranges = [300, 300, 660, 250, 1] 
        if not self.nonoise:
            ranges += [3]
        if not self.noflags:
            ranges += [1, 1, 1, 12]
        ranges += [300, 300, 660, 0.1]
        if not self.noflags:
            ranges += [1]
        ranges += [13, 0.05]
        return ranges

class MustacheDataset(ECALDataset):
    def __init__(self, path, nonoise=False, noflags=False):
        self.nonoise = nonoise
        self.noflags = noflags
        super().__init__(path)

    @property
    def variables(self): 
        variables = ['Hit_X', 'Hit_Y', 'Hit_Z', 'RecHitEn', 'RecHitFrac']
        if not self.nonoise:
            variables += ['HitNoise']
        if not self.noflags:
            variables += ['RecHitFlag_kGood', 'RecHitFlag_kOutOfTime', 'RecHitFlag_kPoorCalib', 
                           'RecHitGain']
        variables += ['rho']
        return variables
 
    @property
    def processed_file_names(self):
        fnames = ['%s_scaled.pt'%var for var in self.variables]
        fnames += ['slices_xECAL.pt']
        if not self.noflags:
            fnames += ['slices_fECAL.pt']
        fnames += ['slices_gx.pt']
        return fnames

    @property
    def eventlevel(self):
        return [len(self.variables)-1]

    def dump(self, data, slices):
        for i in range(len(data)):
            print("saving",self.variables[i],"to",self.processed_paths[i])
            torch.save(data[i], self.processed_paths[i], pickle_protocol=4)

        i+=1
        print("saving slices",self.variables[0],"to",self.processed_paths[i])
        torch.save(slices[0], self.processed_paths[i], pickle_protocol=4)
        if not self.noflags:
            i+=1
            print("saving slices",self.variables[7],"to",self.processed_paths[i])
            torch.save(slices[7], self.processed_paths[i], pickle_protocol=4)
        i+=1
        print("saving slices",self.variables[-1],"to",self.processed_paths[i])
        torch.save(slices[-1], self.processed_paths[-1], pickle_protocol=4)

    def read(self):
        datalist = []
        sliceslist = []
        for path in self.processed_paths:
            if 'slice' in path:
                print("reading", path)
                sliceslist.append(torch.load(path))
            else:
                print("reading", path)
                datalist.append(torch.load(path))

        data = Data()
        slices = {}

        xECALlist = [ datalist[0], datalist[1], datalist[2], datalist[3]*datalist[4] ]
        if not self.nonoise:
            xECALlist += [datalist[5]]
        data.xECAL = torch.stack( xECALlist, 1 )
        slices['xECAL'] = sliceslist[0]

        if not self.noflags:
            i = 5 if self.nonoise else 6
            flagslist = [datalist[i], datalist[i+1], datalist[i+2]]
            flagsint = torch.zeros(len(datalist[i]), dtype=torch.int64)
            for j in range(len(flagslist)):
                flagsint += torch.round(datalist[j]).int() * (2**j)
            data.fECAL = flagsint 
            slices['fECAL'] = sliceslist[1]

            data.gainECAL = datalist[i+3] 
            data.gainECAL = torch.round(2*data.gainECAL).int()
            slices['gainECAL'] = sliceslist[1]


        data.gx = datalist[-1]
        slices['gx'] = sliceslist[-1]
        
        return data, slices

    def process(self):
        super().process()

    @property
    def minima(self):
        minima = [-150., -150., -330., 0., 0.]
        if not self.nonoise:
            minima += [0.9]
        if not self.noflags:
            minima += [0.,0.,0.,0.]
        minima+=[0.]
        return minima

    @property
    def ranges(self):
        ranges = [300, 300, 660, 250, 1]
        if not self.nonoise:
            ranges += [3.0]
        if not self.noflags:
            ranges += [1.0, 1.0, 1.0, 12.0]
        ranges += [13.0]
        return ranges

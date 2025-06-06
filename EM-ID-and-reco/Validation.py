import seaborn as sns
from scipy.stats import norm
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import itertools
from time import time
import numpy as np
import pickle
from Hist import Hist, transforms
from tqdm import tqdm
import pandas as pd
import math
from bw_conv_cb.python.fit import fit as bwfit
import os
import awkward as ak
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d, binned_statistic
import mplhep as hep
from scipy.special import voigt_profile
from scipy.signal import convolve
from scipy.interpolate import interp1d
import probfit

plt.style.use(hep.style.CMS)

variables = ['trueE', 'rawE']
variables = ['trueE', 'abseta', 'HoE', 'rho', 'phi', 'R9',  'rawE']
#variables = ['trueE', 'eta', 'pt', 'rawE']

units = {
    'trueE' : ' [GeV]',
    'R9' : '',
    'eta' : '',
    'rho' : ' [GeV]',
    'phi' : '',
    'nHit' : '',
    'HoE' : '',
    'abseta': '',
    'sigma_pred': '',
    'correction' : '',
    'response' : ''
}


xlabels = {
    'trueE' : r'$E_{true}$',
    'R9' : 'R9',
    'eta' : r'$\eta$',
    'rho' : 'Median Transverse Energy Density',
    'phi' : r'$\phi$',
    'nHit' : r'$N_{hits}$',
    'HoE' : r'$H/E$',
    'abseta': r'$|\eta|$',
    'sigma_pred': r'$\log(\sigma_{y})$',
    'correction' : r'$E_{pred}/E_{raw}$',
    'response' : r'$E_{pred}/E_{true}$'
}

varnames = {
    'trueE': 'Ele_Gen_E',
    'rawE': 'Ele_SCRawE',
    'R9': 'Ele_R9',
    'dR': 'dR',
    'eta' : 'Ele_Gen_Eta',
    'phi' : 'Ele_Gen_Phi',
    'HoE' : 'Ele_HadOverEm',
    'etareco' : 'eta',
    'phireco' : 'phi',
    'rho' : 'rho',
    'subdet' : 'subdet',
    'iEta' : 'iEta',
    'iPhi' : 'iPhi',
    'iZ' : 'iZ'
}

varnames_pho = {
    'trueE': 'Pho_Gen_E',
    'rawE': 'Pho_SCRawE',
    'R9': 'Pho_R9',
    'dR': 'dR',
    'eta' : 'Pho_Gen_Eta',
    'phi' : 'Pho_Gen_Phi',
    'HoE' : 'Pho_HadOverEm',
    'etareco' : 'eta',
    'phireco' : 'phi',
    'rho' : 'rho',
    'subdet' : 'subdet',
    'iEta' : 'iEta',
    'iPhi' : 'iPhi',
    'iZ' : 'iZ'
   }

import matplotlib

plasma = matplotlib.cm.get_cmap("cool")
viridis = matplotlib.cm.get_cmap('Wistia')

def gaus(x, A, mu, sigma):
    return A * np.exp( -np.square(x-mu)/(2*sigma*sigma))

def resfun(x, A, B, C):
    stoch2 = A*A/x 
    noise2 = B*B/(x*x)
    const2 = C*C
    return np.sqrt(stoch2 + noise2 + const2)

def resfun_noiseless(x, A, C):
    return resfun(x, A, 0, C)

class Validation: 

#####################
# CONSTRUCTOR       #
#####################

    def __init__(self, folder=None, data_folder=None,  
            idx_name = 'all',
            target = 'logratioflip', photons=False, sigmoid=None):
        ''' 
        Initialize validation class
        Set up folders where data is read from and written to

        If you've already evaluated the model, 
        you don't need to bother with the various model parameters

        @param folder: the folder that everything lives in (model, output files, etc)
        @param data_folder: the folder that the data is from
        @param idx_name: the name of the train/test split index file
    
        @param target: the target name

        @param loop: whether the model has self loops
        @param pool: the type of pooling to use
            must be one of max, mean, add
        @param in_layers: depth of input network
        @param agg_layers: number of aggregation layers
        @param mp_layers: depth of message-passing networks
        @param out_layers: depth of output network
        '''
        self.folder = folder
        self.data_folder = '/home/rusack/shared/pickles/%s'%data_folder
        #self.data_folder = '/nobackup/users/srothman/%s'%data_folder
        self.idx_name = idx_name

        self.target = target


        self.data = {}
        self.bin_results = {
                'train' : {},
                'valid' : {}}

        self.mee_fit = {}

        self.sigmoid = sigmoid
        if self.sigmoid is not None and self.sigmoid<0:
            self.sigmoid=None

        self.photons=photons

####################
# Data loading     #
####################

    def _loadVariable (self, var):
        '''
        Load in variable var
        '''
        if self.photons:
            varname = varnames_pho[var]
        else:
            varname = varnames[var]

        print('Loading %s...'%var,'(%s)'%varname)
        t0 = time()
        filename = '%s/%s.pickle'%(self.data_folder,varname)
        with open(filename, 'rb') as f:
            self.data[var] = pickle.load(f)
        t1 = time()
        print("\tTook %.3f seconds"%(t1-t0))
        print()

    def loadVariables(self):
        '''
        Load in variables (trueE, eta, etc)
        '''
        for var in variables:
            if var is 'abseta':
                self._loadVariable('eta')
                self.data['abseta'] = np.abs(self.data['eta'])
            else:
                self._loadVariable(var) 
        
        if 'eta' in variables:
            self.data['abseta'] = np.abs(self.data['eta'])

    def loadValidIdx(self):
        prefix = '%s/%s'%(self.data_folder,self.idx_name)
        valididx_file = prefix+'_valididx.pickle'
        trainidx_file = prefix+'_trainidx.pickle'

        if os.path.exists(valididx_file):
            with open(valididx_file, "rb") as f:
                self.valid_idx = np.asarray(pickle.load(f))
        else:
            print("no such valididx as", valididx_file)
            self.valid_idx = np.arange(len(self.data['rawE']))


        if os.path.exists(trainidx_file):
            with open(trainidx_file, 'rb') as f:
                self.train_idx = np.asarray(pickle.load(f))
        else:
            self.train_idx = []

        print(len(self.valid_idx), 'valid points')
        print(len(self.train_idx), 'train points')

    def load(self):
        self.loadVariables()
        self.loadValidIdx()
        self.loadPredictions()

####################
# Load predictions #
####################

    def loadPredictions(self, idx=True): 
        predname = '%s/pred.pickle'%(self.folder)
        #truename = '%s/true.pickle'%(self.folder)

        with open(predname, 'rb') as f:
            if self.sigmoid is not None:
                self.y_pred = (sigmoid( np.asarray(pickle.load(f))) - 0.5) * 2 * np.log(self.sigmoid);
            else:
                self.y_pred = np.asarray( pickle.load(f))

    def loadSigmaPred(self):
        with open('%s/sigma_pred.pickle'%self.folder, 'rb') as f:
            self.sigma_pred = np.asarray(pickle.load(f))

    def loadParams(self):
        with open("%s/params.pickle"%self.folder, 'rb') as f:
            self.params = pickle.load(f)

#################################
# Utility                       #
#################################

    def getFactor(self):
        if self.target == 'trueE':
            return self.y_pred/self.data['rawE']
        elif self.target == 'logratioflip':
            return np.exp(-self.y_pred)
        elif self.target == 'ratioflip':
            return np.reciprocal(self.y_pred)
        elif self.target == 'ratio':
            return self.y_pred
        else:
            raise NotImplementedError

    def getEpred(self):
        return self.getFactor() * self.data['rawE']

    def getResponse(self):
        return self.getFactor() * self.data['rawE'] / self.data['trueE']

    def getSigmaEoverE(self):
        if self.target == 'trueE':
            return self.sigma_pred/self.y_pred
        elif self.target == 'ratio':
            #E_pred +- sigmaE = E_raw * (factor +- sigma_pred)
            #sigmaE = sigma-pred * E_raw
            #sigmaE/E = sigma_pred * E_raw / Epred
            #sigmaE/E = sigma_pred / factor
            return self.sigma_pred/self.getFactor()
        else:
            raise NotImplementedError

    def getSigmaE(self):
        return self.getSigmaEoverE() * self.getEpred()

#######################
# Fit Cruijff in bins #
#######################

    def loadBins(self, prefix, variable, what='valid'):
        '''
        Load in the results of binning
        '''
        filename = '%s/%s_%s_results.csv'%(self.folder, prefix, variable)
        self.bin_results[what][variable] = pd.read_csv(filename, index_col = 0)

    def doBins(self, bins, prefix, variable, what='valid'):
        '''
        Bin the predictions along the given indepdent variable 

        Produces a whole bunch of histograms,
            and the csv containing the binning results named:
                %s/%s_%s_results.csv %(self.data_folder, prefix, variable)
                e.g. cartesian/validbins_trueE_results.csv
        
        @param bins: the bins to use. List of (min, max) bin edges 
        @param prefix: the prefix to name all the resulting datafiles with
        @param what: one of 'valid' or 'train', depending on what you want to bin
        '''

        var = self.data[variable] #get indepdendent variable

        results = []
        for bin in bins:
            if what == 'valid':
                idx = self.valid_idx
                y_pred = self.y_pred[self.valid_idx]
            else:
                idx = self.train_idx
                y_pred = self.y_pred[self.train_idx]

            mean, meanerr, res, reserr, chisq, center = Hist.do_histogram(
                y_pred, self.data['rawE'][idx], self.data['trueE'][idx],
                self.target,
                "%s/%s"%(self.folder, prefix), 
                binvar = var[idx], binmin = bin[0], binmax=bin[1], binname = variable)

            results += [(bin[0], bin[1], mean, meanerr, res, reserr, chisq, center)]
        
        results = np.reshape(results, (-1, 8))
        results = pd.DataFrame(results, 
            columns = ['Min', 'Max','Mean', 'MeanErr', 'Resolution', 'ResolutionErr', 'Chisq', "Center"])
        
        self.bin_results[what][variable] = results

        outname = '%s/%s_%s_results.csv'%(self.folder, prefix, variable)
        results.to_csv(outname)

    def doFineBins(self, binsize, prefix, variable, what='valid', EE = False):
        '''
        Bin the predictions along the given independent variable
        Construct bins such that each bin has the same number of datapoints in it

        @param binsize: the number of datapoints to put in each bin
        @param prefix: the prefix to name all the resulting datafiles with
        @param what: one of 'valid' or 'train', depending on what you want to bin
        @param EE: whether or not we are in the endcaps. 
            If we are in the barrel, we truncate the eta range to be within the barrel
        '''
        var = self.data[variable]
        bins = []
        if what == 'valid':
            in_order = sorted(var[self.valid_idx])
        else:
            in_order = sorted(var[self.train_idx])

        i = 0

        #cut off eta outside the barrel
        #while variable=='eta' and not EE and in_order[i] < -1.4442:
        #    i+=1

        i_prev = i
        print(np.max(in_order), np.min(in_order))
        print("LAST:", in_order[-binsize])
        print('BINSIZE',binsize)

        while i <len(in_order)-binsize:
            i+=binsize
            if in_order[i_prev]!=in_order[i]:
                #print(in_order[i])
                #if we're binning eta and we've run out of barrel, cut off there
                #if variable == 'eta' and not EE and in_order[i] > 1.4442:
                    #bins += [(in_order[i_prev], 1.4442)]
                    #break
                #else: #keep going
                bins += [(in_order[i_prev], in_order[i])]
                i_prev = i

        self.doBins(bins, prefix, variable, what)

    def binEverything(self, name='bins', train=False, bin_like=None, bin_like_prefix=None):
        '''
        Bin everything worth binning
        @param name: the name to identify all the output files, etc generated by this run
        '''
        bin_variables = variables[:-1]
        if 'eta' in variables:
            bin_variables += ['abseta'] #not rawE
        whats = ['valid']
        if train:
            whats+=['train']

        nbins=20
    
        binsize_train = math.ceil(len(self.train_idx)/nbins)
        binsize_valid = math.ceil(len(self.valid_idx)/nbins)

        if bin_like is not None:
            like_mod = Validation(bin_like, self.data_folder)

        for var, what in itertools.product(bin_variables, whats):
            prefix = '%s_%s'%(name,what)
            print('binning',var,what)

            if what=='train':
                N = binsize_train
            else:
                N = binsize_valid

            if bin_like is None:
                self.doFineBins(N, prefix, var, what)
            else:
                like_mod.loadBins(bin_like_prefix, var, what)
                like_csv = like_mod.bin_results[what][var]
                binmin = like_csv['Min']
                binmax = like_csv['Max']
                bins = [(binmin[i], binmax[i]) for i in range(len(binmin))]
                self.doBins(bins, prefix, var, what)

#########################
# Make line plots       #
#########################

    def plot(self, quantity, variable, prefix, what = 'valid', color = 1, done = True, label = None, ax = None):

        xlabel = xlabels[variable]
        unit = units[variable]
        
        var = self.bin_results[what][variable]

        x = var['Center']
        x = (var['Min'] + var['Max'])/2

        if quantity == 'mean':
            y = var['Mean']
            yerr = var['MeanErr']
            title = 'Mean vs %s'%xlabel
            ylabel = r'Mean $E_{pred}/E_{true}$'
        else:
            y = var['Resolution'] * 100
            yerr = var['ResolutionErr']*100
            title = 'Relative Resolution vs %s'%xlabel
            ylabel = 'Relative Resolution [%]'

        if isinstance(color, float):
            ecolor = viridis(color)
            color = plasma(color)
            fmt = 'o'
        elif color == 1:
            ecolor = 'steelblue'
            color = 'steelblue'
            fmt = 'o'
        else:
            ecolor = 'purple'
            color = 'purple'
            fmt = '^'

        if ax is None:
            ax = plt.gca()
            plt.title(title)
            plt.xlabel(xlabel + unit)

        if quantity!='mean':
            mask = abs(yerr/y) < 1
        else:
            mask = yerr<10000

        mask = np.logical_and(mask, y>0)
        mask = np.logical_and(mask, yerr>0)

        if np.all(yerr==0):
            ax.plot(x, y, color=color, label=label)
        else:
            ax.errorbar(x[mask], y[mask], yerr=yerr[mask], fmt=fmt, ls='--', color=color,
                ecolor = ecolor, mec = color, capsize=3, ms=10, mfc='none', lw=3,
                label = label)

        #ax.set_xlabel(xlabel + unit)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if label is not None:
            if variable is not 'rho':
                ax.legend(loc='best')
            else:
                ax.legend(loc='upper left')

        if done:
            outname = '%s/%s_%splot_%s.png'%(self.folder,prefix,quantity,variable)
            print("dumping %s..."%outname)
            plt.savefig(outname, format='png', bbox_inches='tight')
            plt.cla()
            plt.clf()

    def compareTrainValid(self, variable, idx_name):
        folders = [self.folder]*2
        labels = ['validation', 'training']
        outfolder = self.folder
        prefixes = ['bins_%s_valid'%idx_name, 'bins_%s_train'%idx_name]

        self.compare(variable, folders, labels, outfolder, 'TrainValid', prefixes) 

#################################################
# Distributions of variables in response tails  #
#################################################

    def tailHists(self, variable, prefix, com=None):
        plt.clf()
        plt.cla()

        y=transforms[self.target](self.y_pred, self.data['rawE'], self.data['trueE'])[self.valid_idx]
        lower = y<0.5
        upper = y>1.5

        if com is not None:
            y2=transforms[com.target](com.y_pred, com.data['rawE'], com.data['trueE'])[com.valid_idx]
            lower2 = y2<0.5
            upper2 = y2>1.5

        print("Upper:",np.sum(upper))
        print("Lower:",np.sum(lower))

        if variable == 'sigma_pred':
            self.loadSigmaPred()
            var = np.log(self.sigma_pred[self.valid_idx])
        elif variable == 'correction':
            var = y*((self.data['trueE']/self.data['rawE'])[self.valid_idx])
        else:
            var = self.data[variable][self.valid_idx]

        if com is not None:
            if variable == 'correction':
                var2 = y2*((com.data['trueE']/com.data['rawE'])[com.valid_idx])
            else:
                var2 = com.data[variable][com.valid_idx]

        if variable == 'correction':
            plt.yscale('log')

        xl = xlabels[variable]
        un = units[variable]

        if variable == 'abseta':
            plt.axvline(1.4442, color='k', alpha=0.5, ls=':')
            plt.axvline(1.57, color='k', alpha=0.5, ls=':')
        
        if com is None:
            labelsL = ['Lower tail', 'Overall']
            labelsU = ['Upper tail', 'Overall']
            varsL = [var[lower], var]
            varsU = [var[upper], var]
            colors = ['steelblue', 'orange']
        else:
            labelsL = ['DRN: Lower tail', 'Overall', "BDT: Lower tail"]
            labelsU = ['DRN: Upper tail', 'Overall', "BDT: Upper tail"]
            varsL = [var[lower], var, var2[lower2]]
            varsU = [var[upper], var, var2[upper2]]
            colors = ['steelblue', 'orange', 'purple']

        if variable == 'R9':
            plt.hist(varsL, bins=100, range=[0,2], label = labelsL, density=True, histtype='step', color=colors)
        else:
            plt.hist(varsL, bins=100, label = labelsL, density=True, histtype='step', color=colors)

        plt.xlabel(xl + un)
        plt.ylabel("Normalized frequency")
        plt.title(r"%s histogram for lower tail (E$_{pred}$/E$_{true} < 0.5)$"%xl)
        fname = '%s/%s_lowertail_%s.png' %(self.folder, prefix, variable)
        plt.legend()
        plt.savefig(fname, format='png', bbox_inches='tight')

        plt.clf()
        plt.cla()

        if variable == 'abseta':
            plt.axvline(1.4442, color='k', alpha=0.5, ls=':')
            plt.axvline(1.57, color='k', alpha=0.5, ls=':')
        if variable=='R9':
            plt.hist(varsU, bins=100, range=[0,2], label = labelsU, density=True, histtype='step', color=colors)
        else:
            plt.hist(varsU, bins=100, label = labelsU, density=True, histtype='step', color=colors)

        if variable == 'correction':
            plt.yscale('log')

        plt.xlabel(xl + un)
        plt.ylabel("Normalized frequency")
        plt.title(r"%s histogram for upper tail (E$_{pred}$/E$_{true} > 1.5$)"%xl)
        fname = '%s/%s_uppertail_%s.png' %(self.folder, prefix, variable)
        plt.legend()
        plt.savefig(fname, format='png', bbox_inches='tight')

    def tailCorrMat(self, prefix):
        plt.clf()
        plt.cla()

        y=transforms[self.target](self.y_pred, self.data['rawE'], self.data['trueE'])[self.valid_idx]
        lower = y<0.5
        upper = y>1.5

        ldict = {
            'trueE' : self.data['trueE'][self.valid_idx][lower],
            'abseta': self.data['abseta'][self.valid_idx][lower],
            'HoE' : self.data['HoE'][self.valid_idx][lower],
            'rho' : self.data['rho'][self.valid_idx][lower],
            'phi': self.data['phi'][self.valid_idx][lower],
            'R9': self.data['R9'][self.valid_idx][lower],
            'nHit': self.data['nHit'][self.valid_idx][lower],
            'corretion': (y*self.data['trueE'][self.valid_idx]/self.data['rawE'][self.valid_idx])[lower]
        }
        ldf = pd.DataFrame.from_dict(ldict)

        udict = {
            'trueE' : self.data['trueE'][self.valid_idx][upper],
            'abseta': self.data['abseta'][self.valid_idx][upper],
            'HoE' : self.data['HoE'][self.valid_idx][upper],
            'rho' : self.data['rho'][self.valid_idx][upper],
            'phi': self.data['phi'][self.valid_idx][upper],
            'R9': self.data['R9'][self.valid_idx][upper],
            'nHit': self.data['nHit'][self.valid_idx][upper],
            'corretion': (y*self.data['trueE'][self.valid_idx]/self.data['rawE'][self.valid_idx])[upper]
        }
        udf = pd.DataFrame.from_dict(udict)

        sns.pairplot(ldf)
        lfname = '%s/%s_lowertail_correl.png'%(self.folder, prefix)
        plt.savefig(lfname, format='png', bbox_inches='tight')

        plt.clf()
        plt.cla()
        sns.pairplot(udf)
        ufname = '%s/%s_uppertail_correl.png'%(self.folder, prefix)
        plt.savefig(ufname, format='png', bbox_inches='tight')

#############################
# Occupancy plots           #
#############################

    def _occupancy(self, prefix, EE, iEtas, iPhis, iZs, cmap, title):
        print("building histogram with title %s..."%title)

        iZ2 = ak.where(iZs==1, iZs, 0)
        xs = ak.to_numpy(ak.flatten(iEtas + 100*iZ2))
        ys = ak.to_numpy(ak.flatten(iPhis))

        if EE:
            binsX = np.arange(201)+0.5
            binsY = np.arange(101)+0.5
        else:
            binsX = np.arange(-85,87)+0.5
            binsY = np.arange(1,361)+0.5

        bins = [binsX, binsY]

        if EE:
            plt.xlabel("iX")
            plt.ylabel("iY")
        else:
            plt.xlabel("iEta")
            plt.ylabel("iPhi")
        plt.title(title)

        plt.hist2d(xs,ys, bins=bins, cmap=cmap, cmin=1)
        plt.colorbar()
        if EE:
            plt.gca().set_aspect('equal')
        fname = "%s/%s_occupancy.png"%(self.folder, prefix)
        plt.savefig(fname, format='png', bbox_inches='tight')

        plt.cla()
        plt.clf()

    def occupancy(self, prefix, EE):
        self._loadVariable('iEta')
        self._loadVariable('iPhi')
        self._loadVariable("rawE")
        self._loadVariable("trueE")
        self._loadVariable('iZ')

        self.loadPredictions()
        
        iEtas = self.data['iEta'][self.valid_idx]
        iPhis = self.data['iPhi'][self.valid_idx]
        iZs = self.data['iZ'][self.valid_idx]

        cmap = plt.cm.Oranges

        y=transforms[self.target](self.y_pred, self.data['rawE'], self.data['trueE'])[self.valid_idx]
        lower = y<0.5
        upper = y>1.5

        self._occupancy(prefix, EE, iEtas, iPhis, iZs, cmap, "Crystal occupancy over all hits")
        self._occupancy(prefix+"_lowertail", EE, iEtas[lower], iPhis[lower], iZs[lower], cmap, "Crystal occupancy for hits in the lower tail\n(E$_{pred}$/E$_{true} < 0.5)$")
        self._occupancy(prefix+"_uppertail", EE, iEtas[upper], iPhis[upper], iZs[upper], cmap, "Crystal occupancy for hits in the upper tail\n(E$_{pred}$/E$_{true} > 1.5)$")

##################################
# plot() multiple models at once #
##################################

    def _compare(models, labels, quantity, variable, prefix, out_folder, what='valid', watermark=True, note = None):

        if len(models) == 2 and len(models[0].bin_results[what][variable]['Min']) == len(models[1].bin_results[what][variable]['Min']): #make ratio plot as well
            fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, 
                    gridspec_kw={'height_ratios':[3,1]}, sharex=True)

            models[0].plot(quantity, variable, prefix, what=what, 
                    color = 1, done=False, label = labels[0], ax = ax0)
            models[1].plot(quantity, variable, prefix, what=what,
                    color = 2, done=False, label = labels[1], ax = ax0)



            topdata = models[0].bin_results[what][variable]
            bottomdata = models[1].bin_results[what][variable]

            if quantity == 'mean':
                title = r'Mean $E_{pred}/E_{true}$ vs '+xlabels[variable]
                top = topdata['Mean']
                toperr = topdata['MeanErr']
                bottom = bottomdata['Mean']
                bottomerr = bottomdata['MeanErr']
                ax0.set_ylim(.95,1.05)
                ax0.axhline(1,c='k',ls='--',alpha=0.5)
            else:
                title = r'Relative Resolution vs '+xlabels[variable]
                top = topdata['Resolution']
                toperr = topdata['ResolutionErr']
                bottom = bottomdata['Resolution']
                bottomerr = bottomdata['ResolutionErr']
        
            ratio = top/bottom
            
            #error propegation
            toperr = toperr/top
            toperr = toperr*toperr

            bottomerr = bottomerr/bottom
            bottomerr = bottomerr*bottomerr

            ratioerr = ratio*np.sqrt(toperr + bottomerr)

            x = (topdata['Min']+topdata['Max'])/2

            ax1.axhline(1,c='k',ls='--',alpha=0.5)
            if quantity == 'mean':
                ax1.set_ylim(0.9,1.1)
                ax1.set_yticks([0.9, 1.0, 1.1])
            else:
                ax1.set_ylim(0.7,1.1)
                ax1.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1])


            mask = np.logical_and(top>0, bottom>0)

            ax1.errorbar(x[mask], ratio[mask], yerr=ratioerr[mask], fmt='o', ls='--', capsize=3,
                    ecolor='steelblue', mec='steelblue', color='steelblue', mfc='none', ms=10, lw=3)
            ax1.set_ylabel("Ratio")
            ax1.grid(True)
            ax1.set_xlabel(xlabels[variable] + units[variable])

            if note is not None:
                t = ax0.text(0.95, 0.95, note, transform=ax0.transAxes, ha='right', va='top')
                if variable != 'rho':
                    fig.canvas.draw()
                    bbox = t.get_window_extent().transformed(ax0.transAxes.inverted())
                    ax0.legend(loc=(0.65, bbox.y0-0.20))
                else:
                    ax0.legend(loc='upper left')

            #ax0.set_title(title)
            plt.tight_layout()

            if watermark:
                #fig.text(0.18, 0.90, 'CMS', transform=ax0.transAxes, fontsize=16, color='black', fontweight='bold')
                #fig.text(0.29, 0.90, 'Work in progress', transform=ax0.transAxes, fontsize=12, color='black', style='italic')
                hep.cms.text("Simulation Preliminary", ax=ax0, loc=0)

            fig.align_ylabels()
            
        else:
            for model, label, i in zip(models, labels, range(len(models))):
                color = float(i/(len(models)-1))
                model.plot(quantity, variable, prefix, what=what, 
                        color = color, done=False, label = label)

        outname = '%s/%s_%splot_%s.png'%(out_folder, prefix, quantity, variable)
        plt.savefig(outname, format='png', bbox_inches='tight')
        plt.cla()
        plt.clf()

    @staticmethod
    def compare(variable, folders, labels, outfolder, fname, prefixes='bins_valid', note=None):
        models = [Validation(folder) for folder in folders]

        if isinstance(prefixes, str):
            prefixes = [prefixes]*len(folders)
        elif len(prefixes)==1:
            prefixes = prefixes*len(folders)

        for model, prefix in zip(models, prefixes):
            model.loadBins(prefix, variable)

        median = False
        for pref in prefixes:
            if pref.startswith('med'):
                median = True
                break

        Validation._compare(models, labels, 'res', variable, fname, outfolder, note=note)
        if not median:
            Validation._compare(models, labels, 'mean', variable, fname, outfolder, note=note)

#####################
# Invariant masses  #
#####################

    def calculate_mee(self, prefix, Hgg=True):
        from coffea.nanoevents.methods import vector
        ak.behavior.update(vector.behavior)
        if self.target == 'logratioflip':
            E = np.exp(-self.y_pred)*self.data['rawE']
        elif self.target == 'trueE':
            E = np.asarray(self.y_pred)
        else:
            print("Error: unimplimented target")

        phi = np.asarray(self.data['phireco'])
        eta = np.asarray(self.data['etareco'])
        pt = E/np.cosh(eta)

        phi = phi[self.valid_idx]
        eta = eta[self.valid_idx]
        pt = pt[self.valid_idx]
        E = E[self.valid_idx]

        fourvecs = ak.zip(
            {

                'pt': pt,
                'eta' : eta,
                'phi' : phi,
                'energy' : E
            },
            with_name='PtEtaPhiELorentzVector',
        )

        N = len(E)

        firsts = fourvecs[0:N:2]
        seconds = fourvecs[1:N:2]

        self.Mee = np.asarray(np.abs(firsts+seconds))

        if Hgg:
            for i in tqdm(range(len(self.Mee)), total=len(self.Mee)):
                if self.Mee[i] < 100 or self.Mee[i] > 180:
                    continue

                first = i*2
                second = i*2+1
                if fourvecs[first]['pt'] > fourvecs[second]['pt']:
                    higher = first
                    lower = second
                else:
                    higher = second
                    lower = first
                if fourvecs[higher]['pt'] < self.Mee[i]/3 or fourvecs[lower]['pt'] < self.Mee[i]/4:
                    self.Mee[i] = 0
                    continue

        print("Total of %d pairs"%(np.count_nonzero(self.Mee)))

        with open("%s/%s_Mee.pickle"%(self.folder, prefix),'wb') as f:
            pickle.dump(np.asarray(self.Mee), f)

    def load_mee(self):
        with open("%s/%s_Mee.pickle"%(self.folder, self.idx_name), 'rb') as f:
            self.Mee = np.asarray(pickle.load(f))

    def plot_mee(self, prefix):
        plt.clf()
        med, sigma = self.fit_mee()
        label = '$\mu=%.2f; \sigma=%.2f$'%(med,sigma)
        plt.hist(self.Mee, bins=80, histtype='step', range=[80,100], label=label)
        plt.legend()
        plt.xlabel("$M_{ee}$ [GeV]")
        plt.ylim(bottom=0)
        plt.ylabel("Events/0.25GeV")
        plt.savefig("%s/%s_Mee.png"%(self.folder, prefix), format='png', bbox_inches='tight')
        plt.clf()


    def fit_mee(self, bins=10000):
        bin_hts, bin_edges = np.histogram(self.Mee, bins=bins, range=[85,95], density=True)
        bin_cts = (bin_edges[1:]+bin_edges[:-1])/2
        
        Zwidth = 2.4952
        Zmass = 91.1876

        '''
        def voigt(x, A, mu, sigma):
            center = mu + Zmass
            return A * voigt_profile(x-center, sigma, Zwidth)

        p0 = [1.0, -1, 1]
        bounds = [ [0, -5, 0], [2, 5, 5]]

        def gauss(x, A, mu, sigma):
            center = mu+Zmass
            return A*norm.pdf(x, loc=center, scale=sigma)
        
        def bw(x, A, mu, width):
            mass = mu + Zmass
            M2 = mass*mass
            W2 = width*width
            gamma = np.sqrt(M2*(M2+W2))
            k = 2*np.sqrt(2)*mass*gamma*width/(np.pi * np.sqrt(M2 + gamma))
            return A* k/(np.square(x*x - M2) + M2*W2)

        def bw_conv_gauss(x, A, mu, sigma):
            x_gauss = np.linspace(-10, 10, 1000)
            y_gauss = gauss(x_gauss, 1, mu, sigma)

            x_bw = np.linspace(Zmass-10, Zmass+10, 1000)
            y_bw = bw(x_bw, 1, 0, Zwidth)

            convolution = convolve(y_bw, y_gauss, mode='same')
            print(convolution.shape)
            function = interp1d(x_gauss+Zmass, convolution)
            return A*function(x)

        popt, pvar = curve_fit(bw_conv_gauss, bin_cts, bin_hts, p0=p0, bounds = bounds, sigma=np.sqrt(bin_hts))
        perr = np.sqrt(np.diag(pvar))

        mu = popt[1] + Zmass
        sigma = popt[2]
        dmu = perr[1]
        dsigma = perr[2]

        res = sigma/mu
        d1 = np.square(dmu/mu)
        d2 = np.square(dsigma/sigma)
        dres = res * np.sqrt(d1 + d2)
        print(popt)
        print("A: %0.3f +- %0.3f"%(popt[0], perr[0]))
        print('mu: %0.3f +- %0.3f'%(mu, dmu))
        print('res: %0.3f +- %0.3f%%'%(100*res, 100*dres))
        
        pred_hts = bw_conv_gauss(bin_cts, *popt)
        raw = bw(bin_cts, 1, 0, Zwidth)

        diff = np.sum(np.square(bin_hts - pred_hts)/bin_hts)
        NDF = bins - 2 - 1
        chisq = diff/NDF
        
        plt.title("Chisq/NDF = %0.3f"%(diff/NDF))

        plt.hist(self.Mee, bins=100, range=[85,95], histtype='step', density=True)
        plt.plot(bin_cts, pred_hts)
        plt.plot(bin_cts, raw)
        plt.show()
        #plt.clf()
        '''

        import probfit
        from iminuit import Minuit
        bw = lambda x: probfit.pdf.rtv_breitwigner(x, Zmass, Zwidth)
        gaus = probfit.pdf.gaussian
        bwconvgauss = probfit.functor.Convolve(bw, gaus, (-10,10), 1000)
        cb = lambda x, alpha, n, mean, sigma, A : A*probfit.pdf.crystalball(x, alpha, n, mean, sigma)
        bwconvcb = probfit.functor.Convolve(bw, cb, (-10,10), 1000)
        
        mask = np.logical_and(self.Mee < 95, self.Mee > 85)
        data = self.Mee[mask].astype(np.float64) #having some weird dtype issues
                                                 #this fixes them; don't really know why
                                                 #probably a bug in probfit
        cost = probfit.costfunc.BinnedChi2(bwconvcb, data)
        #cost = probfit.costfunc.BinnedLH(probfit.pdf.rtv_breitwigner, self.Mee[mask])
        print("fitting...")
        #minuit = Minuit(cost, m=Zmass, gamma=Zwidth)
        #minuit = Minuit(cost, mean=-0.7, sigma=2.0)
        minuit = Minuit(cost, mean=-0.5079, sigma=1.798, alpha=1.031, n=2.426, A=np.sum(mask)/10)
        minuit.migrad()
        minuit.print_fmin()
        cost.draw(minuit)
        plt.show()

        #mu, dmu, sigma, dsigma, chisq = bwfit(bin_cts, bin_hts)
        #median = np.median(self.Mee)
        #filt = np.logical_and(self.Mee>80, self.Mee<100)
        #r = Hist.find_core(self.Mee[filt],0.60)
        #sigma_eff = r[1]-r[0]

        chisq = cost()
        print('chisq = %f'%chisq)

        popt = minuit.values
        perr = minuit.errors
        print(popt)
        print(perr)
        mu = Zmass + popt['mean']
        sigma = popt['sigma']
        dmu = perr['mean']
        dsigma = perr['sigma']
        
        with open("%s/%s_DSCB.ascii"%(self.folder, self.idx_name), 'w') as f:
            f.write("%0.5f\t%0.5f\n%0.5f\t%0.5f\n%0.5f"%(mu, dmu, sigma, dsigma, chisq))

        self.mee_fit['mu'] = mu
        self.mee_fit['dmu'] = dmu
        self.mee_fit['sigma'] = sigma
        self.mee_fit['dsigma'] = dsigma
        self.mee_fit['chisq'] = chisq 

    def fit_mgg(self, bins=80):
        bin_hts, bin_edges = np.histogram(self.Mee, bins=bins, range=[115,135])
        bin_cts = (bin_edges[1:]+bin_edges[:-1])/2
        p0 = [np.max(bin_hts), 125, 4]

        import probfit
        from iminuit import Minuit
        mask = np.logical_and(self.Mee < 130, self.Mee > 120)
        data = self.Mee[mask].astype(np.float64) #having some weird dtype issues
                                                 #this fixes them; don't really know why
                                                 #probably a bug in probfit
        gaus = lambda x, A, mean, sigma : A * probfit.pdf.gaussian(x, mean=mean, sigma=sigma)
        bw = lambda x, mean, sigma: A * probfit.pdf.rtv_breitwigner(x, mean, sigma)
        cost = probfit.costfunc.BinnedChi2(probfit.functor.Extended(probfit.pdf.cruijff), data)
        minuit = Minuit(cost, m_0=125, sigma_L=1.3, sigma_R=1.3, alpha_L=0.25, alpha_R=0.17, N=np.sum(mask)) 
        minuit.migrad()
        minuit.print_fmin()
        cost.draw(minuit)
        plt.show()
        chisq = minuit.fval/34
        popt = minuit.values
        print(popt)
        perr = minuit.errors
        print(perr)

        print('chisqr is',chisq)
        sigma = (np.abs(popt['sigma_L']) + np.abs(popt['sigma_R']))/2
        dsigma = np.sqrt(np.square(perr['sigma_L']) + np.square(perr['sigma_R']))/2
        self.mee_fit['mu'] = popt['m_0']
        self.mee_fit['dmu'] = perr['m_0']
        self.mee_fit['sigma'] = sigma
        self.mee_fit['dsigma'] = dsigma
        self.mee_fit['chisq'] = chisq
        print(self.mee_fit)

        with open("%s/%s_GAUS.ascii"%(self.folder, self.idx_name), 'w') as f:
            f.write("%0.5g\t%0.5g\n%0.5g\t%0.5g\n%0.5g"%(self.mee_fit['mu'], self.mee_fit['dmu'], self.mee_fit['sigma'], self.mee_fit['dsigma'], chisq))
        return self.mee_fit['mu'], self.mee_fit['sigma']

    def load_meefit(self):
        with open("%s/%s_DSCB.ascii"%(self.folder, self.idx_name), 'r') as f:
            lines = f.readlines()

            l0 = lines[0].split()
            l1 = lines[1].split()
            l2 = lines[2]

            self.mee_fit['mu'] = float(l0[0])
            self.mee_fit['dmu'] = float(l0[1])
            self.mee_fit['sigma'] = float(l1[0])
            self.mee_fit['dsigma'] = float(l1[1])
            self.mee_fit['chisq'] = float(l2)

    def load_mggfit(self):
        with open("%s/%s_GAUS.ascii"%(self.folder, self.idx_name), 'r') as f:
            lines = f.readlines()

            l0 = lines[0].split()
            l1 = lines[1].split()
            l2 = lines[2]

            self.mee_fit['mu'] = float(l0[0])
            self.mee_fit['dmu'] = float(l0[1])
            self.mee_fit['sigma'] = float(l1[0])
            self.mee_fit['dsigma'] = float(l1[1])
            self.mee_fit['chisq'] = float(l2)

    @staticmethod
    def compare_mee(folders,  labels, prefixes, outfolder, fname, watermark=True, data=True, note=None, pho=False):
        mods = []
        for folder, prefix in zip(folders, prefixes):
            mods.append(Validation(folder, None, idx_name=prefix))

        textstr = ''

        Mees = []
        if pho:
            plotrange = [115, 135]
        else:
            plotrange = [80, 100]

        for i in range(len(mods)):
            mod = mods[i]
            prefix = prefixes[i] 
            if pho:
                mod.load_mee()
                mod.load_mggfit()
            else:
                mod.load_mee()
                mod.load_meefit()
            Mees.append(mod.Mee)
            sigma = mod.mee_fit['sigma']
            mu = mod.mee_fit['mu']
            textstr += '%s:\n\t$\mu=%.3f$\n\t$\sigma/\mu=%.3f$%%\n'%(labels[i], mu, sigma/mu*100)
        textstr = textstr[:-1]

        if len(folders)==2:
            colors = ['steelblue', 'purple']
            fmts = ['o', '^']
        else:
            colors = None
            fnts = ['o'] * len(folders)

        plt.clf()
        if data:
            for Mee, color, label, fmt in zip(Mees, colors, labels, fmts):
                hist, edges = np.histogram(Mee, bins=80, range=plotrange)
                centers = (edges[1:] + edges[:-1])/2
                plt.errorbar(centers, hist, yerr=np.sqrt(hist), fmt=fmt, label = label, mec=color, color=color, ms=6, mfc='none', capsize=3, lw=2)
        else:
            plt.hist(Mees, bins=80, histtype='step', range=plotrange, color=colors, label = labels)
        plt.xlabel("$M_{ee}$ [GeV]")
        plt.ylim(bottom=0)
        plt.ylabel("Events / 0.25 GeV")
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.05,0.95, textstr, transform=plt.gca().transAxes, fontsize=20, va='top', bbox=props, ha='left')

        if note is not None:
            plt.text(0.95, 0.95, note, ha='right', va='top', transform=plt.gca().transAxes, fontsize=20)

        if watermark:
            #plt.text(0.68, 0.75, 'CMS', transform=plt.gca().transAxes, fontsize=16, color='black', fontweight='bold')
            #plt.text(0.68, 0.70, 'Work in progress', transform=plt.gca().transAxes, fontsize=12, color='black', style='italic')
            if data:
                hep.cms.text("Preliminary",loc=0)
                hep.cms.lumitext(r"2018, 59.83 fb$^{-1}$, $\sqrt{s}=13$ TeV")
            else:
                hep.cms.text("Simulation Preliminary", loc=0)

        plt.legend(loc='lower center')
        plt.savefig("%s/%s_Mee.png"%(outfolder, fname),format='png', bbox_inches='tight')
        plt.clf()

#############################
# Compare p_T distributions #
#############################

    @staticmethod
    def compare_pt(folders, dfolders, labels, targets, idxs, outfolder, fname, pho, sigmoids):
        if sigmoids is None:
            sigmoids = [None] * len(folders)
        mods = []
        for folder, dfolder, target, idx, sigmoid in zip(folders, dfolders, targets, idxs, sigmoids):
            mods.append(Validation(folder, dfolder, target=target, idx_name=idx, photons=pho, sigmoid=sigmoid))

        pts = []
        for mod in mods:
            #mod._loadVariable('trueE')
            mod._loadVariable('rawE')
            mod._loadVariable('etareco')
            mod.loadValidIdx()
            mod.loadPredictions()
            if mod.target == 'trueE':
                predE = mod.y_pred
            elif mod.target == 'logratioflip':
                predE = np.exp(-mod.y_pred)*mod.data['rawE']

            pts.append(predE/np.cosh(mod.data['etareco']))

        plt.clf()
        plt.hist( pts, label = labels, bins=200, histtype='step')
        plt.yscale('log')
        plt.xlabel("Pt [GeV]")
        plt.ylabel("Count")
        plt.legend()
        fname = "%s/%s_pts.png"%(outfolder, fname)
        plt.savefig(fname, format='png', bbox_inches='tight')
        plt.clf()

################
# Heatmaps     #
################

    def plotCorrection(self, bins=100):
        x = self.data['eta'][self.valid_idx]
        y = self.data['phi'][self.valid_idx]
        corr = self.getFactor()[self.valid_idx]
        filt = corr>2
        x = x[filt]
        y = y[filt]
        corr = corr[filt]
        means, xedges, yedges, _ = binned_statistic_2d(x, y, corr, bins=bins)
        plt.clf()
        plt.title("Mean correction factors %s"%xlabels['correction'])
        plt.xlabel("eta")
        plt.ylabel("phi")
        cmap = plt.cm.Oranges
        hep.hist2dplot(means, xedges, yedges, cmap=cmap)
        plt.savefig("%s/%s_corr.png"%(self.folder, self.idx_name), format='png', bbox_inches='tight')
        plt.clf()

    def plotResponse(self, bins=100):
        x = self.data['eta'][self.valid_idx]
        y = self.data['phi'][self.valid_idx]
        resp = self.getResponse()[self.valid_idx]
        corr = self.getFactor()[self.valid_idx]
        filt = corr>2
        x = x[filt]
        y = y[filt]
        resp = resp[filt]
        means, xedges, yedges, _ = binned_statistic_2d(x, y, resp, bins=bins)
        plt.clf()
        plt.title("Mean response %s"%xlabels['response'])
        plt.xlabel("eta")
        plt.ylabel("phi")
        cmap = plt.cm.Oranges
        hep.hist2dplot(means, xedges, yedges, cmap=cmap)
        plt.savefig("%s/%s_resp.png"%(self.folder, self.idx_name), format='png', bbox_inches='tight')
        plt.clf()

    def plotCts(self, bins=100):
        x = self.data['eta'][self.valid_idx]
        y = self.data['phi'][self.valid_idx]
        corr = self.getFactor()[self.valid_idx]
        filt = corr>2
        x = x[filt]
        y = y[filt]
        means, xedges, yedges, _ = binned_statistic_2d(x, y, None, bins=bins, statistic='count')
        plt.clf()
        plt.title("Reco object distribution (counts)")
        plt.xlabel("eta")
        plt.ylabel("phi")
        cmap = plt.cm.Oranges
        hep.hist2dplot(means, xedges, yedges, cmap=cmap)
        plt.savefig("%s/%s_cts.png"%(self.folder, self.idx_name), format='png', bbox_inches='tight')
        plt.clf()

#####################
# Model convergence #
#####################

    def plotConvergence(self, done=True, label=None, semiparam=False, train=False, watermark=True):
        x = np.load(self.folder+'/summaries.npz')
        #plt.rcParams["figure.figsize"] = (30,20)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.title("Convergence")
        plt.yscale('log')
        #plt.ylim((1.5e-1,0.5)) #spicy 
        #plt.ylim(9.0,11.0)

        if train:
            loss = x['train_loss']
        else:
            loss = x['valid_loss']

        if semiparam:
            plt.plot(x['epoch']+1,loss+10,'o--',label=label)
        else:
            plt.plot(x['epoch']+1,loss,'o--',label=label)

        plt.grid(True)

        
        if label is not None:
            plt.legend()

        if done:
            if watermark:
                #plt.text(0.17, 0.90, 'CMS', transform=plt.gca().transAxes, fontsize=16, color='black', fontweight='bold')
                #plt.text(0.29, 0.90, 'Work in progress', transform=plt.gca().transAxes, fontsize=12, color='black', style='italic')
                hep.cms.text("Simulation Preliminary", loc=0)

            outname = '%s/convergence.png'%self.folder
            plt.savefig(outname, format='png', bbox_inches='tight')
            plt.clf()
            plt.cla()

    @staticmethod
    def _compareConvergence(models, labels, outdir, fname,semiparam=False):
        for mod, label in zip(models, labels):
            mod.plotConvergence(done=False, label=label,semiparam=semiparam)

        outname = '%s/%s_convergence.png'%(outdir,fname)
        plt.savefig(outname, format='png', bbox_inches='tight')
        plt.clf()
        plt.cla()

    @staticmethod
    def compareConvergence( folders, labels, outdir, fname, semiparam=False):
        mods = [Validation(folder) for folder in folders]
        Validation._compareConvergence(mods, labels, outdir, fname,semiparam)

#################################
# Dists of all semiparam output #
#################################

    @staticmethod
    def compareParams(folders, dfolders, labels, idxs, prefix, outfolder):
        if type(idxs)==str:
            idxs = [idxs]*len(folders)
        elif len(idxs)==1:
            idxs = idxs*len(folders)

        if type(dfolders) == str:
            dfolders = [dfolders]*len(folders)
        elif len(dfolders) == 1:
            dfolders = dfolders*len(folders)

        mods = [Validation(folder, dfolder, idx_name = idx) for folder, dfolder, idx in zip(folders,dfolders, idxs)]
        for mod in mods:
            mod.loadParams()
            mod.loadValidIdx()

        paramss = [mod.params[:,mod.valid_idx] for mod in mods]

        mus = [params[0,:] for params in paramss]
        sigmas = [params[1,:] for params in paramss]
        alphaLs = [params[2,:] for params in paramss]
        nLs = [params[3,:] for params in paramss]
        alphaRs = [params[4,:] for params in paramss]
        nRs = [params[5,:] for params in paramss]

        plt.clf()
        plt.yscale('log')
        plt.hist(mus, label=labels, bins=100, histtype='step')
        plt.legend(loc='best')
        plt.savefig('%s/%s_mus.png'%(outfolder,prefix), format='png')
        plt.clf()

        plt.yscale('log')
        plt.hist(sigmas, label=labels, bins=100, histtype='step')
        plt.legend(loc='best')
        plt.savefig('%s/%s_sigmas.png'%(outfolder,prefix), format='png')
        plt.clf()

        plt.yscale('log')
        plt.hist(alphaLs, label=labels, bins=100, histtype='step')
        plt.legend(loc='best')
        plt.savefig('%s/%s_alphaLs.png'%(outfolder,prefix), format='png')
        plt.clf()

        plt.yscale('log')
        plt.hist(nLs, label=labels, bins=100, histtype='step')
        plt.legend(loc='best')
        plt.savefig('%s/%s_nLs.png'%(outfolder,prefix), format='png')
        plt.clf()

        plt.yscale('log')
        plt.hist(alphaRs, label=labels, bins=100, histtype='step')
        plt.legend(loc='best')
        plt.savefig('%s/%s_alphaRs.png'%(outfolder,prefix), format='png')
        plt.clf()

        plt.yscale('log')
        plt.hist(nRs, label=labels, bins=100, histtype='step')
        plt.legend(loc='best')
        plt.savefig('%s/%s_nRs.png'%(outfolder,prefix), format='png')
        plt.clf()

##################################
# Response hists                 #
##################################

    @staticmethod
    def compareHistograms(folders, labels,  targets, idxs, data_folders, outdir, fname, core=False, correction=False, pho=False, sigmoids=None):
        plt.clf();
        plt.cla();

        if not core:
            range = [0.0, 2.0]
            plt.yscale('log',nonpositive='mask')

        models = []
        if sigmoids is None:
            sigmoids = [None] * len(folders)
        if len(sigmoids) == 1:
            sigmoids = sigmoids * len(folders)
        for folder,data_folder, target, idx, sigmoid in zip(folders,data_folders, targets, idxs, sigmoids):
            models.append(Validation(folder = folder, data_folder=data_folder, target = target, idx_name = idx, photons=pho, sigmoid=sigmoid))

        ys = []
        Ns = []

        rangemin = 1000
        rangemax = -1000
        sigma60s = []
    
        for mod, label in zip(models, labels):
            mod._loadVariable('rawE')
            mod.loadValidIdx()
            mod.loadPredictions()
            if not correction:
                mod._loadVariable('trueE')
                ys+=[(transforms[mod.target](mod.y_pred, mod.data['rawE'], mod.data['trueE'])[mod.valid_idx])]
            else:
                if mod.target == 'trueE':
                    ys += [(mod.y_pred/mod.data['rawE'])[mod.valid_idx]]
                elif mod.target == 'logratioflip':
                    ys += [np.exp(-mod.y_pred[mod.valid_idx])]
#                ys+=[(transforms[mod.target](mod.y_pred, mod.data['rawE'], mod.data['trueE'])[mod.valid_idx])]
            #if correction:
            #    ys[-1] = ys[-1]*mod.data['trueE'][mod.valid_idx]/mod.data['rawE'][mod.valid_idx]
            #Ns.append([1/len(ys[-1])] * len(ys[-1]))

            r60 = Hist.find_core(ys[-1],0.6)
            sigma60s.append(r60[1]-r60[0])

            newmin, newmax = Hist.find_core(ys[-1], 0.9)
            rangemin = min(newmin, rangemin)
            rangemax = max(newmax, rangemax)

        if core:
            range = [rangemin, rangemax]
        elif not correction:
            plt.axvline(rangemin, color='k', alpha=0.5, ls=':')
            plt.axvline(rangemax, color='k', alpha=0.5, ls=':')

        for y in ys:
            testbins, _ = np.histogram(y, bins=100, range=range)
            Ns.append([1/np.max(testbins)]*len(y))

        if len(ys) == 2:
            colors = ['steelblue', 'purple']
        elif len(ys) ==3:
            colors = ['steelblue', 'orange', 'purple']
        else:
            colors = None

        plt.hist(ys, bins=100, range=range, label = labels, density=False, weights=Ns, color=colors, histtype='step')

        if not core:
            nL = []
            for y in ys:
                large = y>range[-1]
                nlarge = np.sum(large)
                nL.append([nlarge/len(y)])

            x = [[range[-1]+0.15]]*len(ys)

            plt.text(range[-1]+0.05, 5*np.max(nL), "Remaining RH tail", rotation=80)
            plt.axvline(range[-1],color='k')
            plt.hist(x, bins=1, range=[range[-1],range[-1]+0.3], density=False, weights=nL, color=colors)
            plt.xlim(right = range[-1]+0.3)
        else:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            textstr = ''
            for sigma60, label in zip(sigma60s, labels):
                textstr+='%s: $\sigma_{60}=%.4f$\n'%(label, sigma60)
            textstr = textstr[:-1]
            plt.text(0.05,0.95, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=props)

        plt.legend()

        if correction:
            plt.xlabel("$E_{pred}/E_{raw}$")
        else:
            plt.xlabel("$E_{pred}/E_{true}$")

        plt.ylabel("Normalized frequency")

        if correction:
            plt.savefig("%s/%s_correction_hist.png"%(outdir, fname), format='png', bbox_inches='tight')
        else:
            plt.savefig("%s/%s_hist.png"%(outdir, fname), format='png', bbox_inches='tight')
        plt.clf()
        plt.cla()

###################################
# Predicted sigma                 #
###################################

    def plotError(self):   
        plt.clf()
        if not hasattr(self, 'sigma'):
            self.loadSigmaPred()
        if not hasattr(self,'y_pred'):
            self.loadPredictions()
        if not hasattr(self, 'valid_idx'):
            self.loadValidIdx()

        self._loadVariable("trueE")
        self._loadVariable('rawE')

        x = (self.getEpred() - self.data['trueE'])/self.getSigmaE()

        x = x[self.valid_idx]

        mask = np.logical_and(x<3, -3<x)
        x = x[mask]

        mean = np.mean(x)
        stdev = np.std(x)

        plt.hist(x, bins=100)

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = '$\mu = %.3f$\n$\sigma = %.3f$'%(mean, stdev)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=20, va='top', bbox=props, ha='left')

        plt.xlabel(r"$(E_{True} - E_{Pred})/\sigma_{E,Pred}$")
        plt.ylabel("Count")

        plt.savefig("%s/error.png"%self.folder, format='png', bbox_inches='tight')
        plt.clf()

        plt.hist(self.getSigmaEoverE()[self.valid_idx], bins=100, range=[0, 0.1], histtype='step')
        plt.xlabel(r"$\sigma_{E}/E$")
        plt.ylabel("Count")
        plt.savefig("%s/sigmas.png"%self.folder, format='png', bbox_inches='tight')
        plt.clf()

    def medSigma(self, variable, bins=100, what='valid', prefix=None):
        if prefix is None:
            prefix = 'med_%s_%s'%(self.idx_name, what)
        if what == 'valid':
            idx = self.valid_idx
        else:
            idx = self.train_idx

        sigmaEoE = self.getSigmaEoverE()[idx]
        xdata = self.data[variable][idx]

        med, edges, _ = binned_statistic(xdata, sigmaEoE, statistic='median', bins=bins)

        binmin = edges[:-1]
        binmax = edges[1:]

        binmean, _, _ = binned_statistic(xdata, xdata, statistic='mean', bins=bins)

        results = pd.DataFrame(columns = ['Min', 'Max', 'Mean', 'MeanErr', 'Resolution', 'ResolutionErr', 'Chisq', 'Center'])
        results['Min'] = binmin
        results['Max'] = binmax
        results['Mean'] = [-1]*bins #not relevant
        results['MeanErr'] = [-1]*bins #not relevant
        results['Resolution'] = med 
        results['ResolutionErr'] = [0]*bins #medians don't have error bars
        results['Chisq'] = [-1]*bins #not relevant
        results['Center'] = binmean 

        self.bin_results[what][variable] = results

        outname = '%s/%s_%s_results.csv'%(self.folder, prefix, variable)
        results.to_csv(outname)

        return results

    def sigmaVerror(self):
        plt.clf()
        if not hasattr(self, 'sigma'):
            self.loadSigmaPred()
        if not hasattr(self,'y_pred'):
            self.loadPredictions()
        if not hasattr(self, 'valid_idx'):
            self.loadValidIdx()

        self._loadVariable("trueE")
        self._loadVariable('rawE')

        sigmaEoverE = self.getSigmaEoverE()[self.valid_idx]
        propError = np.abs((self.getEpred() - self.data['trueE']) / self.data['trueE'])[self.valid_idx]

        plt.hist([sigmaEoverE, propError], bins=100, range=[0, 0.1])
        plt.show()

        plt.hist2d(sigmaEoverE, propError, range=[(0,0.1), (0,0.1)], bins=100)
        plt.xlabel("Predicted proportional error")
        plt.ylabel("True proportional error")
        plt.show()

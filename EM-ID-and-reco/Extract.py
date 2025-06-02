import uproot 
import numpy as np
import awkward as ak
from time import time
import pickle
import tqdm
from torch_geometric.data import Data
import torch

MISSING = -999999

########################################
# HGCAL Values                         #
########################################

HGCAL_X_Min = -36
HGCAL_X_Max = 36

HGCAL_Y_Min = -36
HGCAL_Y_Max = 36

HGCAL_Z_Min = 13
HGCAL_Z_Max = 265

HGCAL_Min = 0
HGCAL_Max = 2727

########################################
# ECAL Values                          #
########################################

X_Min = -150
X_Max = 150

Y_Min = -150
Y_Max = 150

Z_Min = -330
Z_Max = 330

Eta_Min = -3.0
Eta_Max = 3.0

Phi_Min = -np.pi
Phi_Max = np.pi

iEta_Min = -85
iEta_Max = 85

iPhi_Min = 1
iPhi_Max = 360

iX_Min = 1
iX_Max = 100

iY_Min = 1
iY_Max = 100

ECAL_Min = 0
ECAL_Max = 250

ES_Min = 0
ES_Max = 0.1

Rho_Min = 0
Rho_Max = 13

HoE_Min = 0
HoE_Max = 0.05

Noise_Min = 0.9
Noise_Max = 3.0

def rescale(feature, minval, maxval):
    top = feature-minval
    bot = maxval-minval
    return top/bot

def dphi(phi1, phi2):
    dphi = np.abs(phi1-phi2)
    gt = dphi > np.pi
    dphi[gt] = 2*np.pi - dphi[gt]
    return dphi

def dR(eta1, eta2, phi1, phi2):
    dp = dphi(phi1, phi2)
    de = np.abs(eta1-eta2)

    return np.sqrt(dp*dp + de*de)

def cartfeat_HGCAL(x, y, z, En):
    E = rescale(En, HGCAL_Min, HGCAL_Max)
    x = rescale(x, HGCAL_X_Min, HGCAL_X_Max)
    y = rescale(y, HGCAL_Y_Min, HGCAL_Y_Max)
    z = rescale(z, HGCAL_Z_Min, HGCAL_Z_Max)

    return ak.concatenate((x[:,:,None], y[:,:,None], z[:,:,None], E[:,:,None]), -1)

def cartfeat(x, y, z, En ,frac, det=None):
    E = rescale(En*frac, ECAL_Min, ECAL_Max)
    x = rescale(x, X_Min, X_Max)
    y = rescale(y, Y_Min, Y_Max)
    z = rescale(z, Z_Min, Z_Max)

    if det is None:
        return ak.concatenate((x[:,:,None], y[:,:,None], z[:,:,None], E[:,:,None]), -1)
    else:
        return ak.concatenate((x[:,:,None], y[:,:,None], z[:,:,None], E[:,:,None], det[:,:,None]), -1)

def projfeat(eta, phi, z, En ,frac, det=None):
    E = rescale(En*frac, ECAL_Min, ECAL_Max)
    eta = rescale(eta, Eta_Min, Eta_Max)
    phi = rescale(phi, Phi_Min, Phi_Max)
    z = rescale(z, Z_Min, Z_Max)

    if det is None:
        return ak.concatenate((eta[:,:,None], phi[:,:,None], z[:,:,None], E[:,:,None]), -1)
    else:
        return ak.concatenate((eta[:,:,None], phi[:,:,None], z[:,:,None], E[:,:,None], det[:,:,None]), -1)

def localfeat(i1, i2, z, En ,frac, det=None):
    '''
    In the barrel:
        i1 = iEta
        i2 = iPhi
    In the endcaps:
        i1 = iX
        i2 = iY
    '''

    if det is not None:
        print("Error: local coordinates not defined for ES")
        return

    E = rescale(En*frac, ECAL_Min, ECAL_Max)

    Zfirst = ak.firsts(z)
    barrel = np.abs(Zfirst) < 300 #this is 1 if we are in the barrel, 0 in the endcaps
    
    xmax = barrel * iEta_Max + ~barrel * iX_Max
    xmin = barrel * iEta_Min + ~barrel * iX_Min

    ymax = barrel * iPhi_Max + ~barrel * iY_Max
    ymin = barrel * iPhi_Min + ~barrel * iY_Min

    x = rescale(i1, xmin, xmax)
    y = rescale(i2, ymin, ymax)

    whichEE = 2*(Zfirst > 300) - 1 #+1 to the right of 0, -1 to the left of 0

    iZ = whichEE * ~barrel #0 in the barrel, -1 in left EE, +1 in right EE

    iZ, _ = ak.broadcast_arrays(iZ, x)

    return ak.concatenate((x[:,:,None], y[:,:,None], iZ[:,:,None], E[:,:,None]), -1)

def make_feat_v4(x_ECAL, y_ECAL, z_ECAL, E_ECAL, f_ECAL, noise_ECAL = None,
        good_ECAL = None, time_ECAL = None, calib_ECAL = None, gain_ECAL=None,
        x_ES = None, y_ES=None, z_ES=None, E_ES=None, 
        good_ES = None,
        rho = None, HoE=None):
    print("rescaling")
    x_ECAL = rescale(x_ECAL, X_Min, X_Max)
    y_ECAL = rescale(y_ECAL, Y_Min, Y_Max)
    z_ECAL = rescale(z_ECAL, Z_Min, Z_Max)
    E_ECAL = rescale(E_ECAL * f_ECAL, ECAL_Min, ECAL_Max)

    if noise_ECAL is None:
        feat_ECAL = ak.concatenate( (x_ECAL[:,:,None], 
                                     y_ECAL[:,:,None], 
                                     z_ECAL[:,:,None], 
                                     E_ECAL[:,:,None]), -1)
    else:
        noise_ECAL = rescale(noise_ECAL, Noise_Min, Noise_Max)
        feat_ECAL = ak.concatenate( (x_ECAL[:,:,None], 
                                     y_ECAL[:,:,None], 
                                     z_ECAL[:,:,None], 
                                     E_ECAL[:,:,None],
                                     noise_ECAL[:,:,None]), -1)

    if good_ECAL is not None:
        flags_ECAL = np.concatenate( (good_ECAL[:,:,None],
                                      time_ECAL[:,:,None],
                                      calib_ECAL[:,:,None],
                                      gain_ECAL[:,:,None]), -1)

    if x_ES is not None:
        x_ES = rescale(x_ES, X_Min, X_Max)
        y_ES = rescale(y_ES, Y_Min, Y_Max)
        z_ES = rescale(z_ES, Z_Min, Z_Max)
        E_ES = rescale(E_ES, ES_Min, ES_Max)

        feat_ES = ak.concatenate( (x_ES[:,:,None],
                                   y_ES[:,:,None],
                                   z_ES[:,:,None],
                                   E_ES[:,:,None]), -1)

    if good_ES is not None:
        flags_ES = good_ES

    if rho is not None:
        rho = rescale(rho, Rho_Min, Rho_Max)
    if HoE is not None:
        HoE = rescale(HoE, HoE_Min, HoE_Max)

    if rho is not None and HoE is not None:
        gx = np.concatenate((rho[:,None], HoE[:,None]), -1)
    elif rho is not None:
        gx = rho
    elif HoE is not None:
        gx = HoE
    else:
        gx = None

    
    #torchify
    print("torchifying")
    if feat_ECAL is not None:
        feat_ECAL = ak.pad_none(feat_ECAL, 5 if noise_ECAL is not None else 0, 2)
        feat_ECAL = [torch.from_numpy(ak.to_numpy(x).astype(np.float32)) for x in feat_ECAL]

    if good_ECAL is not None:
        flags_ECAL = ak.pad_none(flags_ECAL, 4, 2)
        flags_ECAL = [torch.from_numpy(ak.to_numpy(x).astype(np.int64)) for x in flags_ECAL]
    else:
        flags_ECAL = [None] * len(feat_ECAL)

    if x_ES is not None:
        feat_ES = ak.pad_none(feat_ES, 4, 2, clip=True)
        feat_ES = [torch.from_numpy(ak.to_numpy(x).astype(np.float32)) for x in feat_ES]
    else:
        feat_ES = [None] * len(feat_ECAL)

    if good_ES is not None:
        flags_ES = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.int64)) for x in flags_ES]
    else:
        flags_ES = [None] * len(feat_ECAL)

    if gx is not None:
        gx = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in gx]
    else:
        gx = [None] * len(feat_ECAL)

    print("building list")
    data = [makedata(xECAL, fECAL, xES, fES, xgx) for xECAL, fECAL, xES, fES, xgx in zip(feat_ECAL, flags_ECAL, feat_ES, flags_ES, gx)]

    print("done")
    return data

def makedata(xECAL, fECAL, xES, fES, xgx):
    result = Data()
    result.xECAL = xECAL
    result.fECAL = fECAL
    result.xES = xES
    result.fES = fES
    result.gx = xgx
    return result

def torchify(feat, graph_x = None):
    data = [Data(x = torch.from_numpy(ak.to_numpy(ele).astype(np.float32))) for ele in feat]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data

def npify(feat):
    t0 = time()
    data = [ak.to_numpy(ele) for ele in feat]
    print("took %f"%(time()-t0))
    return data

varlists = {
    'Hgg_v4': [
                #generic
                'nPhotons',
                #reco info
                'Pho_SCRawE', 'pt', 'eta', 'phi',
                'Pho_R9', 'Pho_HadOverEm', 'rho',
                #ECAL hits
                'iEtaPho1', 'iEtaPho2',
                'iPhiPho1', 'iPhiPho2',
                'Hit_Eta_Pho1', 'Hit_Eta_Pho2',
                'Hit_Phi_Pho1', 'Hit_Phi_Pho2',
                'Hit_X_Pho1', 'Hit_X_Pho2',
                'Hit_Y_Pho1', 'Hit_Y_Pho2',
                'Hit_Z_Pho1', 'Hit_Z_Pho2',
                'RecHitEnPho1', 'RecHitEnPho2',
                'RecHitFracPho1', 'RecHitFracPho2',
                #ECAL flags
                'RecHitFlag_kGood_pho1', 'RecHitFlag_kGood_pho2',
                'RecHitFlag_kOutOfTime_pho1', 'RecHitFlag_kOutOfTime_pho2',
                'RecHitFlag_kPoorCalib_pho1', 'RecHitFlag_kPoorCalib_pho2',
                #'RecHitQuality1', 'RecHitQuality2',
                'RecHitGain1', 'RecHitGain2',
                'HitNoisePho1', 'HitNoisePho2',
                #ES hits
                'Hit_ES_Eta_Pho1', 'Hit_ES_Eta_Pho2',
                'Hit_ES_Phi_Pho1', 'Hit_ES_Phi_Pho2',
                'Hit_ES_X_Pho1', 'Hit_ES_X_Pho2',
                'Hit_ES_Y_Pho1', 'Hit_ES_Y_Pho2',
                'Hit_ES_Z_Pho1', 'Hit_ES_Z_Pho2',
                'ES_RecHitEnPho1', 'ES_RecHitEnPho2',
                #ES flags
                'RecHitFlag_kESGood_pho1', 'RecHitFlag_kESGood_pho2',
                #BDT info
                'energy', 'energy_ecal_mustache'
    ],
    
    'Zee_v4': [
                #generic
                'nElectrons',
                #reco info
                'Ele_SCRawE', 'pt', 'eta', 'phi',
                'Ele_R9', 'Ele_HadOverEm', 'rho',
                #ECAL hits
                'iEtaEle1', 'iEtaEle2',
                'iPhiEle1', 'iPhiEle2',
                'Hit_Eta_Ele1', 'Hit_Eta_Ele2',
                'Hit_Phi_Ele1', 'Hit_Phi_Ele2',
                'Hit_X_Ele1', 'Hit_X_Ele2',
                'Hit_Y_Ele1', 'Hit_Y_Ele2',
                'Hit_Z_Ele1', 'Hit_Z_Ele2',
                'RecHitEnEle1', 'RecHitEnEle2',
                'RecHitFracEle1', 'RecHitFracEle2',
                #ECAL flags
                'RecHitFlag_kGood_ele1', 'RecHitFlag_kGood_ele2',
                'RecHitFlag_kOutOfTime_ele1', 'RecHitFlag_kOutOfTime_ele2',
                'RecHitFlag_kPoorCalib_ele1', 'RecHitFlag_kPoorCalib_ele2',
                #'RecHitQuality1', 'RecHitQuality2',
                'RecHitGain1', 'RecHitGain2',
                'HitNoiseEle1', 'HitNoiseEle2',
                #ES hits
                'Hit_ES_Eta_Ele1', 'Hit_ES_Eta_Ele2',
                'Hit_ES_Phi_Ele1', 'Hit_ES_Phi_Ele2',
                'Hit_ES_X_Ele1', 'Hit_ES_X_Ele2',
                'Hit_ES_Y_Ele1', 'Hit_ES_Y_Ele2',
                'Hit_ES_Z_Ele1', 'Hit_ES_Z_Ele2',
                'ES_RecHitEnEle1', 'ES_RecHitEnEle2',
                #ES flags
                'RecHitFlag_kESGood_ele1', 'RecHitFlag_kESGood_ele2',
                #BDT info
                'energy', 'energy_ecal_mustache'
    ],

    'gjets' : [
        #generic
        'nPhotons',
        #gen info
        'Pho_Gen_E', 'Pho_Gen_Eta', 'Pho_Gen_Phi', 
        #reco info
        'Pho_SCRawE', 'pt', 'eta', 'phi',
        'Pho_R9', 'Pho_HadOverEm', 'rho',
        #ECAL hits
        'iEtaPho1', 'iEtaPho2',
        'iPhiPho1', 'iPhiPho2',
        'Hit_Eta_Pho1', 'Hit_Eta_Pho2',
        'Hit_Phi_Pho1', 'Hit_Phi_Pho2',
        'Hit_X_Pho1', 'Hit_X_Pho2',
        'Hit_Y_Pho1', 'Hit_Y_Pho2',
        'Hit_Z_Pho1', 'Hit_Z_Pho2',
        'RecHitEnPho1', 'RecHitEnPho2',
        'RecHitFracPho1', 'RecHitFracPho2',
        #ES hits
        'Hit_ES_Eta_Pho1', 'Hit_ES_Eta_Pho2',
        'Hit_ES_Phi_Pho1', 'Hit_ES_Phi_Pho2',
        'Hit_ES_X_Pho1', 'Hit_ES_X_Pho2',
        'Hit_ES_Y_Pho1', 'Hit_ES_Y_Pho2',
        'Hit_ES_Z_Pho1', 'Hit_ES_Z_Pho2',
        'ES_RecHitEnPho1', 'ES_RecHitEnPho2',
        #BDT info
        'energy', 'energy_ecal_mustache'
    ],


    'dR_pho' : [
        'nPhotons', 
        'Pho_Gen_E', 'Pho_Gen_Eta', 'Pho_Gen_Phi',
        'eta', 'phi',
        'iEtaPho1', 'iEtaPho2', 'Hit_Z_Pho1', 'Hit_Z_Pho2',
    ],

    'dR_ele' : [
        'nElectrons',
        'Ele_Gen_E', 'Ele_Gen_Eta', 'Ele_Gen_Phi',
        'eta', 'phi',
        'iEtaEle1', 'iEtaEle2', 'Hit_Z_Ele1', 'Hit_Z_Ele2',
    ],

    'gun_v4_pho': [
                #generic
                'nPhotons',
                #gen info, 
                'Pho_Gen_E', 'Pho_Gen_Eta', 'Pho_Gen_Phi', 
                #reco info
                'Pho_SCRawE', 'pt', 'eta', 'phi',
                'Pho_R9', 'Pho_HadOverEm', 'rho',
                #ECAL hits
                'iEtaPho1', 'iEtaPho2',
                'iPhiPho1', 'iPhiPho2',
                'Hit_Eta_Pho1', 'Hit_Eta_Pho2',
                'Hit_Phi_Pho1', 'Hit_Phi_Pho2',
                'Hit_X_Pho1', 'Hit_X_Pho2',
                'Hit_Y_Pho1', 'Hit_Y_Pho2',
                'Hit_Z_Pho1', 'Hit_Z_Pho2',
                'RecHitEnPho1', 'RecHitEnPho2',
                'RecHitFracPho1', 'RecHitFracPho2',
                #ECAL flags
                'RecHitFlag_kGood_pho1', 'RecHitFlag_kGood_pho2',
                'RecHitFlag_kOutOfTime_pho1', 'RecHitFlag_kOutOfTime_pho2',
                'RecHitFlag_kPoorCalib_pho1', 'RecHitFlag_kPoorCalib_pho2',
                #'RecHitQuality1', 'RecHitQuality2',
                'RecHitGain1', 'RecHitGain2',
                'HitNoisePho1', 'HitNoisePho2',
                #ES hits
                'Hit_ES_Eta_Pho1', 'Hit_ES_Eta_Pho2',
                'Hit_ES_Phi_Pho1', 'Hit_ES_Phi_Pho2',
                'Hit_ES_X_Pho1', 'Hit_ES_X_Pho2',
                'Hit_ES_Y_Pho1', 'Hit_ES_Y_Pho2',
                'Hit_ES_Z_Pho1', 'Hit_ES_Z_Pho2',
                'ES_RecHitEnPho1', 'ES_RecHitEnPho2',
                #ES flags
                'RecHitFlag_kESGood_pho1', 'RecHitFlag_kESGood_pho2',
                #BDT info
                'energy', 'energy_ecal_mustache', 'energy_error',
    ],

    'gun_v4_ele': [
                #generic
                'nElectrons',
                #gen info, 
                'Ele_Gen_E', 'Ele_Gen_Eta', 'Ele_Gen_Phi', 
                #reco info
                'Ele_SCRawE', 'pt', 'eta', 'phi',
                'Ele_R9', 'Ele_HadOverEm', 'rho',
                #ECAL hits
                'iEtaEle1', 'iEtaEle2',
                'iPhiEle1', 'iPhiEle2',
                'Hit_Eta_Ele1', 'Hit_Eta_Ele2',
                'Hit_Phi_Ele1', 'Hit_Phi_Ele2',
                'Hit_X_Ele1', 'Hit_X_Ele2',
                'Hit_Y_Ele1', 'Hit_Y_Ele2',
                'Hit_Z_Ele1', 'Hit_Z_Ele2',
                'RecHitEnEle1', 'RecHitEnEle2',
                'RecHitFracEle1', 'RecHitFracEle2',
                #ECAL flags
                'RecHitFlag_kGood_ele1', 'RecHitFlag_kGood_ele2',
                'RecHitFlag_kOutOfTime_ele1', 'RecHitFlag_kOutOfTime_ele2',
                'RecHitFlag_kPoorCalib_ele1', 'RecHitFlag_kPoorCalib_ele2',
                #'RecHitQuality1', 'RecHitQuality2',
                'RecHitGain1', 'RecHitGain2',
                'HitNoiseEle1', 'HitNoiseEle2',
                #ES hits
                'Hit_ES_Eta_Ele1', 'Hit_ES_Eta_Ele2',
                'Hit_ES_Phi_Ele1', 'Hit_ES_Phi_Ele2',
                'Hit_ES_X_Ele1', 'Hit_ES_X_Ele2',
                'Hit_ES_Y_Ele1', 'Hit_ES_Y_Ele2',
                'Hit_ES_Z_Ele1', 'Hit_ES_Z_Ele2',
                'ES_RecHitEnEle1', 'ES_RecHitEnEle2',
                #ES flags
                'RecHitFlag_kESGood_ele1', 'RecHitFlag_kESGood_ele2',
                #BDT info
                'energy', 'energy_ecal_mustache'
    ],

    'BDTvars': ['Pho_R9', #'Pho_S4', S4 is not populated
                 'Pho_SigIEIE', 'Pho_SigIPhiIPhi',
                 'Pho_SCEtaW', 'Pho_SCPhiW',
                 #'Pho_CovIEtaIEta', 'Pho_CovIEtaIPhi','Pho_ESSigRR', not populated
                 'Pho_SCRawE', 
                 'Pho_SC_ESEnByRawE', 'Pho_HadOverEm',
                 'eta', 'phi',
                 'Pho_Gen_Eta', 'Pho_Gen_Phi',
                 'iEtaPho1', 'iEtaPho2', 'Hit_Z_Pho1', 'Hit_Z_Pho2', "Pho_Gen_E"],
    'gun_pho': ['nPhotons', 
                'Pho_Gen_E', 'Pho_Gen_Eta', 'Pho_Gen_Phi', 
                'Pho_SCRawE', 'pt', 'eta', 'phi',
                'Pho_R9', 'Pho_HadOverEm', 'rho',
                'iEtaPho1', 'iEtaPho2',
                'iPhiPho1', 'iPhiPho2',
                'Hit_ES_Eta_Pho1', 'Hit_ES_Eta_Pho2',
                'Hit_ES_Phi_Pho1', 'Hit_ES_Phi_Pho2',
                'Hit_ES_X_Pho1', 'Hit_ES_X_Pho2',
                'Hit_ES_Y_Pho1', 'Hit_ES_Y_Pho2',
                'Hit_ES_Z_Pho1', 'Hit_ES_Z_Pho2',
                'ES_RecHitEnPho1', 'ES_RecHitEnPho2',
                'Hit_Eta_Pho1', 'Hit_Eta_Pho2',
                'Hit_Phi_Pho1', 'Hit_Phi_Pho2',
                'Hit_X_Pho1', 'Hit_X_Pho2',
                'Hit_Y_Pho1', 'Hit_Y_Pho2',
                'Hit_Z_Pho1', 'Hit_Z_Pho2',
                'RecHitEnPho1', 'RecHitEnPho2',
                'RecHitFracPho1', 'RecHitFracPho2',
                #'passLooseId', 'passMediumId', 'passTightId',
                'energy'],
    'Hgg': ['nPhotons', 
                'Pho_SCRawE', 'eta', 'phi',
                'Pho_R9', 'Pho_HadOverEm', 'rho',
                'iEtaPho1', 'iEtaPho2',
                'iPhiPho1', 'iPhiPho2',
                'Hit_ES_Eta_Pho1', 'Hit_ES_Eta_Pho2',
                'Hit_ES_Phi_Pho1', 'Hit_ES_Phi_Pho2',
                'Hit_ES_X_Pho1', 'Hit_ES_X_Pho2',
                'Hit_ES_Y_Pho1', 'Hit_ES_Y_Pho2',
                'Hit_ES_Z_Pho1', 'Hit_ES_Z_Pho2',
                'ES_RecHitEnPho1', 'ES_RecHitEnPho2',
                'Hit_Eta_Pho1', 'Hit_Eta_Pho2',
                'Hit_Phi_Pho1', 'Hit_Phi_Pho2',
                'Hit_X_Pho1', 'Hit_X_Pho2',
                'Hit_Y_Pho1', 'Hit_Y_Pho2',
                'Hit_Z_Pho1', 'Hit_Z_Pho2',
                'RecHitEnPho1', 'RecHitEnPho2',
                'RecHitFracPho1', 'RecHitFracPho2',
                #'passLooseId', 'passMediumId', 'passTightId',
                'energy'],

    'gun_30M': ['nElectrons', 
                'Ele_Gen_E', 'Ele_Gen_Eta', 'Ele_Gen_Phi', 
                'Ele_SCRawE', 'eta', 'phi',
                'Ele_R9', 'Ele_HadOverEm', 'rho',
                'iEtaEle1', 'iEtaEle2',
                'iPhiEle1', 'iPhiEle2',
                'Hit_ES_Eta_Ele1', 'Hit_ES_Eta_Ele2',
                'Hit_ES_Phi_Ele1', 'Hit_ES_Phi_Ele2',
                'Hit_ES_X_Ele1', 'Hit_ES_X_Ele2',
                'Hit_ES_Y_Ele1', 'Hit_ES_Y_Ele2',
                'Hit_ES_Z_Ele1', 'Hit_ES_Z_Ele2',
                'ES_RecHitEnEle1', 'ES_RecHitEnEle2',
                'Hit_Eta_Ele1', 'Hit_Eta_Ele2',
                'Hit_Phi_Ele1', 'Hit_Phi_Ele2',
                'Hit_X_Ele1', 'Hit_X_Ele2',
                'Hit_Y_Ele1', 'Hit_Y_Ele2',
                'Hit_Z_Ele1', 'Hit_Z_Ele2',
                'RecHitEnEle1', 'RecHitEnEle2',
                'RecHitFracEle1', 'RecHitFracEle2',
                'passLooseId', 'passMediumId', 'passTightId',
                'energy_ecal_mustache'],
    'gun_v3': ['nElectrons', 
                'Ele_Gen_E', 'Ele_Gen_Eta', 'Ele_Gen_Phi', 
                'Ele_SCRawE', 'eta', 'phi',
                'Ele_R9', 'Ele_HadOverEm', 'rho',
                'iEtaEle1', 'iEtaEle2',
                'iPhiEle1', 'iPhiEle2',
                'Hit_ES_Eta_Ele1', 'Hit_ES_Eta_Ele2',
                'Hit_ES_Phi_Ele1', 'Hit_ES_Phi_Ele2',
                'Hit_ES_X_Ele1', 'Hit_ES_X_Ele2',
                'Hit_ES_Y_Ele1', 'Hit_ES_Y_Ele2',
                'Hit_ES_Z_Ele1', 'Hit_ES_Z_Ele2',
                'ES_RecHitEnEle1', 'ES_RecHitEnEle2',
                'Hit_Eta_Ele1', 'Hit_Eta_Ele2',
                'Hit_Phi_Ele1', 'Hit_Phi_Ele2',
                'Hit_X_Ele1', 'Hit_X_Ele2',
                'Hit_Y_Ele1', 'Hit_Y_Ele2',
                'Hit_Z_Ele1', 'Hit_Z_Ele2',
                'RecHitEnEle1', 'RecHitEnEle2',
                'RecHitFracEle1', 'RecHitFracEle2'],
    'Zee_data': ['nElectrons', 
                'Ele_SCRawE', 'eta', 'phi',
                'Ele_R9', 'Ele_HadOverEm', 'rho',
                'iEtaEle1', 'iEtaEle2',
                'iPhiEle1', 'iPhiEle2',
                'Hit_ES_Eta_Ele1', 'Hit_ES_Eta_Ele2',
                'Hit_ES_Phi_Ele1', 'Hit_ES_Phi_Ele2',
                'Hit_ES_X_Ele1', 'Hit_ES_X_Ele2',
                'Hit_ES_Y_Ele1', 'Hit_ES_Y_Ele2',
                'Hit_ES_Z_Ele1', 'Hit_ES_Z_Ele2',
                'ES_RecHitEnEle1', 'ES_RecHitEnEle2',
                'Hit_Eta_Ele1', 'Hit_Eta_Ele2',
                'Hit_Phi_Ele1', 'Hit_Phi_Ele2',
                'Hit_X_Ele1', 'Hit_X_Ele2',
                'Hit_Y_Ele1', 'Hit_Y_Ele2',
                'Hit_Z_Ele1', 'Hit_Z_Ele2',
                'RecHitEnEle1', 'RecHitEnEle2',
                'RecHitFracEle1', 'RecHitFracEle2',
                'energy_ecal_mustache'],
    'Zee_MC' : [#'nElectrons', 
                'Ele_Gen_E', 'Ele_Gen_Eta', 'Ele_Gen_Phi', 
                'Ele_SCRawE', 'eta', 'phi',
                'Ele_R9', 'Ele_HadOverEm', #'rho',
                'iEtaEle1', 'iEtaEle2',
                'iPhiEle1', 'iPhiEle2',
                'Hit_ES_Eta_Ele1', 'Hit_ES_Eta_Ele2',
                'Hit_ES_Phi_Ele1', 'Hit_ES_Phi_Ele2',
                'Hit_ES_X_Ele1', 'Hit_ES_X_Ele2',
                'Hit_ES_Y_Ele1', 'Hit_ES_Y_Ele2',
                'Hit_ES_Z_Ele1', 'Hit_ES_Z_Ele2',
                'ES_RecHitEnEle1', 'ES_RecHitEnEle2',
                'Hit_Eta_Ele1', 'Hit_Eta_Ele2',
                'Hit_Phi_Ele1', 'Hit_Phi_Ele2',
                'Hit_X_Ele1', 'Hit_X_Ele2',
                'Hit_Y_Ele1', 'Hit_Y_Ele2',
                'Hit_Z_Ele1', 'Hit_Z_Ele2',
                'RecHitEnEle1', 'RecHitEnEle2',
                'RecHitFracEle1', 'RecHitFracEle2',
                'energy_ecal_mustache'],

} 

varlists['Hgg_v4_matched'] = varlists['Hgg_v4'] + ['Pho_Gen_Eta', 'Pho_Gen_Phi', "Pho_Gen_E"]

gun_readcut = 'nElectrons>0'
gun_pho_readcut = 'nPhotons>0'
Zee_readcut = 'nElectrons==2'

readcuts = {
    'gun_30M': gun_readcut,
    'gun_v3': gun_readcut,

    'gun_v4_ele' : '(nElectrons>0) & (Ele_Gen_E[:,0]<300) & (Ele_Gen_E[:,0]>5)',
    'gun_v4_pho' : '(nPhotons>0) & (Pho_Gen_E[:,0]<300) & (Pho_Gen_E[:,0]>5)',

    'dR_ele' : '(nElectrons>0) & (Ele_Gen_E[:,0]<300) & (Ele_Gen_E[:,0]>5)',
    'dR_pho' : '(nPhotons>0) & (Pho_Gen_E[:,0]<300) & (Pho_Gen_E[:,0]>5)',

    'gjets' : '(nPhotons>1)',

    'Zee_data': Zee_readcut,
    'Zee_MC': Zee_readcut,
    'Zee_v4' : 'nElectrons==2',
    'Hgg_v4' : 'nPhotons==2',
    'Hgg_v4_matched' : 'nPhotons==2',

    'gun_pho' : gun_pho_readcut,

    'BDTvars' : gun_pho_readcut,

    'Hgg' : 'nPhotons==2',
}

def gun_savecut(result):
    return np.logical_and(result['Ele_Gen_E'] < 300, result['Ele_Gen_E'] > 5)

def gun_pho_savecut(result):
    return np.logical_and(result['Pho_Gen_E'] < 300, result['Pho_Gen_E'] > 5)

def Zee_savecut(result):
    return np.ones(result['phi'].shape, dtype=bool)


savecuts = {
    'gun_v4_ele' : Zee_savecut,
    'gun_v4_pho' : Zee_savecut,
    
    'Zee_v4' : Zee_savecut,
    'Hgg_v4' : Zee_savecut,
    'Hgg_v4_matched' : Zee_savecut,

    'dR_ele' : Zee_savecut,
    'dR_pho' : Zee_savecut,

    'gjets' : Zee_savecut,

    'gun_30M': gun_savecut,
    'gun_v3' : gun_savecut,

    'Zee_data': Zee_savecut,
    'Zee_MC': Zee_savecut,
    
    'gun_pho' : gun_pho_savecut,
    'BDTvars' : gun_pho_savecut,

    'Hgg' : Zee_savecut,
}

#one of:
#match: perform gen matching
#unmatch: perform inverted gen matching
#dR: compute dR but don't make any cuts on it
#none: don't perform any gen matching (used when gen information not present)
matching = { 
    'Zee_v4' : 'none',
    'Hgg_v4' : 'none',
    'Hgg_v4_matched' : 'dR',

    'gun_v4_ele' : 'match',
    'gun_v4_pho' : 'match',

    'dR_ele' : 'dR',
    'dR_pho' : 'dR',

    'gjets' : 'dR',

    'gun_30M' : 'match',
    'gun_v3' : 'match',

    'Zee_data' : 'none',
    'Zee_MC' : 'match',

    'gun_pho' : 'match',
    'BDTvars' : 'match',

    'Hgg': 'none',

}

isEle = {
    'Zee_v4' : True,
    'Hgg_v4' : False,
    'Hgg_v4_matched' : False,

    'gun_v4_ele' : True,
    'gun_v4_pho' : False,

    'dR_ele' : True,
    'dR_pho' : False,

    'gjets' : False,

    'gun_30M' : True,
    'gun_v3' : True,

    'Zee_data': True,
    'Zee_MC': True,

    'gun_pho' : False,
    'BDTvars' : False,

    'Hgg' : False,
}


class Extract:
    def __init__(self, outfolder, path, treeName='nTuplelize/T'):
        if path is not None:
            #path = '~/shared/nTuples/%s'%path
            self.tree = uproot.open("%s:%s"%(path, treeName))

        self.outfolder = outfolder

    def get_subdet(self):
        print("Getting subdet")

        t0 = time()
        with open("%s/Hit_Z.pickle"%self.outfolder, 'rb') as f:
            Z = pickle.load(f)
        print("\tLoaded Hit_Z in %0.2f seconds"%(time()-t0))
        t0 = time()
    
        subdet = np.abs(ak.to_numpy(ak.firsts(Z))) < 300

        print("dumping...")
        with open("%s/subdet.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(subdet,f)
        print('done')

    def build_localfeat(self, ES=False, scaled=False):
        if ES:
            print("Error: no local coords for ES")
            return 

        print("Building localfeat")
        t0 = time()
        with open("%s/iEta.pickle"%self.outfolder, 'rb') as f:
            iEta = pickle.load(f)
        print("\tLoaded iEta in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/iPhi.pickle"%self.outfolder, 'rb') as f:
            iPhi = pickle.load(f)
        print("\tLoaded iPhi in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/Hit_Z.pickle"%self.outfolder, 'rb') as f:
            Z = pickle.load(f)
        print("\tLoaded Hit_Z in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/RecHitEn.pickle"%self.outfolder, 'rb') as f:
            En = pickle.load(f)
        print("\tLoaded En in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/RecHitFrac.pickle"%self.outfolder, 'rb') as f:
            frac = pickle.load(f)
        print("\tLoaded Frac in %0.2f seconds"%(time()-t0))
        t0 = time()

        if ES:
            with open("%s/iEta.pickle"%self.outfolder, 'rb') as f:
                iEta = pickle.load(f)
            print("\tLoaded iEta in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/iPhi.pickle"%self.outfolder, 'rb') as f:
                iPhi = pickle.load(f)
            print("\tLoaded iPhi in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/Hit_Z.pickle"%self.outfolder, 'rb') as f:
                Z = pickle.load(f)
            print("\tLoaded Hit_Z in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/RecHitEn.pickle"%self.outfolder, 'rb') as f:
                En = pickle.load(f)
            print("\tLoaded En in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/RecHitFrac.pickle"%self.outfolder, 'rb') as f:
                frac = pickle.load(f)
            print("\tLoaded Frac in %0.2f seconds"%(time()-t0))
            t0 = time()


    
        lf = localfeat(iEta, iPhi, Z, En, frac)
        print("\tMake localfeat in %0.2f seconds"%(time()-t0))
        t0 = time()

        lf = torchify(lf)
        print("\tTorchified in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/localfeat.pickle"%self.outfolder, 'wb') as f:
            torch.save(lf, f, pickle_protocol=4)
        print("\tDumped in %0.2f seconds"%(time()-t0))

    def build_projfeat(self, ES=False, scaled=False):
        print("Building projfeat")
        t0 = time()
        with open("%s/Hit_Eta.pickle"%self.outfolder, 'rb') as f:
            Eta = pickle.load(f)
        print("\tLoaded Eta in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/Hit_Phi.pickle"%self.outfolder, 'rb') as f:
            Phi = pickle.load(f)
        print("\tLoaded Phi in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/Hit_Z.pickle"%self.outfolder, 'rb') as f:
            Z = pickle.load(f)
        print("\tLoaded Hit_Z in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/RecHitEn.pickle"%self.outfolder, 'rb') as f:
            En = pickle.load(f)
        print("\tLoaded En in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/RecHitFrac.pickle"%self.outfolder, 'rb') as f:
            frac = pickle.load(f)
        print("\tLoaded Frac in %0.2f seconds"%(time()-t0))
        t0 = time()

        if ES:
            with open("%s/Hit_ES_Eta.pickle"%self.outfolder, 'rb') as f:
                ES_Eta = pickle.load(f)
            print("\tLoaded ES_Eta in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/Hit_ES_Phi.pickle"%self.outfolder, 'rb') as f:
                ES_Phi = pickle.load(f)
            print("\tLoaded ES_Phi in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/Hit_ES_Z.pickle"%self.outfolder, 'rb') as f:
                ES_Z = pickle.load(f)
            print("\tLoaded ES_Z in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/ES_RecHitEn.pickle"%self.outfolder, 'rb') as f:
                ES_En = pickle.load(f)
            print("\tLoaded ES_En in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/ES_RecHitFrac.pickle"%self.outfolder, 'rb') as f:
                ES_frac = pickle.load(f)
            print("\tLoaded ES_Frac in %0.2f seconds"%(time()-t0))
            t0 = time()

            if scaled:
                ES_En = ES_En*3500
                fname = 'projfeat_ES_scaled'
            else:
                fname = 'projfeat_ES'

            ES = ak.ones_like(ES_Eta)
            ECAL = ak.ones_like(Eta)

            Eta = ak.concatenate( (Eta, ES_Eta), axis=1)
            Phi = ak.concatenate( (Phi, ES_Phi), axis=1)
            Z = ak.concatenate( (Z, ES_Z), axis=1)
            En = ak.concatenate( (En, ES_En), axis=1)
            frac = ak.concatenate( (frac, ES_frac), axis=1)
            det = ak.concatenate( (ECAL, ES), axis=1)
        else:
            fname = 'projfeat'
            det = None

        pf = projfeat(Eta, Phi, Z, En, frac, det)
        print("\tMake projfeat in %0.2f seconds"%(time()-t0))
        t0 = time()

        lf = torchify(lf)
        print("\tTorchified in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/%s.pickle"%(self.outfolder, fname), 'wb') as f:
            torch.save(pf, f, pickle_protocol=4)
        print("\tDumped in %0.2f seconds"%(time()-t0))

    def build_cartfeat(self, ES=False, scaled=False, graph_features = None):
        print("Building cartfeat")
        t0 = time()
        with open("%s/Hit_X.pickle"%self.outfolder, 'rb') as f:
            X = pickle.load(f)
        print("\tLoaded X in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/Hit_Y.pickle"%self.outfolder, 'rb') as f:
            Y = pickle.load(f)
        print("\tLoaded Y in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/Hit_Z.pickle"%self.outfolder, 'rb') as f:
            Z = pickle.load(f)
        print("\tLoaded Hit_Z in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/RecHitEn.pickle"%self.outfolder, 'rb') as f:
            En = pickle.load(f)
        print("\tLoaded En in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/RecHitFrac.pickle"%self.outfolder, 'rb') as f:
            frac = pickle.load(f)
        print("\tLoaded Frac in %0.2f seconds"%(time()-t0))
        t0 = time()

        if ES:
            with open("%s/Hit_ES_X.pickle"%self.outfolder, 'rb') as f:
                ES_X = pickle.load(f)
            print("\tLoaded ES_X in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/Hit_ES_Y.pickle"%self.outfolder, 'rb') as f:
                ES_Y = pickle.load(f)
            print("\tLoaded ES_Y in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/Hit_ES_Z.pickle"%self.outfolder, 'rb') as f:
                ES_Z = pickle.load(f)
            print("\tLoaded ES_Z in %0.2f seconds"%(time()-t0))
            t0 = time()

            with open("%s/ES_RecHitEn.pickle"%self.outfolder, 'rb') as f:
                ES_En = pickle.load(f)
            print("\tLoaded ES_En in %0.2f seconds"%(time()-t0))
            t0 = time()

            ES_frac = ak.ones_like(ES_En)

            if scaled:
                ES_En = ES_En*3500
                fname = 'cartfeat_ES_scaled'
            else:
                fname = 'cartfeat_ES'

            ES = ak.ones_like(ES_En)
            ECAL = ak.ones_like(En)

            X = ak.concatenate( (X, ES_X), axis=1)
            Y = ak.concatenate( (Y, ES_Y), axis=1)
            Z = ak.concatenate( (Z, ES_Z), axis=1)
            En = ak.concatenate( (En, ES_En), axis=1)
            frac = ak.concatenate( (frac, ES_frac), axis=1)
            det = ak.concatenate( (ECAL, ES), axis=1)
        else:
            det = None
            fname = 'cartfeat'

        graph_x = None
        if graph_features is not None:
            graph_x = []
            for var in graph_features:
                t0 = time()
                with open("%s/%s.pickle"%(self.outfolder, var), 'rb') as f:
                    graph_x.append(pickle.load(f))
                print("\tLoaded %s in %0.2f seconds"%(var, time()-t0))
                fname += "_%s"%var
            graph_x = np.concatenate(graph_x, 1)


        cf = cartfeat(X, Y, Z, En, frac, det)
        print("\tMake cartfeat in %0.2f seconds"%(time()-t0))
        t0 = time()

        cf = torchify(cf, graph_x)
        print("\tTorchified in %0.2f seconds"%(time()-t0))
        t0 = time()

        with open("%s/%s.pickle"%(self.outfolder, fname), 'wb') as f:
            torch.save(cf, f, pickle_protocol=4)
        print("\tDumped in %0.2f seconds"%(time()-t0))

    def build_feat_v4(self):
        variables = ['Hit_X', 'Hit_Y', 'Hit_Z', 'RecHitEn', 'RecHitFrac', 'HitNoise',
                     'RecHitFlag_kGood', 'RecHitFlag_kOutOfTime', 'RecHitFlag_kPoorCalib', 'RecHitGain',
                     'Hit_ES_X', 'Hit_ES_Y', 'Hit_ES_Z', 'ES_RecHitEn', 
                     'RecHitFlag_kESGood',
                     'rho', 'Ele_HadOverEm']
        data = []
        for var in variables:
            print("Reading",var)
            with open("%s/%s.pickle"%(self.outfolder, var), 'rb') as f:
                data.append(pickle.load(f))
        print("Making feats")
        feats = make_feat_v4(*data)
        print("Made feats")
        return feats

    def add_graph_features(self, coords, ES, scaled, graph_features):
        if type(graph_features)==str:
            graph_features = [graph_features]

        fname = "%s/%sfeat"%(self.outfolder, coords)
        if ES:
            fname += "_ES"
            if scaled:
                fname += "_scaled"

        print("Adding features",graph_features,"to",fname)

        graph_x = []
        suffix = ''
        for var in graph_features:
            t0 = time()
            with open("%s/%s.pickle"%(self.outfolder, var), 'rb') as f:
                graph_x += [pickle.load(f)]
            print("\tLoaded %s in %0.2f seconds"%(var, time()-t0))
            suffix += "_%s"%var
        if len(graph_x)==1:
            graph_x = graph_x[0]
        else:
            graph_x = np.stack(graph_x, 1)

        t0 = time()
        data = torch.load("%s.pickle"%fname)
        print("\tLoaded node features in %0.2f seconds"%(time()-t0))
        
        for d, gx in zip(data, graph_x):
            d.graph_x = torch.tensor(gx)

        fname+=suffix

        t0 = time()
        with open("%s.pickle"%fname, 'wb') as f:
            torch.save(data, f, pickle_protocol=4)
        print("\tDumped in %0.2f seconds"%(time()-t0))

        return data

    @staticmethod
    def gen_match(phigen, phireco, etagen, etareco, threshold=0.05):
        pairs, dRs = Extract.compute_dR(phigen, phireco, etagen, etareco)
        return pairs[dRs < threshold]

    @staticmethod
    def gen_unmatch(phigen, phireco, etagen, etareco, threshold=0.05):
        pairs, dRs = Extract.compute_dR(phigen, phireco, etagen, etareco)
        return pairs[dRs > threshold]

    @staticmethod
    def compute_dR(phigen, phireco, etagen, etareco):
        idxs = ak.argcartesian( (etagen, etareco), axis = 1)

        #awkward insists that I index the cartesitna product pairs with '0' and '1' rather than ints
        genetas = etagen[idxs[:,:,'0']]
        recoetas = etareco[idxs[:,:,'1']]

        genphis = phigen[idxs[:,:,'0']]
        recophis = phireco[idxs[:,:,'1']]

        #single_ele = ak.to_numpy(ak.num(phireco) == 1).nonzero()[0][0]
        #print("IDX",single_ele)
        #print("PHIGEN",phigen[single_ele])
        #print("PHIRECO",phireco[single_ele])
        #print("PAIR",idxs[single_ele])
        #print("GENPHI",genphis[single_ele])
        #print('REOPHI',recophis[single_ele])

        dphis = np.abs(genphis - recophis)
        gt = dphis > np.pi
        #you can't assign to awkward arrays in place, so this is an inefficient hack
        dphis = gt * (2*np.pi - dphis) + (1 - gt)*(dphis) 

        detas = np.abs(genetas - recoetas)

        dR2s = dphis*dphis + detas*detas

        #print("DR",np.sqrt(dR2s[single_ele]))
        
        #which reco particle?
        ele1 = idxs[:,:,'1'] == 0
        ele2 = idxs[:,:,'1'] == 1

        dR2ele1 = dR2s[ele1]
        dR2ele2 = dR2s[ele2]

        #print("DRele1", np.sqrt(dR2ele1[single_ele]))
        #print("DRele2", np.sqrt(dR2ele2[single_ele]))

        dR2ele1_minidx = ak.argmin(dR2ele1, 1, keepdims=True)
        dR2ele2_minidx = ak.argmin(dR2ele2, 1, keepdims=True)

        pairs_ele1 = idxs[ele1][dR2ele1_minidx]
        dR_ele1 = np.sqrt(dR2ele1[dR2ele1_minidx])        
        pairs_ele2 = idxs[ele2][dR2ele2_minidx]
        dR_ele2 = np.sqrt(dR2ele2[dR2ele2_minidx])        

        pairs = ak.concatenate( (pairs_ele1, pairs_ele2), -1)
        dR = ak.concatenate( (dR_ele1, dR_ele2), -1)
        
        #print("PAIR",pairs[single_ele])
        #print("DR",dR[single_ele])

        isnone = ak.is_none(pairs, 1)
        #print("ISNONE",isnone[single_ele])

        pairs = pairs[~isnone]
        dR = dR[~isnone]

        #print("PAIR",pairs[single_ele])
        #print("DR",dR[single_ele])

        #result is (gen, reco) index pairs, dR
        return pairs, dR


    def readfakes(self):
        reco  = ['Pho_Gen_Eta', 'Pho_Gen_Phi', 
                 'pt','eta', 'phi',
                 'energy', "Pho_SCRawE",
                 'Pho_R9']
        hits = ['Hit_X_Pho1', 'Hit_X_Pho2',
                    'Hit_Y_Pho1', 'Hit_Y_Pho2',
                    'Hit_Z_Pho1', 'Hit_Z_Pho2',
                    'Hit_ES_X_Pho1', 'Hit_ES_X_Pho2',
                    'Hit_ES_Y_Pho1', 'Hit_ES_Y_Pho2',
                    'Hit_ES_Z_Pho1', 'Hit_ES_Z_Pho2',
                    'ES_RecHitEnPho1', 'ES_RecHitEnPho2',
                    'RecHitEnPho1', 'RecHitEnPho2',
                    'RecHitFracPho1', 'RecHitFracPho2']

        varnames = reco+hits

        hits_trimmed = ['Hit_X_', 'Hit_Y_', 'Hit_Z_', 
                       'Hit_ES_X_', 'Hit_ES_Y_', 'Hit_ES_Z_',
                       'RecHitEn', 'RecHitFrac',
                       'ES_RecHitEn']

        arrs = self.tree.arrays(varnames, 'nPhotons==2')

        unmatched = self.gen_unmatch(arrs['Pho_Gen_Phi'], arrs['phi'], arrs['Pho_Gen_Eta'], arrs['eta'])
        Pho0 = unmatched[:,0]
        Pho1 = unmatched[:,1]

        result = {}

        reco = ['pt','eta','phi','energy', 'Pho_SCRawE']
        for var in reco:
            arrs[var] = ak.to_regular(arrs[var])
            result[var] = ak.to_numpy(ak.concatenate( (arrs[var][Pho0,0], arrs[var][Pho1,1])))

        #return result

        for var in hits_trimmed:
            Pho0Name = var+'Pho1'
            Pho1Name = var+'Pho2'
            bettername = var
            if var[-1]=='_':
                bettername = var[:-1]
            
            result[bettername] = ak.concatenate( (arrs[Pho0Name][Pho0], arrs[Pho1Name][Pho1]) )

        result['subdet'] = np.abs(ak.to_numpy(ak.firsts(result['Hit_Z']))) < 300

        print("Dumping...");
        for var in result.keys():
            t0 = time()
            varname = var
            with open("%s/%s.pickle"%(self.outfolder, varname), 'wb') as f:
                pickle.dump(result[var], f, protocol = 4)
            print("\tDumping %s took %f seconds"%(varname, time()-t0))

        return result

    def readHGCAL(self):
        print("Reading in HGCAL branches:")

        t0=time()
        print("Reading rechit_x...")
        Hit_X = self.tree['combined_rechit_x'].array()
        print("\ttook %0.3f seconds"%(time()-t0))
        
        t0=time()
        print("Reading rechit_y...")
        Hit_Y = self.tree['combined_rechit_y'].array()
        print("\ttook %0.3f seconds"%(time()-t0))
        
        t0=time()
        print("Reading rechit_z...")
        Hit_Z = self.tree['combined_rechit_z'].array()
        print("\ttook %0.3f seconds"%(time()-t0))
        
        t0=time()
        print("Reading rechit_energy...")
        recHitEn = self.tree['combined_rechits_energy'].array()
        print("\ttook %0.3f seconds"%(time()-t0))
        
        t0=time()
        print("Reading trueBeanEnergy...")
        trueE = self.tree['trueBeamEnergy'].array()
        print("\ttook %0.3f seconds"%(time()-t0))
        
        print()
        print("Building feature matrices...")
        
        t0 = time()
        cf = cartfeat_HGCAL(Hit_X, Hit_Y, Hit_Z, recHitEn)
        print("\tbuilding matrices took %0.3f seconds"%(time()-t0))
        
        t0 = time()
        cf = torchify(cf)  
        print("\tcasting to torch objects took %0.3f seconds"%(time()-t0))

        print()

        print("Building targets...")
        t0 = time()
        rawE = ak.sum(recHitEn, axis=1)

        ratio = trueE/rawE
        ratioflip = rawE/trueE 
        logratioflip = np.log(ratioflip)
        print("\tTook %0.3f seconds"%(time()-t0))

        print()

        print("Dumping:")
        t0=time()
        with open("%s/Hit_X.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(Hit_X, f, protocol = 4)
        print("\tDumped Hit_X in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/Hit_Y.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(Hit_Y, f, protocol = 4)
        print("\tDumped Hit_Y in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/Hit_Z.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(Hit_Z, f, protocol = 4)
        print("\tDumped Hit_Z in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/recHitEn.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(recHitEn, f, protocol = 4)
        print("\tDumped recHitEn in %0.3f seconds"%(time()-t0))
        
        t0=time()
        with open("%s/trueE.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(trueE, f, protocol = 4)
        print("\tDumped trueE in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/rawE.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(rawE, f, protocol = 4)
        print("\tDumped rawE in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/trueE_target.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(trueE, f, protocol = 4)
        print("\tDumped trueE target in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/ratio_target.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(ratio, f, protocol = 4)
        print("\tDumped ratio target in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/ratioflip_target.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(ratioflip, f, protocol = 4)
        print("\tDumped ratioflip target in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/logratioflip_target.pickle"%self.outfolder, 'wb') as f:
            pickle.dump(logratioflip, f, protocol = 4)
        print("\tDumped logratioflip target in %0.3f seconds"%(time()-t0))

        t0=time()
        with open("%s/cartfeat.pickle"%self.outfolder, 'wb') as f:
            torch.save(cf, f, pickle_protocol = 4)
        print("\tDumped features in %0.3f seconds"%(time()-t0))

        print()
        return    

    def read(self, kind, N=None, dR_thresh=0.05, start=0):
        varnames = varlists[kind]
        readcut = readcuts[kind]

        print()
        print()
        print()
        print("-"*40)
        print()
        #print(varnames)
        t0 = time()
        print("Reading in %s..."%kind)
        arrs = self.tree.arrays(varnames, readcut, entry_start=start, entry_stop = N)


        gen = []
        reco = []
        event = []
        hits = []

        result = {}

        for var in arrs.fields:
            if var[-1] == '1' or var[-1] == '2': #hit level information 
                if 'Noise' in var:
                    name = 'HitNoise'
                elif 'Quality' in var or 'Gain' in var:
                    name = var[:-1]
                else:
                    name = var[:-4]
                hits.append(name) 
            elif var[:7] == 'Ele_Gen' or var[:7] == 'Pho_Gen': #gen level information
                gen.append(var)
            elif var == 'rho' or var == 'nElectrons' or var == 'nPhotons': #event level information
                event.append(var)
            else: #reco level information
                reco.append(var) 

        print("\tio took %f seconds"%(time()-t0))

        if matching[kind] == 'match':
            t0=time()
            if isEle[kind]:
                matched_idxs = self.gen_match(arrs['Ele_Gen_Phi'], arrs['phi'],
                                              arrs['Ele_Gen_Eta'], arrs['eta'])
            else:
                matched_idxs = self.gen_match(arrs['Pho_Gen_Phi'], arrs['phi'],
                                              arrs['Pho_Gen_Eta'], arrs['eta'])
            gen_idxs = matched_idxs[:,:,'0']
            reco_idxs = matched_idxs[:,:,'1']
            print("\tgen matching took %f seconds"%(time()-t0))
        elif matching[kind] == 'unmatch':
            t0=time()
            if isEle[kind]:
                matched_idxs = self.gen_unmatch(arrs['Ele_Gen_Phi'], arrs['phi'],
                                              arrs['Ele_Gen_Eta'], arrs['eta'])
            else:
                matched_idxs = self.gen_unmatch(arrs['Pho_Gen_Phi'], arrs['phi'],
                                              arrs['Pho_Gen_Eta'], arrs['eta'])
            gen_idxs = matched_idxs[:,:,'0']
            reco_idxs = matched_idxs[:,:,'1']
            print("\tgen matching took %f seconds"%(time()-t0))
        elif matching[kind] == 'dR':
            t0=time()
            if isEle[kind]:
                matched_idxs, dR = self.compute_dR(arrs['Ele_Gen_Phi'], arrs['phi'],
                                              arrs['Ele_Gen_Eta'], arrs['eta'])
            else:
                matched_idxs, dR = self.compute_dR(arrs['Pho_Gen_Phi'], arrs['phi'],
                                              arrs['Pho_Gen_Eta'], arrs['eta'])
            gen_idxs = matched_idxs[:,:,'0']
            reco_idxs = matched_idxs[:,:,'1']

            arrs['dR'] = dR
            reco.append('dR')
            print("\tgen matching took %f seconds"%(time()-t0))


        if matching[kind] != 'none':
            t0 = time()
            
            for var in gen:
                arrs[var] = arrs[var][gen_idxs]

            for var in reco:
                arrs[var] = arrs[var][reco_idxs]

            print("\tapplying gen matching took %f seconds"%(time()-t0))

        t0 = time()

        #it can happen that there is exactly 1 reco electron, but it is identified as Ele2
        #I have no idea why, and it's super annoying, but here we are
        if not isEle[kind]:
            noEle1 = ak.num(arrs['iEtaPho1']) == 0
        else:
            noEle1 = ak.num(arrs['iEtaEle1']) == 0

        if matching[kind] != 'none':
            Ele1 = np.logical_and(reco_idxs == 0, ~noEle1) #recoidx==0 and there is an ele1
        else:
            Ele1 = ak.local_index(arrs['phi']) == 0

        Ele2 = ~Ele1  

        eventEle1 = ak.any(Ele1, axis=1)
        eventEle2 = ak.any(Ele2, axis=1)

        for var in gen + reco: #per-particle information, flattened 
            result[var] = ak.to_numpy(
                    ak.concatenate( 
                        (ak.flatten(arrs[var][Ele1]), 
                         ak.flatten(arrs[var][Ele2]) )))

        for var in event: #per-event information, broadcasted and flattened
            result[var] = ak.to_numpy(
                    ak.concatenate( 
                        (arrs[var][eventEle1], 
                         arrs[var][eventEle2]) ))

        for var in hits: #hit information, flattened
                         #note that this stays in awkward array format, while everything else is np
            if isEle[kind]:
                if 'Noise' in var:
                    nameEle1 = 'HitNoiseEle1'
                    nameEle2 = 'HitNoiseEle2'
                elif 'Quality' in var or 'Gain' in var:
                    nameEle1 = var+'1'
                    nameEle2 = var+'2'
                elif 'HitFlag' in var:
                    nameEle1 = var + 'ele1'
                    nameEle2 = var + 'ele2'
                else:
                    nameEle1 = var + 'Ele1'
                    nameEle2 = var + 'Ele2'
            else:
                if 'Noise' in var:
                    nameEle1 = 'HitNoisePho1'
                    nameEle2 = 'HitNoisePho2'
                elif 'Quality' in var or 'Gain' in var:
                    nameEle1 = var+'1'
                    nameEle2 = var+'2'
                elif 'HitFlag' in var:
                    nameEle1 = var + 'pho1'
                    nameEle2 = var + 'pho2'
                else:
                    nameEle1 = var + 'Pho1'
                    nameEle2 = var + 'Pho2'

            if var[-1] == '_':
                var = var[:-1]

            result[var] = ak.concatenate( (arrs[nameEle1][eventEle1], arrs[nameEle2][eventEle2]) )

        #for var in result.keys():
        #    print(var, len(result[var]))

        print("\tbroadcasting and flattening took %f seconds"%(time()-t0))

        t0 = time()

        eventEle1 = ak.to_numpy(eventEle1)
        eventEle2 = ak.to_numpy(eventEle2) 

        #event idx
        #usefuly mostly for troubleshooting
        result['eventidx']= np.concatenate( (eventEle1.nonzero()[0], eventEle2.nonzero()[0]) )

        #hit subdetector
        #1: barrel 0: endcaps
        result['subdet'] = np.abs(ak.to_numpy(ak.firsts(result['Hit_Z']))) < 300

        print("\tdetermening aux features took %f seconds"%(time()-t0))

        t0 = time()

        savecut = savecuts[kind](result)

        print("\tapplying savecut took %f seconds"%(time()-t0))

        print("Dumping...");
        for var in result.keys():
            t0 = time()
            varname = var
            with open("%s/%s.pickle"%(self.outfolder, varname), 'wb') as f:
                pickle.dump(result[var], f, protocol = 4)
            print("\tDumping %s took %f seconds"%(varname, time()-t0))

        t0 = time()


        if 'v4' not in kind and 'dR' not in kind:
            print("Building cartesian features..")
            cf = cartfeat(result['Hit_X'], result['Hit_Y'], result['Hit_Z'], result['RecHitEn'], result['RecHitFrac'])
            print("\tBuilding features took %f seconds"%(time()-t0))
            t0 = time()
            result['cartfeat'] = torchify(cf)
            print("\tTorchifying took %f seconds"%(time()-t0))
            t0 = time()
            with open("%s/cartfeat.pickle"%(self.outfolder), 'wb') as f:
                torch.save(result['cartfeat'], f, pickle_protocol = 4)
            print("\tDumping took %f seconds"%(time()-t0))

            print("Building projective features..")
            pf = projfeat(result['Hit_Eta'], result['Hit_Phi'], result['Hit_Z'], result['RecHitEn'], result['RecHitFrac'])
            print("\tBuilding features took %f seconds"%(time()-t0))
            t0 = time()
            result['projfeat'] = torchify(pf)
            print("\tTorchifying took %f seconds"%(time()-t0))
            t0 = time()
            with open("%s/projfeat.pickle"%(self.outfolder), 'wb') as f:
                torch.save(result['projfeat'], f, pickle_protocol = 4)
            print("\tDumping took %f seconds"%(time()-t0))

            print("Building local features..")
            lf = localfeat(result['iEta'], result['iPhi'], result['Hit_Z'], result['RecHitEn'], result['RecHitFrac'])
            print("\tBuilding features took %f seconds"%(time()-t0))
            t0 = time()
            result['localfeat'] = torchify(lf)
            print("\tTorchifying took %f seconds"%(time()-t0))
            t0 = time()
            with open("%s/localfeat.pickle"%(self.outfolder), 'wb') as f:
                torch.save(result['localfeat'], f, pickle_protocol = 4)
            print("\tDumping took %f seconds"%(time()-t0))

        print()
        return result 

import pandas as pd
import numpy as np
import scipy as sc

def get_centroid(r):
    d=r.child_dict
    a=np.array([list(d[k].coord) for k in d])
    return np.mean(a,axis=0)

def get_distances(r1,c2):
    d={(r1.get_id()[1],r1.get_resname(),r2.get_id()[1],r2.get_resname()) :sc.spatial.distance.euclidean(get_centroid(r1), get_centroid(r2), w=None) for r2 in c2.get_residues()}

    df=pd.Series(d).to_frame('distance between centroids')
    df.index.names=['position residue1','aa residue1','position residue2','aa residue2']
    return df.reset_index()
# SPDX-License-Identifier: GPL-3.0-only
"""Adaptation of Richardson et al. Sentinel-2 visibility reference inference.

Feature conventions follow yulunwu8/Optically-Shallow-Deep commit
56dfd1651be2bddf8323bc906e693879d5d99e4d (GPL-3.0). Original source, licence
and weights are retained under the recovery experiment's external directory.
This adapter loads dense weights into PyTorch; it does not import TensorFlow.
Probabilities are not locally calibrated depth limits.
"""
import json
import hashlib
import numpy as np
import h5py
import torch
from scipy.ndimage import uniform_filter,binary_dilation
from scipy.special import expit

FEATURES=((1,15,'sd'),(2,3,'sd'),(2,7,'avg'),(2,9,'avg'),(2,11,'sd'),(2,15,'sd'),
          (3,5,'avg'),(3,11,'sd'),(3,13,'avg'),(3,13,'sd'),(3,15,'sd'),
          (4,13,'sd'),(4,15,'sd'),(7,9,'sd'),(7,13,'sd'),(7,15,'sd'),
          (9,9,'avg'),(9,15,'sd'))
BANDS=('coastal','blue','green','red','rededge1','rededge2','rededge3','nir','nir08','swir16','swir22')


class PublishedVisibility(torch.nn.Module):
    def __init__(self,path):
        super().__init__()
        self.layers=torch.nn.ModuleList()
        self.activations=[]
        self.reference_weights=[]
        with h5py.File(path,'r') as f:
            config=json.loads(f.attrs['model_config'])
            for layer in config['config']['layers']:
                if layer['class_name']!='Dense':continue
                spec=layer['config'];name=spec['name']
                group=f['model_weights'][name][name]
                kernel,bias=group['kernel:0'][:],group['bias:0'][:]
                dense=torch.nn.Linear(kernel.shape[0],kernel.shape[1])
                dense.weight.data.copy_(torch.from_numpy(kernel.T))
                dense.bias.data.copy_(torch.from_numpy(bias))
                activation=spec['activation']
                if isinstance(activation,dict):
                    if activation['class_name']!='LeakyReLU':raise ValueError('Unsupported activation')
                    activation=('leaky',float(activation['config']['alpha']))
                elif activation=='sigmoid':activation=('sigmoid',0)
                else:raise ValueError('Unsupported activation')
                self.layers.append(dense);self.activations.append(activation)
                self.reference_weights.append((kernel,bias,activation))
        self.source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()

    def forward(self,x):
        for layer,(activation,alpha) in zip(self.layers,self.activations):
            x=layer(x)
            x=torch.nn.functional.leaky_relu(x,alpha) if activation=='leaky' else torch.sigmoid(x)
        return x

    def numerical_check(self):
        values=np.random.RandomState(7).uniform(-.02,.3,(256,20)).astype('float32')
        reference=values.copy()
        for kernel,bias,(activation,alpha) in self.reference_weights:
            reference=reference@kernel+bias
            reference=np.where(reference>=0,reference,alpha*reference) if activation=='leaky' else expit(reference)
        with torch.no_grad():pred=self(torch.from_numpy(values)).numpy()
        difference=float(np.max(np.abs(reference-pred)))
        if difference>2e-5:raise AssertionError(f'Dense conversion mismatch: {difference}')
        return {'max_abs_probability_difference_numpy_vs_torch':difference,
                'weights_sha256':self.source_sha256,'tensorflow_runtime_used':False,
                'scope':'documented HDF5 dense architecture equivalence; not a full original-runtime pipeline comparison'}


def features_at_points(image,rows,cols,mgrs='43PFK',baseline=5.1):
    """Native10m DN features, including the reference's geographic convention.

    Require complete15x15 native neighborhoods; unsupported nodata neighborhoods
    are flagged rather than imputed. Source edge-specific inference is not used.
    Preserve the published longitude convention, even though its sign differs
    from geographic longitude for this UTM zone. Altering it would change inputs.
    """
    image=image.astype('float32')
    correction=1000 if baseline<4 else 0
    image+=correction
    rows=np.asarray(rows,dtype=int);cols=np.asarray(cols,dtype=int)
    n=len(rows)
    inside=(rows>=8)&(cols>=8)&(rows<image.shape[1]-8)&(cols<image.shape[2]-8)
    safe_rows=np.clip(rows,0,image.shape[1]-1);safe_cols=np.clip(cols,0,image.shape[2]-1)
    values=np.zeros((n,20),dtype='float32')
    latitude={'C':-76,'D':-68,'E':-60,'F':-52,'G':-44,'H':-36,'J':-28,'K':-20,'L':-12,
              'M':-4,'N':4,'P':12,'Q':20,'R':28,'S':36,'T':44,'U':52,'V':60,'W':68,'X':76}[mgrs[2]]
    values[:,0]=int(180-(int(mgrs[:2])-.5)*6)
    values[:,1]=abs(latitude)
    complete=np.all((image>correction)&(image<30000),axis=0)
    # Any invalid neighborhood is outside this adapter's exact fast-path scope.
    neighborhood=uniform_filter(complete.astype('float32'),15,mode='constant')>.9999
    valid=inside&neighborhood[safe_rows,safe_cols]
    centre=image[:,safe_rows,safe_cols]
    valid&=(centre[9]<1500)&((centre[2]-centre[7])/(centre[2]+centre[7]+1e-8)>0)
    valid&=((centre[2]-centre[9])/(centre[2]+centre[9]+1e-8)<.42)
    for i,(band,size,kind) in enumerate(FEATURES,2):
        data=image[band]
        if kind=='avg':filtered=uniform_filter(data,size,mode='mirror')
        else:
            mean=uniform_filter(data,size)
            # Match the reference's integer second-moment calculation/quantization.
            second=uniform_filter(data.astype('int32')**2,size)
            filtered=np.sqrt(np.maximum(second-mean**2,0))
        filtered=np.clip(filtered,0,65535).astype('uint16')
        values[:,i]=filtered[safe_rows,safe_cols]
    return (values/10000).astype('float16').astype('float32'),valid

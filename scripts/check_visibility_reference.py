# SPDX-License-Identifier: GPL-3.0-only
"""Compare our adapter against inspected pure functions of the author source."""
from pathlib import Path
import ast
import gc
import json
import sys
import warnings
import numpy as np
from scipy.ndimage import uniform_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.visibility_reference import FEATURES,features_at_points

ROOT=Path('runs/2026-11-06/recovery_v2')
source=ROOT/'external/optically_shallow_deep/opticallyshallowdeep/process_as_strips.py'
# Only these inspected numerical functions are executed; no downloaded imports,
# networking, filesystem access, package setup or TensorFlow runtime is executed.
names={'process_image_with_filters','apply_filter','std_dev_filter','get_location_values','get_mean_longitude'}
tree=ast.parse(source.read_text())
selected=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
namespace={'np':np,'uniform_filter':uniform_filter,'warnings':warnings,'gc':gc}
exec(compile(selected,str(source),'exec'),namespace)
rng=np.random.RandomState(16)
image=rng.randint(1001,2800,(11,48,48)).astype('uint16')
rows=np.array([12,20,30]);cols=np.array([13,22,31])
columns=[['long'],['lat_abs']]+[list(spec) for spec in FEATURES]
reference=namespace['process_image_with_filters'](image.transpose(1,2,0).copy(),columns)
location,_=namespace['get_location_values']('43PFK')
raw=np.column_stack([np.full(len(rows),location[0]),np.full(len(rows),location[2]),reference[rows,cols]])
expected=(raw/10000).astype('float16').astype('float32')
actual,_=features_at_points(image,rows,cols)
np.testing.assert_array_equal(expected,actual)
result={'tested_points':len(rows),'tested_features':20,'max_abs_feature_difference':float(abs(expected-actual).max()),
        'scope':'native interior fast-path features; inspected original numerical functions; no original TensorFlow runtime',
        'geography_convention_preserved':location,'local_classification_accuracy_verified':False}
(ROOT/'visibility/feature_reference_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

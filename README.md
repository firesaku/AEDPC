# AEDPC algorithm

All implementation code is in **AEDPC.py**. It contains the proposed AEDPC
algorithm and the helpers needed for its density estimation, parent selection,
local geometry, contact graph and certified reconstruction. Internal shape and
stability tests are part of the model. No comparison-algorithm implementations,
experiment runners or result-analysis code are included.

Use Python 3.12. Install the external dependencies, then place AEDPC.py beside
your script or add its directory to Python's import path:

```sh
python -m pip install -r requirements.txt
```

```python
from AEDPC import AEDPC, AEDPCConfig

config = AEDPCConfig(neighbor_backend="exact")
model = AEDPC(alpha_n=0.01, alpha_m=0.05, config=config)
labels = model.fit_predict(X)  # X: samples-by-features array
```

`AEDPC` aliases `AutomaticAEDPCv2`, the automatic parent/fusion estimator.
`AEDPCCore` (also named `AEDPCv2`) is the configurable core used with the
dedicated STL10 settings. `AEDPCConfig` aliases `AEDPCv2Config`.
Pass alpha_n and alpha_m explicitly to the estimator; they override the
corresponding configuration fields. See the manuscript for reported settings.

The implementation retains the PAk path used in the paper. Third-party
dependencies are installed separately; their source code is not bundled.
DADApy and hnswlib require compatible wheels or a compiler toolchain.

Project repository: https://github.com/firesaku/AEDPC

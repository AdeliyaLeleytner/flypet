_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])  # repository root
import sys, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
od = pd.read_csv(_ROOT + "/data/door/odor.csv", sep=";")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
keys, fps = [], []
for k, s in zip(od.InChIKey.astype(str), od.SMILES.astype(str)):
    m = Chem.MolFromSmiles(s) if s not in ("nan", "SFR", "") else None
    if m is None:
        continue
    keys.append(k)
    fps.append(gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
np.savez(sys.argv[1], keys=np.array(keys), X=np.stack(fps))
print("fingerprints", len(keys))

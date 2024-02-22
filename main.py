import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = Path('/workspace/dlmi')
    
    annots = pd.read_csv(path / "clinical_annotation.csv")
    annots = annots.sort_values("ID")
    
    plt.scatter(annots["LYMPH_COUNT"], annots["LABEL"])
    print(np.count_nonzero(annots[annots["LABEL"] == -1]["LYMPH_COUNT"] > np.max(annots[annots["LABEL"] == 0]["LYMPH_COUNT"])))
    plt.xscale('log')
    plt.savefig("corr.png")

# architecture:  all as layers? but not same number
# rnn images jusqu'a assez confidents
# run img on each separately and threshold malignant
# how to train: take existing model

main()
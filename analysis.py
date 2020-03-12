from ray.tune import Analysis
import pandas as pd
import os
import numpy as np

if __name__ == "__main__":
    analysis = Analysis(
        "/Users/shaobohu/Documents/我的坚果云/project/circles_experiment/TRY_ALL/Train")
    print(sorted(analysis.dataframe()['acc'].tolist()))
    print(analysis.get_best_config('acc', 'max'))

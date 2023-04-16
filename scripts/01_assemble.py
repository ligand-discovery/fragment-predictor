import pandas as pd
import os

root = os.path.dirname(os.path.abspath(__file__))

df = None

for i in range(26):
    fn = "predictions_cache_{0}.csv".format(i)
    df_ = pd.read_csv(os.path.join(root, "..", "data", fn))
    if df is None:
        df = df_
    else:
        df = pd.concat([df, df_])

df.to_csv(os.path.join(root, "..", "data", "predictions.csv"), index=False)

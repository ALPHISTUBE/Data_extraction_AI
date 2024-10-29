import pandas as pd


path= "category.csv"
df = pd.read_csv(path)

data = df["category"]
category = []
for dt in data:
    has = False
    for ct in category:
        if dt == ct:
            has = True
            break
    if not has:
        print(dt)
        category.append(dt)
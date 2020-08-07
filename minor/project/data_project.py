import pandas as pd
from sklearn.utils import shuffle

df1 = pd.read_csv("entertainment_news.csv")
df2 = pd.read_csv("tech_news.csv")
df3 = pd.read_csv("india_life_sports.csv")
df1 = df1.drop_duplicates(subset = 'headline')
df2 = df2.drop_duplicates(subset = 'headline')
df3 = df3.drop_duplicates(subset = 'headline')
df = pd.concat([df1,df2,df3],ignore_index = True)

df = shuffle(df)

df = df.iloc[:,:].values
cat = {"india","entertainment","lifestyle","sports","technology"}

arr = []
for i in range(len(df)):
    if df[i][0] in cat:
        arr.append(df[i])


arr = pd.DataFrame(arr)

arr = arr.drop_duplicates(subset = 2)

arr = arr.dropna()

header = list(df1.columns)
arr.to_csv("news_data.csv",header = header)
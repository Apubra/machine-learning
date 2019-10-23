from nltk.stem.snowball import SnowballStemmer
stremmer = SnowballStemmer('english')
res = stremmer.stem('responsiveness')
print(res)
res = stremmer.stem('responsivity')
print(res)
res = stremmer.stem('unresponsive')
print(res)

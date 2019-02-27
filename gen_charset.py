# coding: utf-8
import pandas as pd
def gen_charset():
    data = pd.read_csv("sentences.csv", header=None)
    charset = "".join([ str(x) for x in data[1]])
    a = "".join([x for x in set(charset)])
    return a

def gen_onehot_matrix():
    charset = gen_charset()
    data = pd.read_csv("sentences.csv", header=None)
    matrix = []
    for x in data[1]:
        tmp = [charset.index(str(c)) for c in str(x)]
        matrix.append([1 if c in tmp else 0 for c in range(len(charset))])

if __name__ == "__main__":
    gen_charset()
    

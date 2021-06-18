from training_model import *
import numpy as np

def text_to_array(a):
    b = []
    for x in a:
        b.append(format(ord(x),'b'))
    c = np.zeros(shape=(len(b),8))
    
    for i in range(len(b)):
        for j in range(len(b[i])):
            c[i][j+1] = b[i][j]
    
    c = c.astype(int)
    return c

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

def array_to_text(c):
    f = []
    
    for i in range(len(c)):
            f.append("".join(str(c[i][j]) for j in range(1,8)))
    
    e = "".join(text_from_bits(i) for i in f)
    e.replace("@"," ")
    return e

f = open("D:/Practice/ANC/pt.txt","r")
a = f.read()
a = a[:len(a)-2]

alice_pred = alicemodel.predict([text_to_array(a), k_batch], None)

alice_pred1 = np.around(alice_pred)
alice_pred1 = alice_pred1.astype(int)
alice_pred1 = array_to_text(alice_pred1)
alice_pred1 = alice_pred1.replace("@"," ")

f = open("D:/Practice/ANC/ct.txt", "w")
f.write(alice_pred1)
f.close()
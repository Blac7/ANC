from training_model import *
from alice import *

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

bob_pred = bobmodel.predict([alice_pred, k_batch], None)
bob_pred = np.around(bob_pred)
bob_pred = bob_pred.astype(int)
bob_pred = array_to_text(bob_pred)
bob_pred = bob_pred.replace("@"," ")

f = open("D:/Practice/ANC/pt_bob.txt", "w")
f.write(bob_pred)
f.close()
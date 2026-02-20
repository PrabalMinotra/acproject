import itertools

def ROL(x, k, word_size=16):
    return ((x << k) & 0xffff) | (x >> (16 - k))

def ROR(x, k, word_size=16):
    return ((x >> k) & 0xffff) | ((x << (16 - k)) & 0xffff)

def encrypt(pt, key, z_seq, m, T, reverse_key, reverse_z, reverse_pt, reverse_ct):
    k = [0] * T
    for i in range(m):
        if reverse_key:
            k[i] = (key >> ((m - 1 - i) * 16)) & 0xffff
        else:
            k[i] = (key >> (i * 16)) & 0xffff
            
    for i in range(m, T):
        tmp = ROR(k[i-1], 3)
        if m == 4:
            tmp ^= k[i-3]
        tmp ^= ROR(tmp, 1)
        z_bit = z_seq[61 - ((i - m) % 62)] if reverse_z else z_seq[(i - m) % 62]
        k[i] = (~k[i-m] & 0xffff) ^ tmp ^ z_bit ^ 3
        
    x = (pt >> 16) & 0xffff
    y = pt & 0xffff
    
    if reverse_pt:
        x, y = y, x

    for i in range(T):
        tmp = (ROL(x, 1) & ROL(x, 8)) ^ ROL(x, 2)
        y, x = y ^ tmp ^ k[i], x

    if reverse_ct:
        x, y = y, x
        
    return (x << 16) | y

z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,
      1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

key = 0x1918111009080100
pt = 0x65656877
expct_ct = 0xc69be9bb

found = False
for rev_k, rev_z, rev_p, rev_c in itertools.product([False, True], repeat=4):
    ct = encrypt(pt, key, z0, 4, 32, rev_k, rev_z, rev_p, rev_c)
    if ct == expct_ct:
        print(f"Match! rev_k={rev_k}, rev_z={rev_z}, rev_p={rev_p}, rev_c={rev_c}")
        found = True

if not found:
    print("No match found.")

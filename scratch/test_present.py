from src.ciphers.present import PresentCipher

p = PresentCipher(0x00000000000000000000)

Sbox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
PBox = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]

def enc(pt):
    state = pt
    for i in range(31):
        state ^= p.key_schedule[i]
        
        
        new_state = 0
        for j in range(16):
            nibble = (state >> (j*4)) & 0xF
            new_state |= (Sbox[nibble] << (j*4))
        state = new_state
        
        
        p_state = 0
        for j in range(64):
            bit = (state >> j) & 1
            p_state |= (bit << PBox[j])
        state = p_state
        
    state ^= p.key_schedule[31]
    return state

print(hex(enc(0)))

from __future__ import print_function
from collections import deque

class SimonCipher(object):
    """Simon 32/64 Block Cipher Object"""
    z0 = 0b01100111000011010100100010111110110011100001101010010001011111

    def __init__(self, key):
        self.block_size = 32
        self.word_size = 16
        self.key_size = 64
        self.rounds = 32
        self.zseq = self.z0
        
        self.mod_mask = (2 ** self.word_size) - 1
        self.key = key & ((2 ** self.key_size) - 1)

        m = self.key_size // self.word_size
        self.key_schedule = []

        k_init = [((self.key >> (self.word_size * ((m-1) - x))) & self.mod_mask) for x in range(m)]
        k_reg = deque(k_init)
        round_constant = self.mod_mask ^ 3

        for x in range(self.rounds):
            rs_3 = ((k_reg[0] << (self.word_size - 3)) + (k_reg[0] >> 3)) & self.mod_mask
            if m == 4:
                rs_3 = rs_3 ^ k_reg[2]
            rs_1 = ((rs_3 << (self.word_size - 1)) + (rs_3 >> 1)) & self.mod_mask
            c_z = ((self.zseq >> (x % 62)) & 1) ^ round_constant
            new_k = c_z ^ rs_1 ^ rs_3 ^ k_reg[m - 1]
            self.key_schedule.append(k_reg.pop())
            k_reg.appendleft(new_k)

    def encrypt(self, plaintext, rounds=None):
        b = (plaintext >> self.word_size) & self.mod_mask
        a = plaintext & self.mod_mask

        limit = rounds if rounds is not None else self.rounds
        
        x = b
        y = a 

        for k in self.key_schedule[:limit]:
            ls_1_x = ((x >> (self.word_size - 1)) + (x << 1)) & self.mod_mask
            ls_8_x = ((x >> (self.word_size - 8)) + (x << 8)) & self.mod_mask
            ls_2_x = ((x >> (self.word_size - 2)) + (x << 2)) & self.mod_mask

            xor_1 = (ls_1_x & ls_8_x) ^ y
            xor_2 = xor_1 ^ ls_2_x
            y = x
            x = k ^ xor_2
            
        return (x << self.word_size) + y

if __name__ == "__main__":
    w = SimonCipher(0x1918111009080100)
    t = w.encrypt(0x65656877)
    print(f"PT: 0x65656877 | Key: 0x1918111009080100 | CT: {hex(t)}")
    assert t == 0xc69be9bb, "Test vector failed!"
    print("Test passed!")

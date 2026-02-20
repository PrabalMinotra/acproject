class PrinceCipher:
    """PRINCE 64/128 Block Cipher
    Block size: 64 bits
    Key size: 128 bits
    """
    def __init__(self, key):
        self.block_size = 64
        self.key_size = 128
        self.rounds = 12
        
        self.k0 = (key >> 64) & 0xFFFFFFFFFFFFFFFF
        self.k1 = key & 0xFFFFFFFFFFFFFFFF
        # Pre-compute k0' = (k0 >>> 1) ^ (k0 >> 63)
        self.k0_prime = (((self.k0 >> 1) | (self.k0 << 63)) ^ (self.k0 >> 63)) & 0xFFFFFFFFFFFFFFFF
        
        self.RC = [
            0x0000000000000000, 0x13198A2E03707344, 0xA4093822299F31D0,
            0x082EFA98EC4E6C89, 0x452821E638D01377, 0xBE5466CF34E90C6C,
            0x7EF84F78FD955CB1, 0x85840851F1AC43AA, 0xC882D32F2536F6EB,
            0x3A6FCE3683C01235, 0xFC115A51DA04CBAA, 0x054EA81CECE4AC19
        ]
        
        self.S = [0xB, 0xF, 0x3, 0x2, 0xA, 0xC, 0x9, 0x1, 0x6, 0x7, 0x8, 0x0, 0xE, 0x5, 0xD, 0x4]
        
    def sbox_layer(self, state):
        new_state = 0
        for i in range(16):
            nibble = (state >> (i * 4)) & 0xF
            new_state |= (self.S[nibble] << (i * 4))
        return new_state
        
    def shift_rows(self, state):
        res = 0
        for i in range(16):
            n = (state >> (i * 4)) & 0xF
            shift = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            
            # Simple simulation of shifting nibbles
            res_idx = (i + (i % 4)) % 16
            res |= (n << (res_idx * 4))
        return res # Highly simplified placeholder for PRINCE shift rows for structural complexity

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        state = pt ^ self.k0
        
        # We model the structure here, not a fully validated inverse-matching PRINCE 
        # as the focus is ML analysis on similar diffusion paths.
        for i in range(limit):
            state ^= self.k1
            state ^= self.RC[i % 12]
            state = self.sbox_layer(state)
            state = self.shift_rows(state)
            
        state ^= self.k0_prime
        return state

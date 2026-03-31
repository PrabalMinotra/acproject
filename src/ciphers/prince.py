class PrinceCipher:
    

    def __init__(self, key):
        self.block_size = 64
        self.key_size = 128
        self.rounds = 12

        self.k0 = (key >> 64) & 0xFFFFFFFFFFFFFFFF
        self.k1 = key & 0xFFFFFFFFFFFFFFFF
        
        self.k0_prime = (((self.k0 >> 1) | (self.k0 << 63)) ^ (self.k0 >> 63)) & 0xFFFFFFFFFFFFFFFF

        self.RC = [
            0x0000000000000000, 0x13198A2E03707344, 0xA4093822299F31D0,
            0x082EFA98EC4E6C89, 0x452821E638D01377, 0xBE5466CF34E90C6C,
            0x7EF84F78FD955CB1, 0x85840851F1AC43AA, 0xC882D32F2536F6EB,
            0x3A6FCE3683C01235, 0xFC115A51DA04CBAA, 0x054EA81CECE4AC19
        ]

        self.S = [0xB, 0xF, 0x3, 0x2, 0xA, 0xC, 0x9, 0x1, 0x6, 0x7, 0x8, 0x0, 0xE, 0x5, 0xD, 0x4]
        self.S_INV = [0xB, 0x7, 0x3, 0x2, 0xF, 0xD, 0x8, 0x9, 0xA, 0x6, 0x4, 0x0, 0x5, 0xE, 0xC, 0x1]

    def _split_nibbles(self, state):
        return [(state >> (4 * i)) & 0xF for i in range(16)]

    def _join_nibbles(self, nibbles):
        state = 0
        for i, n in enumerate(nibbles):
            state |= (n & 0xF) << (4 * i)
        return state & 0xFFFFFFFFFFFFFFFF

    def _sbox_layer(self, state, sbox):
        nibbles = self._split_nibbles(state)
        nibbles = [sbox[n] for n in nibbles]
        return self._join_nibbles(nibbles)

    def _shift_rows(self, state):
        nibbles = self._split_nibbles(state)
        out = [0] * 16
        for r in range(4):
            for c in range(4):
                src_idx = r + 4 * c
                dst_c = (c + r) % 4
                dst_idx = r + 4 * dst_c
                out[dst_idx] = nibbles[src_idx]
        return self._join_nibbles(out)

    def _shift_rows_inv(self, state):
        nibbles = self._split_nibbles(state)
        out = [0] * 16
        for r in range(4):
            for c in range(4):
                src_idx = r + 4 * c
                dst_c = (c - r) % 4
                dst_idx = r + 4 * dst_c
                out[dst_idx] = nibbles[src_idx]
        return self._join_nibbles(out)

    def _mprime_layer(self, state):
        nibbles = self._split_nibbles(state)
        out = [0] * 16
        for c in range(4):
            col = [nibbles[r + 4 * c] for r in range(4)]
            t = col[0] ^ col[1] ^ col[2] ^ col[3]
            out[0 + 4 * c] = t ^ col[0]
            out[1 + 4 * c] = t ^ col[1]
            out[2 + 4 * c] = t ^ col[2]
            out[3 + 4 * c] = t ^ col[3]
        return self._join_nibbles(out)

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        state = pt ^ self.k0

        
        if limit < self.rounds:
            state ^= self.k1
            state ^= self.RC[0]
            for i in range(1, limit + 1):
                state = self._sbox_layer(state, self.S)
                state = self._mprime_layer(state)
                state = self._shift_rows(state)
                state ^= self.k1
                state ^= self.RC[i]
            state ^= self.k0_prime
            return state & 0xFFFFFFFFFFFFFFFF

        
        state ^= self.k1
        state ^= self.RC[0]
        for i in range(1, 6):
            state = self._sbox_layer(state, self.S)
            state = self._mprime_layer(state)
            state = self._shift_rows(state)
            state ^= self.k1
            state ^= self.RC[i]

        state = self._sbox_layer(state, self.S)
        state = self._mprime_layer(state)
        state = self._sbox_layer(state, self.S_INV)

        for i in range(6, 11):
            state ^= self.RC[i]
            state ^= self.k1
            state = self._shift_rows_inv(state)
            state = self._mprime_layer(state)
            state = self._sbox_layer(state, self.S_INV)

        state ^= self.RC[11]
        state ^= self.k1
        state ^= self.k0_prime
        return state & 0xFFFFFFFFFFFFFFFF

if __name__ == "__main__":
    key = 0x000102030405060708090A0B0C0D0E0F
    pt = 0x0123456789ABCDEF
    ct = PrinceCipher(key).encrypt(pt)
    assert ct == 0x63870EEB73889FE8, "PRINCE regression test failed (self-generated)!"
    print(f"PRINCE regression test passed: 0x{ct:x}")

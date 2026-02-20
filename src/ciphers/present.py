class PresentCipher:
    """PRESENT 64/80 Block Cipher"""
    def __init__(self, key):
        self.block_size = 64
        self.key_size = 80
        self.rounds = 31

        self.Sbox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
        self.PBox = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
                     4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
                     8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                     12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]

        self.key_schedule = []
        state = key & ((1 << 80) - 1)
        for i in range(1, self.rounds + 2):
            self.key_schedule.append(state >> 16)
            state = ((state & ((1 << 19) - 1)) << 61) | (state >> 19)
            state = (self.Sbox[state >> 76] << 76) | (state & ((1 << 76) - 1))
            state ^= (i << 15)

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        state = pt & ((1 << 64) - 1)

        for i in range(limit):
            state ^= self.key_schedule[i]
            
            # SBox
            new_state = 0
            for j in range(16):
                nibble = (state >> (j * 4)) & 0xF
                new_state |= (self.Sbox[nibble] << (j * 4))
            state = new_state
            
            # PBox
            p_state = 0
            for j in range(64):
                bit = (state >> j) & 1
                p_state |= (bit << self.PBox[j])
            state = p_state
            
        # Post-whitening if doing all rounds. For reduced rounds, we just return the state.
        if limit == self.rounds:
            state ^= self.key_schedule[self.rounds]

        return state

if __name__ == "__main__":
    p = PresentCipher(0x00000000000000000000)
    t = p.encrypt(0x0000000000000000)
    assert t == 0x5579C1387B228445, "PRESENT Test failed!"
    print(f"PRESENT64/80 works: 0x{t:x}")

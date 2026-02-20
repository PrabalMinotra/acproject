class SpeckCipher(object):
    """Speck 32/64 Block Cipher Object"""

    def __init__(self, key):
        self.block_size = 32
        self.word_size = 16
        self.key_size = 64
        self.rounds = 22
        
        self.mod_mask = (1 << self.word_size) - 1
        
        # Validations
        m = 4
        self.beta_shift = 3
        self.alpha_shift = 7

        k = [(key >> (16 * i)) & self.mod_mask for i in range(m)]

        self.key_schedule = [k[0]]
        l_schedule = [k[i] for i in range(1, m)]

        for i in range(self.rounds - 1):
            new_l = ((self.ROR(l_schedule[i], self.alpha_shift) + self.key_schedule[i]) & self.mod_mask) ^ i
            new_k = self.ROL(self.key_schedule[i], self.beta_shift) ^ new_l

            l_schedule.append(new_l)
            self.key_schedule.append(new_k)

    def ROL(self, x, amount):
        return ((x << amount) & self.mod_mask) | (x >> (self.word_size - amount))

    def ROR(self, x, amount):
        return ((x >> amount) & self.mod_mask) | ((x << (self.word_size - amount)) & self.mod_mask)

    def encrypt(self, plaintext, rounds=None):
        x = (plaintext >> self.word_size) & self.mod_mask
        y = plaintext & self.mod_mask

        limit = rounds if rounds is not None else self.rounds

        for i in range(limit):
            x = ((self.ROR(x, self.alpha_shift) + y) & self.mod_mask) ^ self.key_schedule[i]
            y = self.ROL(y, self.beta_shift) ^ x

        return (x << self.word_size) + y

if __name__ == "__main__":
    w = SpeckCipher(0x1918111009080100)
    t = w.encrypt(0x6574694c)
    assert t == 0xa86842f2, "Test vector failed!"
    print(f"SPECK32/64 works: 0x{t:x}")

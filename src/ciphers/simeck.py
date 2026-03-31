class SimeckCipher:
    def __init__(self, key):
        self.block_size = 32
        self.key_size = 64
        self.word_size = 16
        self.rounds = 32
        self.mod_mask = (1 << self.word_size) - 1

        key &= (1 << self.key_size) - 1
        key_words = self._split_words_be(key, 4)

        t = [0, 0, 0, 0, 0]
        t[3], t[2], t[1], t[0] = key_words

        self.round_keys = []
        constant = 0xFFFC
        sequence = 0x9A42BB1F

        for _ in range(self.rounds):
            self.round_keys.append(t[0])
            constant = (constant & 0xFFFC) | (sequence & 1)
            sequence >>= 1
            t[1], t[0] = self._round(constant, t[1], t[0])
            tmp = t[1]
            t[1] = t[2]
            t[2] = t[3]
            t[3] = tmp

    def _rotl(self, val, shift):
        shift %= self.word_size
        return ((val << shift) & self.mod_mask) | (val >> (self.word_size - shift))

    def _round(self, key, left, right):
        temp = left
        left = (left & self._rotl(left, 5)) ^ self._rotl(left, 1) ^ right ^ key
        left &= self.mod_mask
        right = temp
        return left, right

    def _split_words_be(self, val, count):
        words = []
        for i in range(count):
            shift = self.word_size * (count - 1 - i)
            words.append((val >> shift) & self.mod_mask)
        return words

    def _join_words_be(self, words):
        val = 0
        for word in words:
            val = (val << self.word_size) | (word & self.mod_mask)
        return val

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        words = self._split_words_be(pt, 2)
        left, right = words[0], words[1]

        for i in range(limit):
            left, right = self._round(self.round_keys[i], left, right)

        return self._join_words_be([left, right])


if __name__ == "__main__":
    key = int("1918111009080100", 16)
    pt = int("65656877", 16)
    expected = int("770D2C76", 16)

    ct = SimeckCipher(key).encrypt(pt)
    assert ct == expected, "SIMECK-32 KAT failed"
    print(f"SIMECK-32 KAT passed: 0x{ct:08x}")

class ChamCipher:
    def __init__(self, key):
        self.block_size = 128
        self.key_size = 128
        self.word_size = 32
        self.rounds = 80
        self.mod_mask = (1 << self.word_size) - 1

        key &= (1 << self.key_size) - 1
        key_words = self._split_words_be(key, 4)
        self.subkeys = self._key_schedule(key_words)

    def _rotl(self, val, shift):
        shift %= self.word_size
        return ((val << shift) & self.mod_mask) | (val >> (self.word_size - shift))

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

    def _key_schedule(self, key_words):
        kw = len(key_words)
        subkeys = [0] * (2 * kw)
        for i, word in enumerate(key_words):
            subkeys[i] = word ^ self._rotl(word, 1) ^ self._rotl(word, 8)
            idx = (i + kw) ^ 1
            subkeys[idx] = word ^ self._rotl(word, 1) ^ self._rotl(word, 11)
        return subkeys

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        state = self._split_words_be(pt, 4)
        key_len = len(self.subkeys)

        for i in range(limit):
            rr = i % 4
            r1 = 1 if (rr % 2 == 0) else 8
            r2 = 8 if (rr % 2 == 0) else 1
            idx0 = rr
            idx1 = (rr + 1) % 4
            kk = self.subkeys[i % key_len]

            aa = state[idx0] ^ i
            bb = self._rotl(state[idx1], r1) ^ kk
            state[idx0] = self._rotl((aa + bb) & self.mod_mask, r2)

        return self._join_words_be(state)


if __name__ == "__main__":
    key = int("03020100070605040B0A09080F0E0D0C", 16)
    pt = int("3322110077665544BBAA9988FFEEDDCC", 16)
    expected = int("C3746034B55700C58D64EC32489332F7", 16)

    ct = ChamCipher(key).encrypt(pt)
    assert ct == expected, "CHAM-128 KAT failed"
    print(f"CHAM-128 KAT passed: 0x{ct:032x}")

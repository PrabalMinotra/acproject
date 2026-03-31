class LeaCipher:
    def __init__(self, key):
        self.block_size = 128
        self.key_size = 128
        self.word_size = 32
        self.rounds = 24
        self.mod_mask = 0xFFFFFFFF

        self.delta = [
            0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec,
            0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957,
        ]

        key &= (1 << self.key_size) - 1
        key_words = self._split_words_from_be(key, 4)
        self.round_keys = self._key_schedule_128(key_words)

    def _rotl(self, val, shift):
        shift %= self.word_size
        return ((val << shift) & self.mod_mask) | (val >> (self.word_size - shift))

    def _rotr(self, val, shift):
        shift %= self.word_size
        return ((val >> shift) | (val << (self.word_size - shift))) & self.mod_mask

    def _split_words_from_be(self, val, count):
        data = val.to_bytes(count * 4, 'big')
        return [int.from_bytes(data[i * 4:(i + 1) * 4], 'little') for i in range(count)]

    def _join_words_to_be(self, words):
        data = b''.join((word & self.mod_mask).to_bytes(4, 'little') for word in words)
        return int.from_bytes(data, 'big')

    def _key_schedule_128(self, key_words):
        t = list(key_words)
        round_keys = []
        for i in range(self.rounds):
            d = self.delta[i % 4]
            t[0] = self._rotl((t[0] + self._rotl(d, i)) & self.mod_mask, 1)
            t[1] = self._rotl((t[1] + self._rotl(d, i + 1)) & self.mod_mask, 3)
            t[2] = self._rotl((t[2] + self._rotl(d, i + 2)) & self.mod_mask, 6)
            t[3] = self._rotl((t[3] + self._rotl(d, i + 3)) & self.mod_mask, 11)
            round_keys.append([t[0], t[1], t[2], t[1], t[3], t[1]])
        return round_keys

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        x0, x1, x2, x3 = self._split_words_from_be(pt, 4)

        for i in range(limit):
            k = self.round_keys[i]
            x0_old = x0

            x0 = self._rotl(((x0 ^ k[0]) + (x1 ^ k[1])) & self.mod_mask, 9)
            x1 = self._rotr(((x1 ^ k[2]) + (x2 ^ k[3])) & self.mod_mask, 5)
            x2 = self._rotr(((x2 ^ k[4]) + (x3 ^ k[5])) & self.mod_mask, 3)
            x3 = x0_old

        return self._join_words_to_be([x0, x1, x2, x3])


if __name__ == "__main__":
    key = int("07AB6305B025D83F79ADDAA63AC8AD00", 16)
    pt = int("F28AE3256AAD23B415E028063B610C60", 16)
    expected = int("64D908FCB7EBFEF90FD670106DE7C7C5", 16)

    ct = LeaCipher(key).encrypt(pt)
    assert ct == expected, "LEA-128 KAT failed"
    print(f"LEA-128 KAT passed: 0x{ct:032x}")

class HightCipher:
    def __init__(self, key):
        self.block_size = 64
        self.key_size = 128
        self.rounds = 32

        key &= (1 << self.key_size) - 1
        key_bytes = self._int_to_bytes_be(key, 16)
        self.round_keys = self._key_schedule(key_bytes)

    def _int_to_bytes_be(self, val, length):
        return [(val >> (8 * (length - 1 - i))) & 0xFF for i in range(length)]

    def _bytes_to_int_be(self, data):
        val = 0
        for byte in data:
            val = (val << 8) | (byte & 0xFF)
        return val

    def _rol8(self, val, shift):
        shift &= 7
        return ((val << shift) & 0xFF) | (val >> (8 - shift))

    def _f0(self, val):
        return self._rol8(val, 1) ^ self._rol8(val, 2) ^ self._rol8(val, 7)

    def _f1(self, val):
        return self._rol8(val, 3) ^ self._rol8(val, 4) ^ self._rol8(val, 6)

    def _delta_constants(self):
        delta = []
        state = 0x5A
        for _ in range(128):
            delta.append(state & 0x7F)
            new_bit = (state ^ (state >> 3)) & 1
            state = (state >> 1) | (new_bit << 6)
        return delta

    def _key_schedule(self, key_bytes):
        rkey = [0] * 136
        delta = self._delta_constants()

        for i in range(4):
            rkey[i] = key_bytes[i + 12]
            rkey[i + 4] = key_bytes[i]

        for i in range(8):
            for j in range(8):
                idx = 8 + 16 * i + j
                rkey[idx] = (key_bytes[(j - i) & 7] + delta[idx - 8]) & 0xFF
            for j in range(8):
                idx = 8 + 16 * i + j + 8
                rkey[idx] = (key_bytes[((j - i) & 7) + 8] + delta[idx - 8]) & 0xFF

        return rkey

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        pt_bytes = self._int_to_bytes_be(pt, 8)
        rk = self.round_keys

        xx = [0] * 8
        xx[1] = pt_bytes[1]
        xx[3] = pt_bytes[3]
        xx[5] = pt_bytes[5]
        xx[7] = pt_bytes[7]

        xx[0] = (pt_bytes[0] + rk[0]) & 0xFF
        xx[2] = pt_bytes[2] ^ rk[1]
        xx[4] = (pt_bytes[4] + rk[2]) & 0xFF
        xx[6] = pt_bytes[6] ^ rk[3]

        for r in range(limit):
            k = r + 2
            offset = (7 - r) & 7
            i0 = offset
            i1 = (offset - 1) & 7
            i2 = (offset - 2) & 7
            i3 = (offset - 3) & 7
            i4 = (offset - 4) & 7
            i5 = (offset - 5) & 7
            i6 = (offset - 6) & 7
            i7 = (offset - 7) & 7

            xx[i0] = (xx[i0] ^ (self._f0(xx[i1]) + rk[4 * k + 3])) & 0xFF
            xx[i2] = (xx[i2] + (self._f1(xx[i3]) ^ rk[4 * k + 2])) & 0xFF
            xx[i4] = (xx[i4] ^ (self._f0(xx[i5]) + rk[4 * k + 1])) & 0xFF
            xx[i6] = (xx[i6] + (self._f1(xx[i7]) ^ rk[4 * k + 0])) & 0xFF

        out = [0] * 8
        out[1] = xx[2]
        out[3] = xx[4]
        out[5] = xx[6]
        out[7] = xx[0]

        if limit == self.rounds:
            out[0] = (xx[1] + rk[4]) & 0xFF
            out[2] = xx[3] ^ rk[5]
            out[4] = (xx[5] + rk[6]) & 0xFF
            out[6] = xx[7] ^ rk[7]
        else:
            out[0] = xx[1]
            out[2] = xx[3]
            out[4] = xx[5]
            out[6] = xx[7]

        return self._bytes_to_int_be(out)


if __name__ == "__main__":
    key = int("88E34F8F081779F1E9F394370AD40589", 16)
    pt = int("D76D0D18327EC562", 16)
    expected = int("E4BC2E312277E4DD", 16)

    ct = HightCipher(key).encrypt(pt)
    assert ct == expected, "HIGHT KAT failed"
    print(f"HIGHT KAT passed: 0x{ct:016x}")

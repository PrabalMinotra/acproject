class Salsa20Cipher:
    """Salsa20 Stream Cipher
    Using 256-bit Key, 64-bit Nonce.
    """
    def __init__(self, key):
        self.block_size = 512 
        self.key_size = 256
        self.rounds = 20
        self.k = [(key >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
        # Salsa fixed constants
        self.c = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
        
    def _rotl(self, x, n):
        return ((x << n) & 0xFFFFFFFF) | (x >> (32 - n))
        
    def _qr(self, state, y0, y1, y2, y3):
        state[y1] ^= self._rotl((state[y0] + state[y3]) & 0xFFFFFFFF, 7)
        state[y2] ^= self._rotl((state[y1] + state[y0]) & 0xFFFFFFFF, 9)
        state[y3] ^= self._rotl((state[y2] + state[y1]) & 0xFFFFFFFF, 13)
        state[y0] ^= self._rotl((state[y3] + state[y2]) & 0xFFFFFFFF, 18)

    def generate_keystream(self, rounds):
        state = [
            self.c[0], self.k[0], self.k[1], self.k[2],
            self.k[3], self.c[1], 0, 0, # nonce at index 6,7
            0, 0, self.c[2], self.k[4], # counter at 8,9
            self.k[5], self.k[6], self.k[7], self.c[3]
        ]
        
        working_state = list(state)
        
        for i in range(0, rounds, 2):
            # Column round
            self._qr(working_state, 0, 4, 8, 12)
            self._qr(working_state, 5, 9, 13, 1)
            self._qr(working_state, 10, 14, 2, 6)
            self._qr(working_state, 15, 3, 7, 11)
            
            if i + 1 >= rounds: break
            
            # Row round
            self._qr(working_state, 0, 1, 2, 3)
            self._qr(working_state, 5, 6, 7, 4)
            self._qr(working_state, 10, 11, 8, 9)
            self._qr(working_state, 15, 12, 13, 14)
            
        for i in range(16):
            working_state[i] = (working_state[i] + state[i]) & 0xFFFFFFFF
            
        stream = 0
        for i in range(16):
            stream |= (working_state[i] << (32 * i))
            
        return stream

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        stream = self.generate_keystream(limit)
        return pt ^ stream

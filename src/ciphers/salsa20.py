class Salsa20Cipher:
    
    def __init__(self, key):
        self.block_size = 512 
        self.key_size = 256
        self.rounds = 20
        self.k = [(key >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
        
        self.c = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]

        self.counter_bits = 64
        self.nonce_bits = 64
        self.header_bits = self.counter_bits + self.nonce_bits
        self.message_bits = self.block_size - self.header_bits
        
    def _rotl(self, x, n):
        return ((x << n) & 0xFFFFFFFF) | (x >> (32 - n))
        
    def _qr(self, state, y0, y1, y2, y3):
        state[y1] ^= self._rotl((state[y0] + state[y3]) & 0xFFFFFFFF, 7)
        state[y2] ^= self._rotl((state[y1] + state[y0]) & 0xFFFFFFFF, 9)
        state[y3] ^= self._rotl((state[y2] + state[y1]) & 0xFFFFFFFF, 13)
        state[y0] ^= self._rotl((state[y3] + state[y2]) & 0xFFFFFFFF, 18)

    def _split_header(self, pt):
        counter = pt & ((1 << self.counter_bits) - 1)
        nonce = (pt >> self.counter_bits) & ((1 << self.nonce_bits) - 1)
        message = pt >> self.header_bits
        return counter, nonce, message

    def generate_keystream(self, rounds, counter, nonce):
        nonce_words = [(nonce >> (32 * i)) & 0xFFFFFFFF for i in range(2)]
        counter_words = [(counter >> (32 * i)) & 0xFFFFFFFF for i in range(2)]
        state = [
            self.c[0], self.k[0], self.k[1], self.k[2],
            self.k[3], self.c[1], nonce_words[0], nonce_words[1],
            counter_words[0], counter_words[1], self.c[2], self.k[4],
            self.k[5], self.k[6], self.k[7], self.c[3]
        ]
        
        working_state = list(state)
        
        for i in range(0, rounds, 2):
            
            self._qr(working_state, 0, 4, 8, 12)
            self._qr(working_state, 5, 9, 13, 1)
            self._qr(working_state, 10, 14, 2, 6)
            self._qr(working_state, 15, 3, 7, 11)
            
            if i + 1 >= rounds: break
            
            
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
        counter, nonce, message = self._split_header(pt)
        stream = self.generate_keystream(limit, counter, nonce)

        message_mask = (1 << self.message_bits) - 1
        stream_message = (stream >> self.header_bits) & message_mask
        cipher_message = (message ^ stream_message) & message_mask
        return (cipher_message << self.header_bits) | (nonce << self.counter_bits) | counter

if __name__ == "__main__":
    key = int.from_bytes(bytes(range(32)), 'big')
    pt = 0
    ct = Salsa20Cipher(key).encrypt(pt)
    assert ct == 0x3619BB074E6CBF6F00537314E2F7BBE7F141B1B5079183E00C8B8B176AFB8B68C5F86BF19170DB775EC17E8AD67F704400000000000000000000000000000000, "Salsa20 regression test failed (self-generated)!"
    print(f"Salsa20 regression test passed: 0x{ct:x}")

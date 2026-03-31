class ChaCha20Cipher:
    
    def __init__(self, key):
        self.block_size = 512 
        self.key_size = 256
        self.rounds = 20 

        self.counter_bits = 32
        self.nonce_bits = 96
        self.header_bits = self.counter_bits + self.nonce_bits
        self.message_bits = self.block_size - self.header_bits
        
        
        self.k = [(key >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
        
        
        self.c = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
        
    def _rotl(self, x, n):
        return ((x << n) & 0xFFFFFFFF) | (x >> (32 - n))
        
    def _qr(self, state, a, b, c, d):
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = self._rotl(state[d], 16)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = self._rotl(state[b], 12)
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = self._rotl(state[d], 8)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = self._rotl(state[b], 7)

    def _split_header(self, pt):
        counter = pt & 0xFFFFFFFF
        nonce = (pt >> self.counter_bits) & ((1 << self.nonce_bits) - 1)
        message = pt >> self.header_bits
        return counter, nonce, message

    def generate_keystream(self, rounds, counter, nonce):
        nonce_words = [(nonce >> (32 * i)) & 0xFFFFFFFF for i in range(3)]
        state = self.c + self.k + [counter & 0xFFFFFFFF] + nonce_words
        
        working_state = list(state)
        
        
        
        for i in range(0, rounds, 2):
            
            self._qr(working_state, 0, 4, 8, 12)
            self._qr(working_state, 1, 5, 9, 13)
            self._qr(working_state, 2, 6, 10, 14)
            self._qr(working_state, 3, 7, 11, 15)
            
            if i + 1 >= rounds: break
            
            
            self._qr(working_state, 0, 5, 10, 15)
            self._qr(working_state, 1, 6, 11, 12)
            self._qr(working_state, 2, 7, 8, 13)
            self._qr(working_state, 3, 4, 9, 14)
            
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
    key = int.from_bytes(bytes(range(32)), 'little')
    nonce = int.from_bytes(bytes.fromhex('000000090000004a00000000'), 'little')
    counter = 1
    stream = ChaCha20Cipher(key).generate_keystream(20, counter, nonce)
    expected = int.from_bytes(bytes.fromhex(
        '10f1e7e4d13b5915500fdd1fa32071c4'
        'c7d1f4c733c068030422aa9ac3d46c4e'
        'd2826446079faa0914c2d705d98b02a2'
        'b5129cd1de164eb9cbd083e8a2503c4e'
    ), 'little')
    assert stream == expected, "ChaCha20 RFC8439 KAT failed!"
    print(f"ChaCha20 KAT passed: 0x{stream:x}")

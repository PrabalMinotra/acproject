import struct

class ChaCha20Cipher:
    """ChaCha20 Stream Cipher
    Using 256-bit Key, 64-bit Nonce.
    Simulating block cipher functionality by XORing the stream with a 512-bit block.
    """
    def __init__(self, key):
        self.block_size = 512 # ChaCha generates 512 bit keystreams
        self.key_size = 256
        self.rounds = 20 # Standard is 20 rounds (10 iterations of double-round)
        
        # 8 words of key
        self.k = [(key >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
        
        # Fixed constants
        self.c = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
        
    def _rotl(self, x, n):
        return ((x << n) & 0xFFFFFFFF) | (x >> (32 - n))
        
    def _qr(self, state, a, b, c, d):
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = self._rotl(state[d], 16)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = self._rotl(state[b], 12)
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = self._rotl(state[d], 8)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = self._rotl(state[b], 7)

    def generate_keystream(self, rounds):
        state = self.c + self.k + [0, 0, 0, 0] # Nonce and counter 0
        
        working_state = list(state)
        # rounds parameter should be even (must be executed as double-rounds usually,
        # but here we can define it strictly per round)
        
        for i in range(0, rounds, 2):
            # Column round
            self._qr(working_state, 0, 4, 8, 12)
            self._qr(working_state, 1, 5, 9, 13)
            self._qr(working_state, 2, 6, 10, 14)
            self._qr(working_state, 3, 7, 11, 15)
            
            if i + 1 >= rounds: break
            
            # Diagonal round
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
        stream = self.generate_keystream(limit)
        return pt ^ stream
        
if __name__ == "__main__":
    c = ChaCha20Cipher((1 << 256) - 1)
    print(hex(c.encrypt((1 << 512) - 1, rounds=20)))

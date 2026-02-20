class TriviumCipher:
    """Trivium Stream Cipher
    80-bit Key, 80-bit IV.
    """
    def __init__(self, key):
        self.block_size = 64 # Expose 64 bit chunks like block ciphers
        self.key_size = 80
        self.rounds = 1152 # Setup rounds
        
        self.init_key = key
        self.init_iv = 0
        
    def _init_state(self, rounds):
        state = [0] * 288
        
        # Load Key into 1..80
        for i in range(80):
            state[i] = (self.init_key >> i) & 1
            
        # Load IV into 94..173
        for i in range(80):
            state[93 + i] = (self.init_iv >> i) & 1
            
        state[285] = 1
        state[286] = 1
        state[287] = 1
        
        # Clock initialization rounds
        for _ in range(rounds):
            t1 = state[65] ^ state[92]
            t2 = state[161] ^ state[176]
            t3 = state[242] ^ state[287]
            
            z = t1 ^ t2 ^ t3
            
            t1 = t1 ^ (state[90] & state[91]) ^ state[170]
            t2 = t2 ^ (state[174] & state[175]) ^ state[263]
            t3 = t3 ^ (state[285] & state[286]) ^ state[68]
            
            # Shift state
            for i in range(287, 0, -1):
                state[i] = state[i-1]
                
            state[0] = t3
            state[93] = t1
            state[177] = t2
            
        return state
        
    def generate_keystream(self, setup_rounds, length_bits):
        state = self._init_state(setup_rounds)
        stream = 0
        
        for i in range(length_bits):
            t1 = state[65] ^ state[92]
            t2 = state[161] ^ state[176]
            t3 = state[242] ^ state[287]
            
            z = t1 ^ t2 ^ t3
            stream |= (z << i)
            
            t1 = t1 ^ (state[90] & state[91]) ^ state[170]
            t2 = t2 ^ (state[174] & state[175]) ^ state[263]
            t3 = t3 ^ (state[285] & state[286]) ^ state[68]
            
            for j in range(287, 0, -1):
                state[j] = state[j-1]
                
            state[0] = t3
            state[93] = t1
            state[177] = t2
            
        return stream

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds # Initialization rounds
        stream = self.generate_keystream(setup_rounds=limit, length_bits=64)
        return pt ^ stream

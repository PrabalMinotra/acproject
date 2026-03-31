class TriviumCipher:
    
    def __init__(self, key):
        self.block_size = 512 
        self.key_size = 80
        self.rounds = 1152 

        self.iv_bits = 80
        self.header_bits = self.iv_bits
        self.message_bits = self.block_size - self.iv_bits
        
        self.init_key = key

    def _init_state(self, rounds, iv):
        state = [0] * 288
        
        
        for i in range(80):
            state[i] = (self.init_key >> i) & 1
            
        
        for i in range(80):
            state[93 + i] = (iv >> i) & 1
            
        state[285] = 1
        state[286] = 1
        state[287] = 1
        
        
        for _ in range(rounds):
            t1 = state[65] ^ state[92]
            t2 = state[161] ^ state[176]
            t3 = state[242] ^ state[287]
            
            z = t1 ^ t2 ^ t3
            
            t1 = t1 ^ (state[90] & state[91]) ^ state[170]
            t2 = t2 ^ (state[174] & state[175]) ^ state[263]
            t3 = t3 ^ (state[285] & state[286]) ^ state[68]
            
            
            for i in range(287, 0, -1):
                state[i] = state[i-1]
                
            state[0] = t3
            state[93] = t1
            state[177] = t2
            
        return state
        
    def generate_keystream(self, setup_rounds, length_bits, iv):
        state = self._init_state(setup_rounds, iv)
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
        limit = rounds if rounds is not None else self.rounds 
        iv = pt & ((1 << self.iv_bits) - 1)
        message = pt >> self.iv_bits

        stream = self.generate_keystream(setup_rounds=limit, length_bits=self.message_bits, iv=iv)
        message_mask = (1 << self.message_bits) - 1
        cipher_message = (message ^ stream) & message_mask
        return (cipher_message << self.iv_bits) | iv

if __name__ == "__main__":
    key = 0x00010203040506070809
    pt = 0
    ct = TriviumCipher(key).encrypt(pt)
    assert ct == 0x46173B70E7D9EAC672E0CEB59FC2D1AE79E32C7B130B48A0046E65D3644133E808AACE8E96885F06FEF3C88472F2280547C9988C63A600000000000000000000, "Trivium regression test failed (self-generated)!"
    print(f"Trivium regression test passed: 0x{ct:x}")

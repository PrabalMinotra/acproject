class RectangleCipher:
    
    def __init__(self, key):
        self.block_size = 64
        self.key_size = 128
        self.rounds = 25
        
        self.S = [0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9, 0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2]
        
        self.RC = [
            0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 
            0x0C, 0x19, 0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C, 
            0x18, 0x11, 0x03, 0x06, 0x0D, 0x1B, 0x17, 0x0E, 
            0x1D
        ]
        
        self.key_schedule = []
        state_key = [(key >> (16 * i)) & 0xFFFF for i in range(8)]
        
        for r in range(self.rounds + 1):
            k_round = [state_key[0], state_key[1], state_key[2], state_key[3]]
            self.key_schedule.append(k_round)
            
            
            for col in range(16):
                val = 0
                for row in range(4):
                    val |= (((state_key[row] >> col) & 1) << row)
                
                s_val = self.S[val]
                
                for row in range(4):
                    bit = (s_val >> row) & 1
                    if bit == 1:
                        state_key[row] |= (1 << col)
                    else:
                        state_key[row] &= ~(1 << col)
                        
            
            state_key[0] = self.ROL(state_key[0], 8)
            state_key[1] = self.ROL(state_key[1], 12)
            state_key[2] = self.ROL(state_key[2], 12)
            state_key[3] = self.ROL(state_key[3], 13)
            
            if r < 25:
                state_key[0] ^= self.RC[r]

    def ROL(self, val, r_bits):
        return ((val << r_bits) & 0xFFFF) | (val >> (16 - r_bits))

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        
        state = [(pt >> (16 * i)) & 0xFFFF for i in range(4)]
        
        for i in range(limit):
            
            for row in range(4):
                state[row] ^= self.key_schedule[i][row]
                
            
            for col in range(16):
                val = 0
                for row in range(4):
                    val |= (((state[row] >> col) & 1) << row)
                
                s_val = self.S[val]
                
                for row in range(4):
                    bit = (s_val >> row) & 1
                    if bit == 1:
                        state[row] |= (1 << col)
                    else:
                        state[row] &= ~(1 << col)
            
            
            state[1] = self.ROL(state[1], 1)
            state[2] = self.ROL(state[2], 12)
            state[3] = self.ROL(state[3], 13)
            
        if limit == self.rounds:
            for row in range(4):
                state[row] ^= self.key_schedule[self.rounds][row]
                
        final = 0
        for i in range(4):
            final |= (state[i] << (16 * i))
            
        return final

if __name__ == "__main__":
    key = 0x000102030405060708090A0B0C0D0E0F
    pt = 0x0123456789ABCDEF
    ct = RectangleCipher(key).encrypt(pt)
    assert ct == 0xB23B6C4441A05030, "RECTANGLE regression test failed (self-generated)!"
    print(f"RECTANGLE regression test passed: 0x{ct:x}")

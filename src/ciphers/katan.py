class KatanCipher:
    
    def __init__(self, key):
        self.block_size = 32
        self.key_size = 80
        self.rounds = 254
        
        
        IR = [
            1,1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,
            0,1,0,1,1,1,1,0,1,1,0,0,1,1,0,0,
            1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,
            0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,
            0,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,
            1,1,1,1,0,1,0,1,0,0,0,1,0,1,0,1,
            0,0,1,1,0,0,0,0,1,1,0,0,1,1,1,0,
            1,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,
            0,1,1,0,1,0,0,1,1,1,0,0,1,1,0,1,
            1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,1,
            1,0,0,1,0,1,1,0,1,1,0,1,0,0,0,0,
            1,1,0,1,0,0,1,1,0,0,0,1,1,1,1,1,
            1,0,1,1,1,0,0,0,1,0,1,0,0,0,0,1,
            0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,1,
            0,1,1,0,0,0,1,0,1,0,1,1,0,1,0,0,
            1,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0
        ]
        
        self.k = [((key >> i) & 1) for i in range(80)]
        self.k = [((key >> i) & 1) for i in range(80)]
        self.ka_vec = []
        self.kb_vec = []
        for i in range(self.rounds):
            ka_i = self.k[0] ^ self.k[19] ^ self.k[30] ^ self.k[67] ^ IR[i]
            kb_i = self.k[1] ^ self.k[24] ^ self.k[39] ^ self.k[61]
            self.ka_vec.append(ka_i)
            self.kb_vec.append(kb_i)
            for j in range(79):
                self.k[j] = self.k[j+1]
            self.k[79] = ka_i

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        
        state = pt & 0xFFFFFFFF
        L1 = state & 0x7FFFF 
        L2 = (state >> 19) & 0x1FFF 
        
        l1_arr = [(L1 >> i) & 1 for i in range(19)]
        l2_arr = [(L2 >> i) & 1 for i in range(13)]
        
        for i in range(limit):
            ka = self.ka_vec[i]
            kb = self.kb_vec[i]
            
            f_a = l1_arr[18] ^ l1_arr[7] ^ (l1_arr[12] & l1_arr[10]) ^ (l1_arr[8] & l1_arr[3]) ^ ka
            f_b = l2_arr[12] ^ l2_arr[7] ^ (l2_arr[8] & l2_arr[5]) ^ (l2_arr[3] & l2_arr[8]) ^ kb
            
            
            for j in range(18, 0, -1):
                l1_arr[j] = l1_arr[j-1]
            l1_arr[0] = f_b
            
            
            for j in range(12, 0, -1):
                l2_arr[j] = l2_arr[j-1]
            l2_arr[0] = f_a
            
        final_L1 = 0
        for i in range(19):
            final_L1 |= (l1_arr[i] << i)
        
        final_L2 = 0
        for i in range(13):
            final_L2 |= (l2_arr[i] << i)
            
        return (final_L2 << 19) | final_L1

if __name__ == "__main__":
    key = 0x00010203040506070809
    pt = 0x01234567
    ct = KatanCipher(key).encrypt(pt)
    assert ct == 0xCC904D58, "KATAN regression test failed (self-generated)!"
    print(f"KATAN regression test passed: 0x{ct:x}")

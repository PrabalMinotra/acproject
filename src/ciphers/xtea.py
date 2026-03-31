class XTEACipher:
    
    def __init__(self, key):
        self.block_size = 64
        self.key_size = 128
        self.rounds = 32
        self.delta = 0x9e3779b9
        self.k = [(key >> (96 - 32 * i)) & 0xFFFFFFFF for i in range(4)]

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        
        v0 = (pt >> 32) & 0xFFFFFFFF
        v1 = pt & 0xFFFFFFFF
        
        sum = 0
        for _ in range(limit):
            v0 = (v0 + (((v1 << 4 ^ v1 >> 5) + v1) ^ (sum + self.k[sum & 3]))) & 0xFFFFFFFF
            sum = (sum + self.delta) & 0xFFFFFFFF
            v1 = (v1 + (((v0 << 4 ^ v0 >> 5) + v0) ^ (sum + self.k[(sum >> 11) & 3]))) & 0xFFFFFFFF
            
        return (v0 << 32) | v1

if __name__ == "__main__":
    key = 0x0123456789ABCDEF0123456789ABCDEF
    pt = 0x0123456789ABCDEF
    ct = XTEACipher(key).encrypt(pt)
    assert ct == 0x27E795E076B2B537, "XTEA regression test failed (self-generated)!"
    print(f"XTEA regression test passed: 0x{ct:x}")

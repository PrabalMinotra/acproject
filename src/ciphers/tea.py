import struct

class TEACipher:
    """TEA (Tiny Encryption Algorithm) Block Cipher
    Block Size: 64 bits
    Key Size: 128 bits
    """
    def __init__(self, key):
        self.block_size = 64
        self.key_size = 128
        self.rounds = 32 # TEA nominally has 64 Feistel rounds (32 cycles)
        self.delta = 0x9e3779b9
        
        # Key must be an integer, we split it into four 32-bit keys
        self.k = [(key >> (96 - 32 * i)) & 0xFFFFFFFF for i in range(4)]

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        
        v0 = (pt >> 32) & 0xFFFFFFFF
        v1 = pt & 0xFFFFFFFF
        
        sum = 0
        
        for _ in range(limit):
            sum = (sum + self.delta) & 0xFFFFFFFF
            v0 = (v0 + (((v1 << 4) + self.k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + self.k[1]))) & 0xFFFFFFFF
            v1 = (v1 + (((v0 << 4) + self.k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + self.k[3]))) & 0xFFFFFFFF
            
        return (v0 << 32) | v1

if __name__ == "__main__":
    t = TEACipher(0x0123456789abcdef0123456789abcdef)
    ct = t.encrypt(0x0123456789abcdef)
    print(f"TEA Test: {hex(ct)}")

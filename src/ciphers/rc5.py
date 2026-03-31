class RC5Cipher:
    
    def __init__(self, key):
        self.w = 32
        self.rounds = 12
        self.b = 16
        self.block_size = 64
        self.mod = 2**32
        
        
        P = 0xb7e15163
        Q = 0x9e3779b9
        
        c = max(1, (self.b * 8) // self.w)
        L = [(key >> (32 * i)) & 0xFFFFFFFF for i in range(c)]
        
        self.S = [0] * (2 * (self.rounds + 1))
        self.S[0] = P
        for i in range(1, len(self.S)):
            self.S[i] = (self.S[i-1] + Q) % self.mod
            
        i = j = 0
        A = B = 0
        for _ in range(3 * max(len(self.S), c)):
            A = self.S[i] = self.ROL((self.S[i] + A + B) % self.mod, 3)
            B = L[j] = self.ROL((L[j] + A + B) % self.mod, (A + B) % self.w)
            i = (i + 1) % len(self.S)
            j = (j + 1) % c

    def ROL(self, val, r_bits):
        r_bits = r_bits % self.w
        return ((val << r_bits) & (self.mod - 1)) | ((val & (self.mod - 1)) >> (self.w - r_bits))

    def encrypt(self, pt, rounds=None):
        limit = rounds if rounds is not None else self.rounds
        
        A = (pt >> 32) & 0xFFFFFFFF
        B = pt & 0xFFFFFFFF
        
        A = (A + self.S[0]) % self.mod
        B = (B + self.S[1]) % self.mod
        
        for i in range(1, limit + 1):
            A = (self.ROL(A ^ B, B) + self.S[2 * i]) % self.mod
            B = (self.ROL(B ^ A, A) + self.S[2 * i + 1]) % self.mod
            
        return (A << 32) | B

if __name__ == "__main__":
    key = 0x000102030405060708090A0B0C0D0E0F
    pt = 0x0001020304050607
    ct = RC5Cipher(key).encrypt(pt)
    assert ct == 0x7E2055A7FED5B3A6, "RC5 regression test failed (self-generated)!"
    print(f"RC5 regression test passed: 0x{ct:x}")

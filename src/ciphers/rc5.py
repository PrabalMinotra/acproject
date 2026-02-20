class RC5Cipher:
    """RC5 32/12/16 Block Cipher
    Word Size w = 32
    Rounds r = 12
    Key Bytes b = 16 (128 bits)
    Block Size = 64
    """
    def __init__(self, key):
        self.w = 32
        self.rounds = 12
        self.b = 16
        self.block_size = 64
        self.mod = 2**32
        
        # Magic constants for w=32
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
    r = RC5Cipher(0x00000000000000000000000000000000)
    ct = r.encrypt(0x0000000000000000)
    print(f"RC5 Test: {hex(ct)}")

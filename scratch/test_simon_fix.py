class Simon:
    def __init__(self, key):
        self.k = [0] * 32
        for i in range(4):
            self.k[i] = (key >> (i * 16)) & 0xffff
        z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,
              1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]
        for i in range(4, 32):
            tmp = (self.k[i-1] >> 3) | ((self.k[i-1] << 13) & 0xffff)
            tmp ^= self.k[i-3]
            tmp ^= ((tmp >> 1) | ((tmp << 15) & 0xffff))
            self.k[i] = (~self.k[i-4] & 0xffff) ^ tmp ^ z0[(i-4)%62] ^ 3
    
    def enc(self, pt):
        x = (pt >> 16) & 0xffff
        y = pt & 0xffff
        for i in range(32):
            tmp = (((x << 1) & 0xffff) | (x >> 15)) & (((x << 8) & 0xffff) | (x >> 8))
            tmp ^= (((x << 2) & 0xffff) | (x >> 14))
            y, x = y ^ tmp ^ self.k[i], x
        return (x << 16) | y

s = Simon(0x1918111009080100)
print(hex(s.enc(0x65656877)))

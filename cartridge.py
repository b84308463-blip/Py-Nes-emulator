class Cartridge:
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = f.read()

        if data[:4] != b'NES\x1a':
            raise ValueError("Not an iNES ROM (missing NES\\x1a header)")

        flags6 = data[6]
        flags7 = data[7]
        self.nes2 = (flags7 & 0x0C) == 0x08

        prg_banks = data[4]
        chr_banks = data[5]
        self.mapper_id = (flags7 & 0xF0) | (flags6 >> 4)
        self.submapper = 0
        self.region_hint = None

        if self.nes2:
            self.mapper_id |= (data[8] & 0x0F) << 8
            self.submapper = data[8] >> 4
            prg_banks |= (data[9] & 0x0F) << 8
            chr_banks |= (data[9] >> 4) << 8
            timing = data[12] & 0x03
            self.region_hint = {0: "NTSC", 1: "PAL", 2: None, 3: "DENDY"}[timing]
        elif data[9] & 1:
            self.region_hint = "PAL"          # rarely set, but cheap to honour

        self.prg_rom_banks = prg_banks
        self.chr_rom_banks = chr_banks
        self.four_screen = bool(flags6 & 0x08)
        self.mirroring = 4 if self.four_screen else (flags6 & 0x01)   # 0=H 1=V 4=four-screen
        self.has_battery = bool(flags6 & 0x02)
        self.has_trainer = bool(flags6 & 0x04)

        offset = 16 + (512 if self.has_trainer else 0)
        prg_size = prg_banks * 16384
        chr_size = chr_banks * 8192

        prg = bytearray(data[offset:offset + prg_size])
        if len(prg) < prg_size:                       # truncated dump -> pad
            prg += bytes(prg_size - len(prg))
        if len(prg) == 0:
            prg = bytearray(16384)
        self.prg_rom = bytes(prg)
        offset += prg_size

        if chr_size > 0:
            chr_ = bytearray(data[offset:offset + chr_size])
            if len(chr_) < chr_size:
                chr_ += bytearray(chr_size - len(chr_))
            self.chr_rom = chr_
            self.chr_is_ram = False
        else:
            ram = 8192
            if self.nes2 and (data[11] & 0x0F):
                ram = 64 << (data[11] & 0x0F)
            self.chr_rom = bytearray(max(ram, 8192))
            self.chr_is_ram = True

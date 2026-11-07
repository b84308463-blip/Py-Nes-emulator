from apu import APU


class Bus:
    def __init__(self, cart, region, audio=True):
        self.cart = cart
        self.region = region
        self.ram = bytearray(2048)
        self.prg_ram = bytearray(8192)   # $6000-$7FFF
        self.cpu = None
        self.ppu = None
        self.mapper = None
        self.apu = APU(self, region, audio=audio)
        self.controller = [0, 0]         # live button state, bit0=A 1=B 2=Select 3=Start 4=Up 5=Down 6=Left 7=Right
        self.shift = [0, 0]              # latched shift registers
        self.strobe = 0

    def attach_mapper(self, mapper):
        self.mapper = mapper
        self.prg = mapper.prg            # direct references for the fast path
        self.prg_off = mapper.prg_off    # (mappers must mutate, never replace, this list)

    # ------------------------------------------------------------------
    def cpu_read(self, addr):
        if addr >= 0x8000:
            return self.prg[self.prg_off[(addr >> 13) & 3] + (addr & 0x1FFF)]
        if addr < 0x2000:
            return self.ram[addr & 0x07FF]
        if addr < 0x4000:
            return self.ppu.read_register(addr & 7)
        if addr >= 0x6000:
            return self.prg_ram[addr - 0x6000]
        if addr == 0x4015:
            return self.apu.read_status()
        if addr == 0x4016 or addr == 0x4017:
            i = addr & 1
            if self.strobe:
                return 0x40 | (self.controller[i] & 1)
            v = self.shift[i]
            self.shift[i] = (v >> 1) | 0x80      # after 8 reads the pad returns 1s
            return 0x40 | (v & 1)
        if addr >= 0x4020:
            return self.mapper.low_read(addr)
        return 0

    def cpu_write(self, addr, val):
        val &= 0xFF
        if addr < 0x2000:
            self.ram[addr & 0x07FF] = val
        elif addr < 0x4000:
            self.ppu.write_register(addr & 7, val)
        elif addr >= 0x8000:
            self.mapper.cpu_write(addr, val)
        elif addr >= 0x6000:
            self.prg_ram[addr - 0x6000] = val
            self.mapper.low_write(addr, val)
        elif addr == 0x4014:
            base = (val << 8) & 0xFFFF
            ppu = self.ppu
            oam = ppu.oam
            start = ppu.oam_addr
            rd = self.cpu_read
            for i in range(256):
                oam[(start + i) & 0xFF] = rd((base + i) & 0xFFFF)
            ppu.oam_dirty = True
            if self.cpu:
                self.cpu.cycles += 513 + (self.cpu.cycles & 1)
        elif addr == 0x4016:
            new = val & 1
            if new or self.strobe:               # latch while high and on the falling edge
                self.shift[0] = self.controller[0]
                self.shift[1] = self.controller[1]
            self.strobe = new
        elif addr >= 0x4020:
            self.mapper.low_write(addr, val)
        else:                                   # $4000-$4013, $4015, $4017
            self.apu.write_register(addr, val)

    # Battery-backed RAM helpers -----------------------------------------
    def load_sram(self, data):
        n = min(len(data), len(self.prg_ram))
        self.prg_ram[:n] = data[:n]

    def dump_sram(self):
        return bytes(self.prg_ram)

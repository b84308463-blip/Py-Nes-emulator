"""Cartridge mappers.

All mappers work by filling two small bank tables:
    prg_off[4]  - byte offset into PRG ROM for each 8 KiB slot at $8000/$A000/$C000/$E000
    chr_off[8]  - byte offset into CHR for each 1 KiB slot of $0000-$1FFF
The CPU bus and PPU read straight through those tables, so a mapper only needs
to recompute the tables when a register is written.

Mirroring codes (mapper.mirroring): None=use header, 0=horizontal, 1=vertical,
2=single-screen low, 3=single-screen high, 4=four-screen.
"""

H, V, LO, HI = 0, 1, 2, 3
_MIR_VRC = (V, H, LO, HI)       # value written to a "mirroring" register -> code


class Mapper:
    name = "Unknown"
    mirroring = None
    nt_map = None               # optional custom nametable offsets (4 entries)
    has_scanline = False        # PPU calls scanline() once per rendered line
    has_cpu_clock = False       # emulator calls cpu_clock(cycles) after each instruction
    fetch_hook = False          # PPU must use ppu_read() for every pattern fetch

    def __init__(self, cart):
        self.cart = cart
        self.prg = cart.prg_rom
        self.chr = cart.chr_rom
        self.chr_is_ram = cart.chr_is_ram
        self.n_prg8 = max(1, len(self.prg) // 0x2000)
        self.n_chr1 = max(1, len(self.chr) // 0x400)
        self.prg_off = [0, 0, 0, 0]
        self.chr_off = [0] * 8
        self.irq_pending = False
        self.reset()

    # ---- banking helpers ------------------------------------------------
    def prg8(self, slot, bank):
        self.prg_off[slot] = (bank % self.n_prg8) * 0x2000

    def prg16(self, slot, bank):           # slot 0 = $8000, 1 = $C000
        self.prg8(slot * 2, bank * 2)
        self.prg8(slot * 2 + 1, bank * 2 + 1)

    def prg32(self, bank):
        for i in range(4):
            self.prg8(i, bank * 4 + i)

    def chr1(self, slot, bank):
        self.chr_off[slot] = (bank % self.n_chr1) * 0x400

    def chr2(self, slot, bank):            # slot 0..3
        self.chr1(slot * 2, bank * 2)
        self.chr1(slot * 2 + 1, bank * 2 + 1)

    def chr4(self, slot, bank):            # slot 0..1
        for i in range(4):
            self.chr1(slot * 4 + i, bank * 4 + i)

    def chr8(self, bank):
        for i in range(8):
            self.chr1(i, bank * 8 + i)

    @property
    def last8(self):
        return self.n_prg8 - 1

    @property
    def last16(self):
        return max(0, self.n_prg8 // 2 - 1)

    # ---- interface used by bus / ppu -------------------------------------
    def reset(self):
        self.prg32(0)
        self.chr8(0)

    def cpu_write(self, addr, v):          # $8000-$FFFF
        pass

    def low_write(self, addr, v):          # $4020-$7FFF
        pass

    def low_read(self, addr):
        return 0

    def ppu_read(self, addr):
        return self.chr[self.chr_off[addr >> 10] + (addr & 0x3FF)]

    def ppu_write(self, addr, v):
        if self.chr_is_ram:
            self.chr[self.chr_off[addr >> 10] + (addr & 0x3FF)] = v

    def scanline(self):
        pass

    def cpu_clock(self, cycles):
        pass


# ---------------------------------------------------------------------------
class NROM(Mapper):
    name = "NROM"


class MMC1(Mapper):
    name = "MMC1"

    def reset(self):
        self.sr, self.ctrl, self.c0, self.c1, self.p = 0x10, 0x0C, 0, 0, 0
        self.update()

    def cpu_write(self, addr, v):
        if v & 0x80:
            self.sr = 0x10
            self.ctrl |= 0x0C
            self.update()
            return
        done = self.sr & 1
        self.sr = (self.sr >> 1) | ((v & 1) << 4)
        if done:
            reg = (addr >> 13) & 3
            d = self.sr
            self.sr = 0x10
            if reg == 0:
                self.ctrl = d
            elif reg == 1:
                self.c0 = d
            elif reg == 2:
                self.c1 = d
            else:
                self.p = d
            self.update()

    def update(self):
        c = self.ctrl
        self.mirroring = (LO, HI, V, H)[c & 3]
        big = len(self.prg) > 0x40000                  # SUROM: CHR bit 4 picks 256K half
        hi = (self.c0 & 0x10) if big else 0
        p = (self.p & 0x0F) | hi
        mode = (c >> 2) & 3
        if mode < 2:
            self.prg32(p >> 1)
        elif mode == 2:
            self.prg16(0, hi)
            self.prg16(1, p)
        else:
            self.prg16(0, p)
            self.prg16(1, (hi | 0x0F) if big else self.last16)
        if c & 0x10:
            self.chr4(0, self.c0)
            self.chr4(1, self.c1)
        else:
            self.chr8(self.c0 >> 1)


class UxROM(Mapper):
    name = "UxROM"

    def reset(self):
        self.prg16(0, 0)
        self.prg16(1, self.last16)
        self.chr8(0)

    def cpu_write(self, addr, v):
        self.prg16(0, v)


class UxROM180(Mapper):
    name = "UxROM (fixed first bank)"

    def reset(self):
        self.prg16(0, 0)
        self.prg16(1, 0)
        self.chr8(0)

    def cpu_write(self, addr, v):
        self.prg16(1, v)


class CNROM(Mapper):
    name = "CNROM"

    def cpu_write(self, addr, v):
        self.chr8(v)


class MMC3(Mapper):
    name = "MMC3"
    has_scanline = True
    chr_mask = 0xFF

    def reset(self):
        self.cmd = 0
        self.regs = [0, 2, 4, 5, 6, 7, 0, 1]
        self.latch = self.counter = 0
        self.reload = False
        self.irq_enable = False
        self.irq_pending = False
        self.update()

    def cpu_write(self, addr, v):
        sel = (addr >> 13) & 3
        even = not (addr & 1)
        if sel == 0:
            if even:
                self.cmd = v
            else:
                self.regs[self.cmd & 7] = v
            self.update()
        elif sel == 1:
            if even and not self.cart.four_screen:
                self.mirroring = H if v & 1 else V
        elif sel == 2:
            if even:
                self.latch = v
            else:
                self.counter = 0
                self.reload = True
        else:
            if even:
                self.irq_enable = False
                self.irq_pending = False
            else:
                self.irq_enable = True

    def update(self):
        r = self.regs
        m = self.chr_mask
        inv = 4 if self.cmd & 0x80 else 0
        two, one = inv, inv ^ 4
        self.chr1(two, (r[0] & m) & 0xFE)
        self.chr1(two + 1, (r[0] & m) | 1)
        self.chr1(two + 2, (r[1] & m) & 0xFE)
        self.chr1(two + 3, (r[1] & m) | 1)
        for i in range(4):
            self.chr1(one + i, r[2 + i] & m)
        last = self.last8
        if self.cmd & 0x40:
            self.prg8(0, last - 1)
            self.prg8(2, r[6] & 0x3F)
        else:
            self.prg8(0, r[6] & 0x3F)
            self.prg8(2, last - 1)
        self.prg8(1, r[7] & 0x3F)
        self.prg8(3, last)

    def scanline(self):
        if self.counter == 0 or self.reload:
            self.counter = self.latch
            self.reload = False
        else:
            self.counter -= 1
        if self.counter == 0 and self.irq_enable:
            self.irq_pending = True


class TxSROM(MMC3):
    """Mapper 118: MMC3 where CHR bank bit 7 selects the nametable."""
    name = "TxSROM"
    chr_mask = 0x7F

    def update(self):
        super().update()
        r = self.regs
        if self.cmd & 0x80:
            src = [r[2], r[3], r[4], r[5]]
        else:
            src = [r[0], r[0], r[1], r[1]]
        self.nt_map = tuple(0x400 if (s >> 7) & 1 else 0 for s in src)

    def cpu_write(self, addr, v):
        if ((addr >> 13) & 3) == 1:           # mirroring register is not used
            return
        super().cpu_write(addr, v)


class NamcoDx(Mapper):
    """Mapper 206 (DxROM / Namco 108). Mapper 88 limits CHR banks like Namco 3433."""
    name = "DxROM / Namco 108"
    chr_hi = 0

    def reset(self):
        self.cmd = 0
        self.regs = [0, 2, 4, 5, 6, 7, 0, 1]
        self.update()

    def cpu_write(self, addr, v):
        if (addr >> 13) & 3:
            return
        if addr & 1:
            self.regs[self.cmd] = v
            self.update()
        else:
            self.cmd = v & 7

    def update(self):
        r = self.regs
        self.chr1(0, r[0] & 0x3E)
        self.chr1(1, (r[0] & 0x3E) | 1)
        self.chr1(2, r[1] & 0x3E)
        self.chr1(3, (r[1] & 0x3E) | 1)
        for i in range(4):
            self.chr1(4 + i, (r[2 + i] & 0x3F) | self.chr_hi)
        self.prg8(0, r[6] & 0x0F)
        self.prg8(1, r[7] & 0x0F)
        self.prg8(2, self.last8 - 1)
        self.prg8(3, self.last8)


class Namco3433(NamcoDx):
    name = "Namco 3433"
    chr_hi = 0x40


class AxROM(Mapper):
    name = "AxROM"

    def reset(self):
        self.prg32(0)
        self.chr8(0)
        self.mirroring = LO

    def cpu_write(self, addr, v):
        self.prg32(v & 0x0F)
        self.mirroring = HI if v & 0x10 else LO


class MMC2(Mapper):
    name = "MMC2"
    fetch_hook = True

    def reset(self):
        self.latch = [0xFD, 0xFD]
        self.regs = [0, 0, 0, 0]
        self.setup_prg()
        self.sync()

    def setup_prg(self):
        n = self.n_prg8
        self.prg8(0, 0)
        self.prg8(1, n - 3)
        self.prg8(2, n - 2)
        self.prg8(3, n - 1)

    def set_prg(self, v):
        self.prg8(0, v & 0x0F)

    def cpu_write(self, addr, v):
        sel = addr >> 12
        if sel == 0xA:
            self.set_prg(v)
        elif 0xB <= sel <= 0xE:
            self.regs[sel - 0xB] = v & 0x1F
            self.sync()
        elif sel == 0xF:
            self.mirroring = H if v & 1 else V

    def sync(self):
        r = self.regs
        self.chr4(0, r[1] if self.latch[0] == 0xFE else r[0])
        self.chr4(1, r[3] if self.latch[1] == 0xFE else r[2])

    def ppu_read(self, addr):
        v = self.chr[self.chr_off[addr >> 10] + (addr & 0x3FF)]
        m = addr & 0x1FF8
        if m == 0x0FD8:
            self.latch[0] = 0xFD
            self.sync()
        elif m == 0x0FE8:
            self.latch[0] = 0xFE
            self.sync()
        elif m == 0x1FD8:
            self.latch[1] = 0xFD
            self.sync()
        elif m == 0x1FE8:
            self.latch[1] = 0xFE
            self.sync()
        return v


class MMC4(MMC2):
    name = "MMC4"

    def setup_prg(self):
        self.prg16(0, 0)
        self.prg16(1, self.last16)

    def set_prg(self, v):
        self.prg16(0, v & 0x0F)


class ColorDreams(Mapper):
    name = "Color Dreams"

    def cpu_write(self, addr, v):
        self.prg32(v & 3)
        self.chr8(v >> 4)


# ---- Konami VRC family --------------------------------------------------------
class _VRCIrq(Mapper):
    """IRQ counter shared by VRC4 and VRC6."""

    def irq_reset(self):
        self.irq_latch = self.irq_counter = 0
        self.irq_a = self.irq_en = self.irq_mode = 0
        self.prescaler = 341
        self.irq_pending = False
        self.has_cpu_clock = False

    def irq_control(self, v):
        self.irq_a = v & 1
        self.irq_en = (v >> 1) & 1
        self.irq_mode = (v >> 2) & 1
        self.irq_pending = False
        if self.irq_en:
            self.irq_counter = self.irq_latch
            self.prescaler = 341
        self.has_cpu_clock = bool(self.irq_en)

    def irq_ack(self):
        self.irq_pending = False
        self.irq_en = self.irq_a
        self.has_cpu_clock = bool(self.irq_en)

    def _clk(self):
        if self.irq_counter == 0xFF:
            self.irq_counter = self.irq_latch
            self.irq_pending = True
        else:
            self.irq_counter += 1

    def cpu_clock(self, cycles):
        if self.irq_mode:
            for _ in range(cycles):
                self._clk()
        else:
            self.prescaler -= 3 * cycles
            while self.prescaler <= 0:
                self.prescaler += 341
                self._clk()


class VRC4(_VRCIrq):
    """Mappers 21, 22, 23, 25 (VRC2 / VRC4 in all their pin-out variants)."""
    name = "VRC2/VRC4"

    def reset(self):
        self.prg0 = self.prg1 = 0
        self.swap = 0
        self.chrr = [0] * 8
        self.chr_shift = 1 if self.cart.mapper_id == 22 else 0
        self.irq_reset()
        self.update()

    def decode(self, addr):
        mid = self.cart.mapper_id
        if mid == 21:
            a0 = ((addr >> 1) | (addr >> 6)) & 1
            a1 = ((addr >> 2) | (addr >> 7)) & 1
        elif mid == 22:
            a0 = (addr >> 1) & 1
            a1 = addr & 1
        elif mid == 23:
            a0 = (addr | (addr >> 2)) & 1
            a1 = ((addr >> 1) | (addr >> 3)) & 1
        else:
            a0 = ((addr >> 1) | (addr >> 3)) & 1
            a1 = (addr | (addr >> 2)) & 1
        return addr & 0xF000, a0 | (a1 << 1)

    def cpu_write(self, addr, v):
        base, r = self.decode(addr)
        if base == 0x8000:
            self.prg0 = v & 0x1F
        elif base == 0xA000:
            self.prg1 = v & 0x1F
        elif base == 0x9000:
            if r < 2:
                self.mirroring = _MIR_VRC[v & 3]
            elif r == 2:
                self.swap = (v >> 1) & 1
        elif 0xB000 <= base <= 0xE000:
            idx = (((base >> 12) - 0xB) << 1) | (r >> 1)
            if r & 1:
                self.chrr[idx] = (self.chrr[idx] & 0x0F) | ((v & 0x1F) << 4)
            else:
                self.chrr[idx] = (self.chrr[idx] & 0x1F0) | (v & 0x0F)
        elif base == 0xF000:
            if r == 0:
                self.irq_latch = (self.irq_latch & 0xF0) | (v & 0x0F)
            elif r == 1:
                self.irq_latch = (self.irq_latch & 0x0F) | ((v & 0x0F) << 4)
            elif r == 2:
                self.irq_control(v)
            else:
                self.irq_ack()
        self.update()

    def update(self):
        last = self.last8
        if self.swap:
            self.prg8(0, last - 1)
            self.prg8(2, self.prg0)
        else:
            self.prg8(0, self.prg0)
            self.prg8(2, last - 1)
        self.prg8(1, self.prg1)
        self.prg8(3, last)
        for i in range(8):
            self.chr1(i, self.chrr[i] >> self.chr_shift)


class VRC6(_VRCIrq):
    """Mappers 24 / 26 (banking + IRQ only; the extra VRC6 sound channels are not emulated)."""
    name = "VRC6 (no expansion audio)"

    def reset(self):
        self.chrr = [0] * 8
        self.irq_reset()
        self.prg16(0, 0)
        self.prg8(2, 0)
        self.prg8(3, self.last8)
        self.chr8(0)

    def cpu_write(self, addr, v):
        base = addr & 0xF000
        if self.cart.mapper_id == 26:
            r = ((addr & 1) << 1) | ((addr >> 1) & 1)
        else:
            r = addr & 3
        if base == 0x8000:
            self.prg16(0, v & 0x0F)
        elif base == 0xC000:
            self.prg8(2, v & 0x1F)
        elif base == 0xB000:
            if r == 3:
                self.mirroring = _MIR_VRC[(v >> 2) & 3]
        elif base == 0xD000 or base == 0xE000:
            idx = r + (0 if base == 0xD000 else 4)
            self.chrr[idx] = v
            self.chr1(idx, v)
        elif base == 0xF000:
            if r == 0:
                self.irq_latch = v
            elif r == 1:
                self.irq_control(v)
            elif r == 2:
                self.irq_ack()


class VRC1(Mapper):
    name = "VRC1"

    def reset(self):
        self.lo = [0, 0]
        self.hi = [0, 0]
        self.prg8(0, 0)
        self.prg8(1, 0)
        self.prg8(2, 0)
        self.prg8(3, self.last8)
        self.sync()

    def cpu_write(self, addr, v):
        sel = addr >> 12
        if sel == 0x8:
            self.prg8(0, v & 0x0F)
        elif sel == 0x9:
            self.mirroring = H if v & 1 else V
            self.hi = [(v >> 1) & 1, (v >> 2) & 1]
            self.sync()
        elif sel == 0xA:
            self.prg8(1, v & 0x0F)
        elif sel == 0xC:
            self.prg8(2, v & 0x0F)
        elif sel == 0xE:
            self.lo[0] = v & 0x0F
            self.sync()
        elif sel == 0xF:
            self.lo[1] = v & 0x0F
            self.sync()

    def sync(self):
        self.chr4(0, self.lo[0] | (self.hi[0] << 4))
        self.chr4(1, self.lo[1] | (self.hi[1] << 4))


class VRC3(Mapper):
    name = "VRC3"

    def reset(self):
        self.prg16(0, 0)
        self.prg16(1, self.last16)
        self.chr8(0)
        self.latch = self.counter = 0
        self.irq_a = self.irq_en = self.mode8 = 0
        self.has_cpu_clock = False

    def cpu_write(self, addr, v):
        sel = addr >> 12
        if 0x8 <= sel <= 0xB:
            sh = (sel - 8) * 4
            self.latch = (self.latch & ~(0xF << sh)) | ((v & 0xF) << sh)
        elif sel == 0xC:
            self.irq_a = v & 1
            self.irq_en = (v >> 1) & 1
            self.mode8 = (v >> 2) & 1
            self.irq_pending = False
            if self.irq_en:
                self.counter = self.latch
            self.has_cpu_clock = bool(self.irq_en)
        elif sel == 0xD:
            self.irq_pending = False
            self.irq_en = self.irq_a
            self.has_cpu_clock = bool(self.irq_en)
        elif sel == 0xF:
            self.prg16(0, v & 7)

    def cpu_clock(self, cycles):
        for _ in range(cycles):
            if self.mode8:
                if (self.counter & 0xFF) == 0xFF:
                    self.counter = (self.counter & 0xFF00) | (self.latch & 0xFF)
                    self.irq_pending = True
                else:
                    self.counter += 1
            elif self.counter == 0xFFFF:
                self.counter = self.latch
                self.irq_pending = True
            else:
                self.counter += 1


# ---- Taito ----------------------------------------------------------------------
class Taito33(Mapper):
    """Mapper 33 (TC0190)."""
    name = "Taito TC0190"
    is48 = False

    def reset(self):
        self.prg8(0, 0)
        self.prg8(1, 1)
        self.prg8(2, self.last8 - 1)
        self.prg8(3, self.last8)
        self.chr8(0)
        self.latch = self.counter = 0
        self.reload = False
        self.irq_enable = False

    def cpu_write(self, addr, v):
        area = (addr >> 13) & 3
        r = addr & 3
        if area == 0:
            if r == 0:
                self.prg8(0, v & 0x3F)
                if not self.is48:
                    self.mirroring = H if v & 0x40 else V
            elif r == 1:
                self.prg8(1, v & 0x3F)
            elif r == 2:
                self.chr2(0, v)
            else:
                self.chr2(1, v)
        elif area == 1:
            self.chr1(4 + r, v)
        elif self.is48:
            if area == 2:
                if r == 0:
                    self.latch = v ^ 0xFF
                elif r == 1:
                    self.counter = 0
                    self.reload = True
                elif r == 2:
                    self.irq_enable = True
                else:
                    self.irq_enable = False
                    self.irq_pending = False
            elif r == 0:
                self.mirroring = H if v & 0x40 else V


class Taito48(Taito33):
    name = "Taito TC0690"
    is48 = True
    has_scanline = True
    scanline = MMC3.scanline


# ---- Irem / Sunsoft -------------------------------------------------------------------
class IremH3001(Mapper):
    name = "Irem H3001"

    def reset(self):
        self.prg8(0, 0)
        self.prg8(1, 1)
        self.prg8(2, self.last8 - 1)
        self.prg8(3, self.last8)
        self.chr8(0)
        self.latch = self.counter = 0
        self.irq_enable = False
        self.has_cpu_clock = False

    def cpu_write(self, addr, v):
        if addr == 0x8000:
            self.prg8(0, v)
        elif addr == 0x9001:
            self.mirroring = H if v & 0x80 else V
        elif addr == 0x9003:
            self.irq_enable = bool(v & 0x80)
            self.irq_pending = False
            self.has_cpu_clock = self.irq_enable and self.counter > 0
        elif addr == 0x9004:
            self.counter = self.latch
            self.irq_pending = False
            self.has_cpu_clock = self.irq_enable and self.counter > 0
        elif addr == 0x9005:
            self.latch = (self.latch & 0x00FF) | (v << 8)
        elif addr == 0x9006:
            self.latch = (self.latch & 0xFF00) | v
        elif addr == 0xA000:
            self.prg8(1, v)
        elif 0xB000 <= addr <= 0xB007:
            self.chr1(addr & 7, v)
        elif addr == 0xC000:
            self.prg8(2, v)

    def cpu_clock(self, cycles):
        self.counter -= cycles
        if self.counter <= 0:
            self.counter = 0
            self.irq_pending = True
            self.has_cpu_clock = False


class Sunsoft3(Mapper):
    name = "Sunsoft 3"

    def reset(self):
        self.prg16(0, 0)
        self.prg16(1, self.last16)
        self.chr8(0)
        self.counter = 0
        self.toggle = 0
        self.irq_enable = False
        self.has_cpu_clock = False

    def cpu_write(self, addr, v):
        sel = addr & 0xF800
        if sel in (0x8800, 0x9800, 0xA800, 0xB800):
            self.chr2((sel >> 12) - 8, v)
        elif sel == 0xC800:
            if not self.toggle:
                self.counter = (self.counter & 0x00FF) | (v << 8)
            else:
                self.counter = (self.counter & 0xFF00) | v
            self.toggle ^= 1
        elif sel == 0xD800:
            self.irq_enable = bool(v & 0x10)
            self.toggle = 0
            self.irq_pending = False
            self.has_cpu_clock = self.irq_enable
        elif sel == 0xE800:
            self.mirroring = _MIR_VRC[v & 3]
        elif sel == 0xF800:
            self.prg16(0, v)

    def cpu_clock(self, cycles):
        self.counter -= cycles
        if self.counter < 0:
            self.counter &= 0xFFFF
            self.irq_pending = True
            self.irq_enable = False
            self.has_cpu_clock = False


class FME7(Mapper):
    name = "Sunsoft FME-7 (no expansion audio)"

    def reset(self):
        self.cmd = 0
        self.prg8(0, 0)
        self.prg8(1, 1)
        self.prg8(2, 2)
        self.prg8(3, self.last8)
        self.chr8(0)
        self.counter = 0
        self.irq_enable = False
        self.counter_enable = False
        self.has_cpu_clock = False

    def cpu_write(self, addr, v):
        sel = (addr >> 13) & 3
        if sel == 0:
            self.cmd = v & 0x0F
        elif sel == 1:
            c = self.cmd
            if c < 8:
                self.chr1(c, v)
            elif c == 9:
                self.prg8(0, v & 0x3F)
            elif c == 10:
                self.prg8(1, v & 0x3F)
            elif c == 11:
                self.prg8(2, v & 0x3F)
            elif c == 12:
                self.mirroring = _MIR_VRC[v & 3]
            elif c == 13:
                self.irq_enable = bool(v & 1)
                self.counter_enable = bool(v & 0x80)
                self.irq_pending = False
                self.has_cpu_clock = self.counter_enable
            elif c == 14:
                self.counter = (self.counter & 0xFF00) | v
            elif c == 15:
                self.counter = (self.counter & 0x00FF) | (v << 8)

    def cpu_clock(self, cycles):
        n = self.counter - cycles
        if n < 0:
            if self.irq_enable:
                self.irq_pending = True
            n &= 0xFFFF
        self.counter = n


# ---- simple discrete-logic boards -------------------------------------------------
class GxROM(Mapper):
    name = "GxROM"

    def cpu_write(self, addr, v):
        self.prg32((v >> 4) & 3)
        self.chr8(v & 3)


class Mapper34(Mapper):
    name = "BNROM / NINA-001"

    def cpu_write(self, addr, v):
        if self.chr_is_ram:
            self.prg32(v)

    def low_write(self, addr, v):
        if not self.chr_is_ram:
            if addr == 0x7FFD:
                self.prg32(v)
            elif addr == 0x7FFE:
                self.chr4(0, v)
            elif addr == 0x7FFF:
                self.chr4(1, v)


class _Disc16(Mapper):
    """Shared reset for boards with a switchable first 16K and fixed last 16K."""

    def reset(self):
        self.prg16(0, 0)
        self.prg16(1, self.last16)
        self.chr8(0)


class Mapper70(_Disc16):
    name = "Bandai 74161"

    def cpu_write(self, addr, v):
        self.prg16(0, v >> 4)
        self.chr8(v & 0x0F)


class Mapper152(_Disc16):
    name = "Bandai 74161 (single-screen)"

    def cpu_write(self, addr, v):
        self.prg16(0, (v >> 4) & 7)
        self.chr8(v & 0x0F)
        self.mirroring = HI if v & 0x80 else LO


class Mapper78(_Disc16):
    name = "Holy Diver board"

    def cpu_write(self, addr, v):
        self.prg16(0, v & 7)
        self.chr8(v >> 4)
        self.mirroring = V if v & 8 else H


class Camerica(_Disc16):
    name = "Camerica / Codemasters"

    def cpu_write(self, addr, v):
        if addr >= 0xC000:
            self.prg16(0, v)
        elif 0x9000 <= addr < 0xA000:
            self.mirroring = HI if v & 0x10 else LO


class Mapper79(Mapper):
    name = "NINA-03 / NINA-06"

    def low_write(self, addr, v):
        if (addr & 0xE100) == 0x4100:
            self.prg32((v >> 3) & 1)
            self.chr8(v & 7)


class Mapper87(Mapper):
    name = "Jaleco / Konami CHR swap"

    def low_write(self, addr, v):
        if addr >= 0x6000:
            self.chr8(((v & 1) << 1) | ((v >> 1) & 1))


class Mapper89(_Disc16):
    name = "Sunsoft 2 (single-screen)"

    def cpu_write(self, addr, v):
        self.prg16(0, (v >> 4) & 7)
        self.chr8((v & 7) | ((v >> 4) & 8))
        self.mirroring = HI if v & 8 else LO


class Mapper93(_Disc16):
    name = "Sunsoft 2 (CHR-RAM)"

    def cpu_write(self, addr, v):
        self.prg16(0, (v >> 4) & 7)


class Mapper94(_Disc16):
    name = "UN1ROM"

    def cpu_write(self, addr, v):
        self.prg16(0, (v >> 2) & 0x0F)


class Mapper140(Mapper):
    name = "Jaleco JF-11/14"

    def low_write(self, addr, v):
        if addr >= 0x6000:
            self.prg32((v >> 4) & 3)
            self.chr8(v & 0x0F)


class Mapper184(Mapper):
    name = "Sunsoft 1"

    def low_write(self, addr, v):
        if addr >= 0x6000:
            self.chr4(0, v & 7)
            self.chr4(1, 4 | ((v >> 4) & 7))


class Mapper116(Mapper):
    """SOMARI-P / Huang-1 board (Somari, AV Girl Fighting, Kart Fighter, Garo Densetsu Special...).

    One cartridge that can behave like a VRC2b, an MMC3 or an MMC1, chosen by the
    supervisor register at $4100 (any address with A14 and A8 set below $8000):
        bits 0-1: 0 = VRC2, 1 = MMC3, 2/3 = MMC1      bit 2: outer 256 KiB CHR bank
    Each chip keeps its own registers, so switching modes swaps the whole state.
    """
    name = "SOMARI-P (116)"
    has_scanline = True                      # MMC3 mode IRQ; scanline() ignores other modes

    def reset(self):
        self.mode = 1                         # power-on value $01 = MMC3 mode (Somari boots in it)
        self.vrc_chr = [0xFF, 0xFF, 0xFF, 0xFF, 4, 5, 6, 7]    # power-up $FF is relied on by Somari
        self.vrc_prg = [0, 1]
        self.vrc_mir = 0
        self.m3 = [0, 2, 4, 5, 6, 7, -4, -3, -2, -1]            # 6 CHR, 2 PRG, 2 fixed PRG
        self.m3_ctrl = 0
        self.m3_mir = 1 if self.cart.mirroring == 0 else 0      # header: 0 = horizontal
        self.irq_latch = 0
        self.irq_counter = 0
        self.irq_reload = False
        self.irq_enable = False
        self.irq_pending = False
        self.m1 = [0x0C, 0, 0, 0]
        self.m1_buf = 0
        self.m1_shift = 0
        self.update()

    # ---- helpers ---------------------------------------------------------
    def _p16(self, slot, bank8):             # 16 KiB at slot (0=$8000, 1=$C000), given in 8 KiB units
        self.prg8(slot * 2, bank8)
        self.prg8(slot * 2 + 1, bank8 + 1)

    def update(self):
        self._update_prg()
        self._update_chr()
        self._update_mirroring()

    def _update_prg(self):
        m = self.mode & 3
        if m == 0:
            self.prg8(0, self.vrc_prg[0])
            self.prg8(1, self.vrc_prg[1])
            self.prg8(2, -2)
            self.prg8(3, -1)
        elif m == 1:
            r = self.m3
            pm = (self.m3_ctrl >> 5) & 2
            self.prg8(0, r[6 + pm])
            self.prg8(1, r[7])
            self.prg8(2, r[6 + (pm ^ 2)])
            self.prg8(3, r[9])
        else:
            r = self.m1
            bank = r[3] & 0x0F
            if r[0] & 0x08:
                if r[0] & 0x04:
                    self._p16(0, bank << 1)
                    self._p16(1, 0x0F << 1)
                else:
                    self._p16(0, 0)
                    self._p16(1, bank << 1)
            else:
                base = (bank & 0xFE) << 1
                for i in range(4):
                    self.prg8(i, base + i)

    def _update_chr(self):
        m = self.mode & 3
        outer = (self.mode & 0x04) << 6
        if m == 0:
            for i in range(8):
                self.chr1(i, outer | self.vrc_chr[i])
        elif m == 1:
            r = self.m3
            sw = 4 if self.m3_ctrl & 0x80 else 0
            self.chr1(0 ^ sw, outer | (r[0] & 0xFE))
            self.chr1(1 ^ sw, outer | (r[0] | 1))
            self.chr1(2 ^ sw, outer | (r[1] & 0xFE))
            self.chr1(3 ^ sw, outer | (r[1] | 1))
            self.chr1(4 ^ sw, outer | r[2])
            self.chr1(5 ^ sw, outer | r[3])
            self.chr1(6 ^ sw, outer | r[4])
            self.chr1(7 ^ sw, outer | r[5])
        else:
            r = self.m1
            if r[0] & 0x10:
                self.chr4(0, r[1])
                self.chr4(1, r[2])
            else:
                base = (r[1] & 0xFE) << 2
                for i in range(8):
                    self.chr1(i, base + i)

    def _update_mirroring(self):
        m = self.mode & 3
        if m == 0:
            self.mirroring = H if self.vrc_mir & 1 else V
        elif m == 1:
            self.mirroring = H if self.m3_mir & 1 else V
        else:
            self.mirroring = (LO, HI, V, H)[self.m1[0] & 3]

    # ---- registers ----------------------------------------------------------
    def low_write(self, addr, v):
        if (addr & 0xE100) == 0x4100:         # $4100-$5FFF with A8 set
            self.mode = v
            if addr & 1:                      # switching modes resets the MMC1 shifter (Garo Densetsu needs this)
                self.m1[0] = 0x0C
                self.m1[3] = 0
                self.m1_buf = self.m1_shift = 0
            self.update()

    def cpu_write(self, addr, v):
        m = self.mode & 3
        if m == 0:
            self._write_vrc2(addr, v)
        elif m == 1:
            self._write_mmc3(addr, v)
        else:
            self._write_mmc1(addr, v)

    def _write_vrc2(self, addr, v):
        if 0xB000 <= addr <= 0xE003:
            idx = ((((addr & 0x02) | (addr >> 10)) >> 1) + 2) & 7
            sh = (addr & 1) << 2
            self.vrc_chr[idx] = (self.vrc_chr[idx] & (0xF0 >> sh)) | ((v & 0x0F) << sh)
            self._update_chr()
        else:
            reg = addr & 0xF000
            if reg == 0x8000:
                self.vrc_prg[0] = v
                self._update_prg()
            elif reg == 0xA000:
                self.vrc_prg[1] = v
                self._update_prg()
            elif reg == 0x9000:
                self.vrc_mir = v
                self._update_mirroring()

    def _write_mmc3(self, addr, v):
        reg = addr & 0xE001
        if reg == 0x8000:
            self.m3_ctrl = v
            self.update()
        elif reg == 0x8001:
            self.m3[self.m3_ctrl & 7] = v
            self.update()
        elif reg == 0xA000:
            self.m3_mir = v
            self.update()
        elif reg == 0xC000:
            self.irq_latch = v
        elif reg == 0xC001:
            self.irq_reload = True
        elif reg == 0xE000:
            self.irq_enable = False
            self.irq_pending = False
        elif reg == 0xE001:
            self.irq_enable = True

    def _write_mmc1(self, addr, v):
        if v & 0x80:
            self.m1[0] |= 0x0C
            self.m1_buf = self.m1_shift = 0
            self.update()
        else:
            self.m1_buf |= (v & 1) << self.m1_shift
            self.m1_shift += 1
            if self.m1_shift == 5:
                self.m1[((addr >> 13) - 4) & 3] = self.m1_buf
                self.m1_buf = self.m1_shift = 0
                self.update()

    def scanline(self):
        if (self.mode & 3) != 1:
            return
        if self.irq_counter == 0 or self.irq_reload:
            self.irq_counter = self.irq_latch
        else:
            self.irq_counter -= 1
        if self.irq_counter == 0 and self.irq_enable:
            self.irq_pending = True
        self.irq_reload = False


MAPPERS = {
    0: NROM, 1: MMC1, 2: UxROM, 3: CNROM, 4: MMC3, 7: AxROM, 9: MMC2, 10: MMC4, 11: ColorDreams,
    21: VRC4, 22: VRC4, 23: VRC4, 24: VRC6, 25: VRC4, 26: VRC6, 33: Taito33, 34: Mapper34,
    48: Taito48, 65: IremH3001, 66: GxROM, 67: Sunsoft3, 69: FME7, 70: Mapper70, 71: Camerica,
    73: VRC3, 75: VRC1, 78: Mapper78, 79: Mapper79, 87: Mapper87, 88: Namco3433, 89: Mapper89,
    93: Mapper93, 94: Mapper94, 116: Mapper116, 118: TxSROM, 140: Mapper140, 146: Mapper79, 152: Mapper152,
    180: UxROM180, 184: Mapper184, 206: NamcoDx,
}


def create_mapper(cart):
    cls = MAPPERS.get(cart.mapper_id)
    if cls is None:
        print(f"WARNING: Mapper {cart.mapper_id} not implemented, falling back to NROM.")
        cls = NROM
    return cls(cart)

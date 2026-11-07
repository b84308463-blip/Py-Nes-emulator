import numpy as np


def _build_pixel_table():
    bits = (np.arange(256)[:, None] >> np.arange(7, -1, -1)[None, :]) & 1      # (256, 8)
    table = bits[:, None, :] | (bits[None, :, :] << 1)                          # [lo][hi] -> 8 pixels
    return table.reshape(65536, 8).astype(np.uint8)


PIX8 = _build_pixel_table()

# nametable index -> byte offset in VRAM, for each mirroring mode
MIRROR = {
    0: (0, 0, 0x400, 0x400),        # horizontal
    1: (0, 0x400, 0, 0x400),        # vertical
    2: (0, 0, 0, 0),                # single screen, low
    3: (0x400,) * 4,                # single screen, high
    4: (0, 0x400, 0x800, 0xC00),    # four-screen
}

_HEX = [
    0x7C7C7C, 0x0000FC, 0x0000BC, 0x4428BC, 0x940084, 0xA80020, 0xA81000, 0x881400,
    0x503000, 0x007800, 0x006800, 0x005800, 0x004058, 0x000000, 0x000000, 0x000000,
    0xBCBCBC, 0x0078F8, 0x0058F8, 0x6844FC, 0xD800CC, 0xE40058, 0xF83800, 0xE45C10,
    0xAC7C00, 0x00B800, 0x00A800, 0x00A844, 0x008888, 0x000000, 0x000000, 0x000000,
    0xF8F8F8, 0x3CBCFC, 0x6888FC, 0x9878F8, 0xF878F8, 0xF85898, 0xF87858, 0xFCA044,
    0xF8B800, 0xB8F818, 0x58D854, 0x58F898, 0x00E8D8, 0x787878, 0x000000, 0x000000,
    0xFCFCFC, 0xA4E4FC, 0xB8B8F8, 0xD8B8F8, 0xF8B8F8, 0xF8A4C0, 0xF0D0B0, 0xFCE0A8,
    0xF8D878, 0xD8F878, 0xB8F8B8, 0xB8F8D8, 0x00FCFC, 0xF8D8F8, 0x000000, 0x000000,
]
PALETTE_RGB = np.array([[(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF] for h in _HEX], dtype=np.uint8)

_EVENT_POINTS = (1, 256, 257, 260, 281, 341)
_AR33 = np.arange(33, dtype=np.intp)
_AR8 = np.arange(8, dtype=np.intp)


class PPU:
    def __init__(self, bus, region):
        self.bus = bus
        self.region = region
        self.lines = region.lines
        self.vblank_line = region.vblank_line
        self.pre_line = region.pre_line

        self.ctrl = self.mask = self.status = 0
        self.oam_addr = 0
        self.v = self.t = self.x = self.w = 0
        self.read_buffer = 0
        self.latch = 0

        self.cycle = 0
        self.scanline = 0
        self.frame = 0
        self.nmi_pending = False
        self.line_v = 0
        self.s0_dot = None
        self.next_c = 1
        self.skip_render = False

        self.vram = bytearray(4096)
        self.palette = bytearray([0x0F] * 32)
        self.oam = bytearray(256)
        self.pal_np = np.frombuffer(self.palette, dtype=np.uint8)
        self.oam_np = np.frombuffer(self.oam, dtype=np.uint8)
        self.frame_buf = np.zeros((240, 256), dtype=np.uint8)
        self.mapper = None
        self.cart_mirroring = bus.cart.mirroring
        self._bg_cache = {}
        self._miss = 0
        self.oam_dirty = True
        self.spr_lines = None
        self.spr_h = 8

    def attach(self):
        self.mapper = self.bus.mapper

    def reset(self):
        self.ctrl = self.mask = self.status = 0
        self.oam_addr = 0
        self.v = self.t = self.x = self.w = 0
        self.read_buffer = 0
        self.cycle = 0
        self.scanline = 0
        self.nmi_pending = False
        self.s0_dot = None
        self.next_c = 1

    def nt_table(self):
        m = self.mapper
        if m.nt_map is not None:
            return m.nt_map
        mode = m.mirroring
        if mode is None:
            mode = self.cart_mirroring
        return MIRROR[mode]

    @staticmethod
    def _pal_index(addr):
        i = addr & 0x1F
        if (i & 0x13) == 0x10:
            i &= 0x0F
        return i

    def read_vram(self, addr):
        addr &= 0x3FFF
        if addr < 0x2000:
            return self.mapper.ppu_read(addr)
        if addr < 0x3F00:
            return self.vram[self.nt_table()[(addr >> 10) & 3] + (addr & 0x3FF)]
        return self.palette[self._pal_index(addr)] & 0x3F

    def write_vram(self, addr, val):
        addr &= 0x3FFF
        if addr < 0x3F00 and self._bg_cache:
            self._bg_cache.clear()
        if addr < 0x2000:
            self.mapper.ppu_write(addr, val)
        elif addr < 0x3F00:
            self.vram[self.nt_table()[(addr >> 10) & 3] + (addr & 0x3FF)] = val
        else:
            self.palette[self._pal_index(addr)] = val & 0x3F

    def read_register(self, addr):
        if addr == 2:
            res = (self.status & 0xE0) | (self.latch & 0x1F)
            self.status &= 0x7F
            self.w = 0
            self.latch = res
            return res
        if addr == 4:
            self.latch = self.oam[self.oam_addr]
            return self.latch
        if addr == 7:
            a = self.v & 0x3FFF
            if a >= 0x3F00:
                val = self.read_vram(a)
                self.read_buffer = self.vram[self.nt_table()[(a >> 10) & 3] + (a & 0x3FF)]
            else:
                val = self.read_buffer
                self.read_buffer = self.read_vram(a)
            self.v = (self.v + (32 if self.ctrl & 0x04 else 1)) & 0x7FFF
            self.latch = val
            return val
        return self.latch

    def write_register(self, addr, val):
        self.latch = val
        if addr == 0:
            if (val & 0x80) and not (self.ctrl & 0x80) and (self.status & 0x80):
                self.nmi_pending = True
            elif not (val & 0x80):
                self.nmi_pending = False
            self.ctrl = val
            self.t = (self.t & 0xF3FF) | ((val & 0x03) << 10)
        elif addr == 1:
            self.mask = val
        elif addr == 3:
            self.oam_addr = val
        elif addr == 4:
            self.oam[self.oam_addr] = val
            self.oam_dirty = True
            self.oam_addr = (self.oam_addr + 1) & 0xFF
        elif addr == 5:
            if self.w == 0:
                self.t = (self.t & 0xFFE0) | (val >> 3)
                self.x = val & 0x07
                self.w = 1
            else:
                self.t = (self.t & 0x8C1F) | ((val & 0x07) << 12) | ((val & 0xF8) << 2)
                self.w = 0
        elif addr == 6:
            if self.w == 0:
                self.t = (self.t & 0x00FF) | ((val & 0x3F) << 8)
                self.w = 1
            else:
                self.t = (self.t & 0xFF00) | val
                self.v = self.t
                self.w = 0
        elif addr == 7:
            self.write_vram(self.v, val)
            self.v = (self.v + (32 if self.ctrl & 0x04 else 1)) & 0x7FFF

    def render_scanline(self, y):
        if y == 1:
            self._miss = 0
        mask = self.mask
        ctrl = self.ctrl
        pal = self.palette
        backdrop = pal[0] & 0x3F
        row = self.frame_buf[y]
        show_bg = mask & 0x08
        show_spr = mask & 0x10

        if not (show_bg or show_spr):
            row[:] = backdrop & 0x30 if mask & 1 else backdrop
            return

        big = ctrl & 0x20
        h = 16 if big else 8
        oam = self.oam

        # --- THE FIX IS RIGHT HERE ---
        if self.skip_render:
            # frameskip: the picture isn't shown, only sprite-0 timing matters
            if not (show_bg and show_spr and 0 <= y - 1 - oam[0] < h):
                return
        # -----------------------------

        mapper = self.mapper
        chr_ = mapper.chr
        cmap = mapper.chr_off
        hook = mapper.fetch_hook
        rd = mapper.ppu_read

        bgpix = None
        if show_bg:
            v = self.line_v
            fine_y = (v >> 12) & 7
            coarse_y = (v >> 5) & 31
            coarse_x = v & 31
            nt_sel = (v >> 10) & 3
            ntm = self.nt_table()
            table = 0x1000 if ctrl & 0x10 else 0
            fx = self.x
            if hook:
                # MMC2/4: pattern fetches have side effects -> never cache
                vram = np.frombuffer(self.vram, dtype=np.uint8)
                row_off = coarse_y << 5
                attr_off = 0x3C0 + ((coarse_y >> 2) << 3)
                ashift = (coarse_y & 2) << 1
                base = np.full(33, ntm[nt_sel], dtype=np.intp)
                base[32 - coarse_x:] = ntm[nt_sel ^ 1]
                cxm = (_AR33 + coarse_x) & 31
                tiles = vram[base + row_off + cxm]
                attr = vram[base + attr_off + (cxm >> 2)]
                pss = ((attr >> (ashift | (cxm & 2))) & 3) << 2
                a = table + (tiles.astype(np.intp) << 4) + fine_y
                los = [0] * 33
                his = [0] * 33
                for t, x in enumerate(a.tolist()):
                    los[t] = rd(x)
                    his[t] = rd(x + 8)
                idx16 = (np.array(los, dtype=np.intp) << 8) | np.array(his, dtype=np.intp)
                pix = PIX8[idx16].reshape(-1)[fx:fx + 256]
                ps = np.repeat(pss.astype(np.uint8), 8)[fx:fx + 256]
            else:
                cache = self._bg_cache
                key = (v & 0x0FFF, table, ntm, tuple(cmap))
                blk = cache.get(key)
                if blk is not None:
                    self._miss = 0
                    pix = blk[0][fine_y, fx:fx + 256]
                    ps = blk[1][fx:fx + 256]
                else:
                    self._miss += 1
                    vram = np.frombuffer(self.vram, dtype=np.uint8)
                    row_off = coarse_y << 5
                    attr_off = 0x3C0 + ((coarse_y >> 2) << 3)
                    ashift = (coarse_y & 2) << 1
                    base = np.full(33, ntm[nt_sel], dtype=np.intp)
                    base[32 - coarse_x:] = ntm[nt_sel ^ 1]
                    cxm = (_AR33 + coarse_x) & 31
                    tiles = vram[base + row_off + cxm]
                    attr = vram[base + attr_off + (cxm >> 2)]
                    pss = ((attr >> (ashift | (cxm & 2))) & 3) << 2
                    chr_np = np.frombuffer(chr_, dtype=np.uint8)
                    cm = np.array(cmap, dtype=np.intp)
                    t16 = table + (tiles.astype(np.intp) << 4)
                    if self._miss > 2:
                        # scroll/banks change every line: caching can't help, decode just this line
                        a = t16 + fine_y
                        off = cm[a >> 10] + (a & 0x3FF)
                        idx16 = (chr_np[off].astype(np.intp) << 8) | chr_np[off + 8]
                        pix = PIX8[idx16].reshape(-1)[fx:fx + 256]
                        ps = np.repeat(pss.astype(np.uint8), 8)[fx:fx + 256]
                    else:
                        # decode this tile row for all 8 fine-Y values at once
                        a = t16[None, :] + _AR8[:, None]            # (8, 33)
                        off = cm[a >> 10] + (a & 0x3FF)
                        idx16 = (chr_np[off].astype(np.intp) << 8) | chr_np[off + 8]
                        blk = (PIX8[idx16].reshape(8, 264), np.repeat(pss.astype(np.uint8), 8))
                        if len(cache) > 600:
                            cache.clear()
                        cache[key] = blk
                        pix = blk[0][fine_y, fx:fx + 256]
                        ps = blk[1][fx:fx + 256]
            if not (mask & 0x02):
                pix = pix.copy()
                pix[:8] = 0
            row[:] = np.where(pix != 0, self.pal_np[ps + pix], backdrop)
            bgpix = pix
        else:
            row[:] = backdrop

        self.s0_dot = None
        if show_spr:
            if self.oam_dirty or self.spr_lines is None or self.spr_h != h:
                lines = [[] for _ in range(272)]
                for i in range(64):
                    y0 = oam[i * 4] + 1
                    for r in range(h):
                        lines[y0 + r].append(i)
                self.spr_lines = lines
                self.spr_h = h
                self.oam_dirty = False
            idxs = self.spr_lines[y]
            n = len(idxs)
            if n:
                if n > 8:
                    self.status |= 0x20
                    idxs = idxs[:8]
                bgl = bgpix.tolist() if bgpix is not None else None
                filled = [False] * 256
                left_clip = not (mask & 0x04)
                for i in idxs:
                    o = i * 4
                    r = y - 1 - oam[o]
                    tile = oam[o + 1]
                    attr = oam[o + 2]
                    sx0 = oam[o + 3]
                    if attr & 0x80:
                        r = h - 1 - r
                    if big:
                        table = (tile & 1) * 0x1000
                        idx = tile & 0xFE
                        if r >= 8:
                            idx += 1
                            r -= 8
                    else:
                        table = 0x1000 if ctrl & 0x08 else 0
                        idx = tile
                    a = table + (idx << 4) + r
                    if hook:
                        lo = rd(a)
                        hi = rd(a + 8)
                    else:
                        oo = cmap[a >> 10] + (a & 0x3FF)
                        lo = chr_[oo]
                        hi = chr_[oo + 8]
                    if not (lo | hi):
                        continue
                    hflip = attr & 0x40
                    behind = attr & 0x20
                    base_pal = 0x10 + ((attr & 3) << 2)
                    for px in range(8):
                        sx = sx0 + px
                        if sx > 255:
                            break
                        bit = px if hflip else 7 - px
                        p = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
                        if not p:
                            continue
                        if left_clip and sx < 8:
                            continue
                        if filled[sx]:          # lower OAM index already won this pixel
                            continue
                        filled[sx] = True
                        bgo = bgl is not None and bgl[sx]
                        if i == 0 and bgo and sx != 255 and self.s0_dot is None:
                            self.s0_dot = sx + 2
                        if not (behind and bgo):
                            row[sx] = pal[base_pal + p] & 0x3F
        if mask & 1:
            row &= 0x30

    def _calc_next(self):
        c = self.cycle
        nx = 341
        for p in _EVENT_POINTS:
            if p > c:
                nx = p
                break
        s0 = self.s0_dot
        if s0 is not None and c < s0 < nx:
            nx = s0
        self.next_c = nx

    def run(self, n):
        c = self.cycle + n
        if c < self.next_c:
            self.cycle = c
            return
        self._run_slow(n)

    def _run_slow(self, n):
        while n > 0:
            c = self.cycle
            span = 341 - c
            if n < span:
                span = n
            new = c + span
            self._events(c, new)
            n -= span
            self.cycle = new
            if new >= 341:
                self.cycle = 0
                self.scanline += 1
                if self.scanline >= self.lines:
                    self.scanline = 0
                    self.frame += 1
        self._calc_next()

    def _events(self, c, new):
        sl = self.scanline
        if c < 1 <= new:
            if sl == self.vblank_line:
                self.status |= 0x80
                if self.ctrl & 0x80:
                    self.nmi_pending = True
            elif sl == self.pre_line:
                self.status &= 0x1F
                self.nmi_pending = False
            if sl < 240:
                self.s0_dot = None
                self.line_v = self.v
                self.render_scanline(sl)

        if sl < 240 and self.s0_dot is not None and c < self.s0_dot <= new:
            self.status |= 0x40
            self.s0_dot = None

        if self.mask & 0x18 and (sl < 240 or sl == self.pre_line):
            if c < 256 <= new:
                v = self.v
                if (v & 0x7000) != 0x7000:
                    v += 0x1000
                else:
                    v &= ~0x7000
                    y = (v & 0x03E0) >> 5
                    if y == 29:
                        y = 0
                        v ^= 0x0800
                    elif y == 31:
                        y = 0
                    else:
                        y += 1
                    v = (v & ~0x03E0) | (y << 5)
                self.v = v
            if c < 257 <= new:
                self.v = (self.v & 0x7BE0) | (self.t & 0x041F)
            if sl == self.pre_line and c < 305 and new > 280:
                self.v = (self.v & 0x041F) | (self.t & 0x7BE0)
            if c < 260 <= new and self.mapper.has_scanline:
                self.mapper.scanline()

    def get_palette_array(self):
        return PALETTE_RGB

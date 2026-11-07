"""2A03 (6502 without decimal mode) core.

Each opcode is a small function generated from templates at start-up, so a
single instruction costs one dictionary-free table lookup + one call.
Includes the commonly used unofficial opcodes (LAX, SAX, DCP, ISC, SLO, RLA,
SRE, RRA, multi-byte NOPs, SBC $EB).
"""
import re

# --- addressing modes: each snippet leaves the effective address in `a` ------
_FETCH16 = ("pc = self.pc\n"
            "if pc >= 0x8000 and (pc & 0x1FFF) != 0x1FFF:\n"
            "    o = poff[(pc >> 13) & 3] + (pc & 0x1FFF)\n"
            "    {t} = prg[o] | (prg[o + 1] << 8)\n"
            "else:\n"
            "    {t} = rd(pc) | (rd((pc + 1) & 0xFFFF) << 8)\n"
            "self.pc = (pc + 2) & 0xFFFF\n")
# 1-byte operand at PC -> `{t}` (PRG is read directly, everything else through the bus)
_FETCH8 = ("pc = self.pc\n"
           "{t} = prg[poff[(pc >> 13) & 3] + (pc & 0x1FFF)] if pc >= 0x8000 else rd(pc)\n"
           "self.pc = (pc + 1) & 0xFFFF\n")
_PEN = "if (b ^ a) & 0xFF00:\n    self.cycles += 1\n"
MODE = {
    'imm': "a = self.pc\nself.pc = (a + 1) & 0xFFFF\n",
    'zp': _FETCH8.format(t="a") + "#ZP\n",
    'zpx': _FETCH8.format(t="a") + "a = (a + self.x) & 0xFF\n#ZP\n",
    'zpy': _FETCH8.format(t="a") + "a = (a + self.y) & 0xFF\n#ZP\n",
    'abs': _FETCH16.format(t="a"),
    'abx': _FETCH16.format(t="b") + "a = (b + self.x) & 0xFFFF\n" + _PEN,
    'abxw': _FETCH16.format(t="b") + "a = (b + self.x) & 0xFFFF\n",
    'aby': _FETCH16.format(t="b") + "a = (b + self.y) & 0xFFFF\n" + _PEN,
    'abyw': _FETCH16.format(t="b") + "a = (b + self.y) & 0xFFFF\n",
    'izx': (_FETCH8.format(t="z") + "z = (z + self.x) & 0xFF\n"
            "a = ram[z] | (ram[(z + 1) & 0xFF] << 8)\n"),
    'izy': (_FETCH8.format(t="z") +
            "b = ram[z] | (ram[(z + 1) & 0xFF] << 8)\na = (b + self.y) & 0xFFFF\n" + _PEN),
    'izyw': (_FETCH8.format(t="z") +
             "b = ram[z] | (ram[(z + 1) & 0xFF] << 8)\na = (b + self.y) & 0xFFFF\n"),
}
WMODE = {'abx': 'abxw', 'aby': 'abyw', 'izy': 'izyw'}

CYC_READ = {'imm': 2, 'zp': 3, 'zpx': 4, 'zpy': 4, 'abs': 4, 'abx': 4, 'aby': 4, 'izx': 6, 'izy': 5}
CYC_WRITE = {'zp': 3, 'zpx': 4, 'zpy': 4, 'abs': 4, 'abx': 5, 'aby': 5, 'izx': 6, 'izy': 6}
CYC_RMW = {'zp': 5, 'zpx': 6, 'abs': 6, 'abx': 7, 'aby': 7, 'izx': 8, 'izy': 8}

NZ = "self.fz = r == 0\nself.fn = r >> 7\n"
ADC = ("t = self.a + v + self.fc\nr = t & 0xFF\nself.fc = t >> 8\n"
       "self.fv = ((self.a ^ r) & (v ^ r)) >> 7\nself.a = r\n" + NZ)


def _cmp(reg):
    return f"t = self.{reg} - v\nself.fc = t >= 0\nr = t & 0xFF\n" + NZ


# bodies for instructions that read an operand into `v`
READ = {
    'LDA': "r = v\nself.a = r\n" + NZ,
    'LDX': "r = v\nself.x = r\n" + NZ,
    'LDY': "r = v\nself.y = r\n" + NZ,
    'LAX': "r = v\nself.a = r\nself.x = r\n" + NZ,
    'AND': "r = self.a & v\nself.a = r\n" + NZ,
    'ORA': "r = self.a | v\nself.a = r\n" + NZ,
    'EOR': "r = self.a ^ v\nself.a = r\n" + NZ,
    'ADC': ADC,
    'SBC': "v ^= 0xFF\n" + ADC,
    'CMP': _cmp('a'), 'CPX': _cmp('x'), 'CPY': _cmp('y'),
    'BIT': "self.fz = (self.a & v) == 0\nself.fn = v >> 7\nself.fv = (v >> 6) & 1\n",
    'NOP': "pass\n",
}

# read-modify-write: compute new value `r` from `v`
SHIFT = {
    'ASL': "self.fc = v >> 7\nr = (v << 1) & 0xFF\n",
    'LSR': "self.fc = v & 1\nr = v >> 1\n",
    'ROL': "c = self.fc\nself.fc = v >> 7\nr = ((v << 1) | c) & 0xFF\n",
    'ROR': "c = self.fc\nself.fc = v & 1\nr = (v >> 1) | (c << 7)\n",
    'INC': "r = (v + 1) & 0xFF\n",
    'DEC': "r = (v - 1) & 0xFF\n",
}
_NZA = "self.fz = self.a == 0\nself.fn = self.a >> 7\n"
# unofficial combined ops: (shift op, follow-up using `r`)
COMBO = {
    'SLO': ('ASL', "self.a |= r\n" + _NZA),
    'RLA': ('ROL', "self.a &= r\n" + _NZA),
    'SRE': ('LSR', "self.a ^= r\n" + _NZA),
    'RRA': ('ROR', "v = r\n" + ADC),
    'DCP': ('DEC', "v = r\n" + _cmp('a')),
    'ISC': ('INC', "v = r ^ 0xFF\n" + ADC),
}
COMBO_BASE = {'SLO': 0x00, 'RLA': 0x20, 'SRE': 0x40, 'RRA': 0x60, 'DCP': 0xC0, 'ISC': 0xE0}
COMBO_MODES = (('zp', 0x07), ('zpx', 0x17), ('abs', 0x0F), ('abx', 0x1F),
               ('aby', 0x1B), ('izx', 0x03), ('izy', 0x13))

BRANCH = {0x10: "not self.fn", 0x30: "self.fn", 0x50: "not self.fv", 0x70: "self.fv",
          0x90: "not self.fc", 0xB0: "self.fc", 0xD0: "not self.fz", 0xF0: "self.fz"}

PUSH = "ram[0x100 + self.sp] = {v}\nself.sp = (self.sp - 1) & 0xFF\n"
PULL = "self.sp = (self.sp + 1) & 0xFF\n{d} = ram[0x100 + self.sp]\n"

IMPLIED = {
    0xAA: "r = self.x = self.a\n" + NZ, 0xA8: "r = self.y = self.a\n" + NZ,
    0x8A: "r = self.a = self.x\n" + NZ, 0x98: "r = self.a = self.y\n" + NZ,
    0xBA: "r = self.x = self.sp\n" + NZ, 0x9A: "self.sp = self.x\n",
    0xE8: "r = self.x = (self.x + 1) & 0xFF\n" + NZ, 0xC8: "r = self.y = (self.y + 1) & 0xFF\n" + NZ,
    0xCA: "r = self.x = (self.x - 1) & 0xFF\n" + NZ, 0x88: "r = self.y = (self.y - 1) & 0xFF\n" + NZ,
    0x18: "self.fc = 0\n", 0x38: "self.fc = 1\n", 0x58: "self.fi = 0\n", 0x78: "self.fi = 1\n",
    0xB8: "self.fv = 0\n", 0xD8: "self.fd = 0\n", 0xF8: "self.fd = 1\n",
    0xEA: "pass\n",
    0x48: PUSH.format(v="self.a"),
    0x08: PUSH.format(v="self.get_status(0x30)"),
    0x68: PULL.format(d="r") + "self.a = r\n" + NZ,
    0x28: PULL.format(d="s") + "self.set_status(s)\n",
    0x60: (PULL.format(d="lo") + PULL.format(d="hi") +
           "self.pc = (((hi << 8) | lo) + 1) & 0xFFFF\n"),
    0x40: (PULL.format(d="s") + "self.set_status(s)\n" + PULL.format(d="lo") + PULL.format(d="hi") +
           "self.pc = (hi << 8) | lo\n"),
    0x20: ("pc = self.pc\nt = (pc + 1) & 0xFFFF\nnew = rd(pc) | (rd(t) << 8)\n" +
           PUSH.format(v="t >> 8") + PUSH.format(v="t & 0xFF") + "self.pc = new\n"),
    0x4C: "pc = self.pc\nself.pc = rd(pc) | (rd((pc + 1) & 0xFFFF) << 8)\n",
    0x6C: ("pc = self.pc\np = rd(pc) | (rd((pc + 1) & 0xFFFF) << 8)\n"
           "self.pc = rd(p) | (rd((p & 0xFF00) | ((p + 1) & 0xFF)) << 8)\n"),
    0x00: ("pc = (self.pc + 1) & 0xFFFF\n" + PUSH.format(v="pc >> 8") + PUSH.format(v="pc & 0xFF") +
           PUSH.format(v="self.get_status(0x30)") + "self.fi = 1\n"
           "self.pc = rd(0xFFFE) | (rd(0xFFFF) << 8)\n"),
}
IMPLIED_CYC = {0xAA: 2, 0xA8: 2, 0x8A: 2, 0x98: 2, 0xBA: 2, 0x9A: 2, 0xE8: 2, 0xC8: 2, 0xCA: 2, 0x88: 2,
               0x18: 2, 0x38: 2, 0x58: 2, 0x78: 2, 0xB8: 2, 0xD8: 2, 0xF8: 2, 0xEA: 2,
               0x48: 3, 0x08: 3, 0x68: 4, 0x28: 4, 0x60: 6, 0x40: 6, 0x20: 6, 0x4C: 3, 0x6C: 5, 0x00: 7}


class CPU:
    C, Z, I, D, B, U, V, N = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80

    def __init__(self, bus, enable_logging=False):
        self.bus = bus
        self.a = self.x = self.y = 0
        self.sp = 0xFD
        self.pc = 0
        self.fc = self.fz = self.fd = self.fv = self.fn = 0
        self.fi = 1
        self.cycles = 0
        self.rd = bus.cpu_read
        self.wr = bus.cpu_write
        self.ops = [None] * 256
        self.cyc = [2] * 256
        self._build()

    # ---- status register ----------------------------------------------
    def get_status(self, b=0x20):
        return (self.fc | (self.fz << 1) | (self.fi << 2) | (self.fd << 3) | 0x20 | b |
                (self.fv << 6) | (self.fn << 7)) & 0xFF

    def set_status(self, s):
        self.fc = s & 1
        self.fz = (s >> 1) & 1
        self.fi = (s >> 2) & 1
        self.fd = (s >> 3) & 1
        self.fv = (s >> 6) & 1
        self.fn = (s >> 7) & 1

    @property
    def status(self):
        return self.get_status()

    @status.setter
    def status(self, v):
        self.set_status(v)

    # ---- helpers ---------------------------------------------------------
    def read16(self, addr):
        return self.rd(addr) | (self.rd((addr + 1) & 0xFFFF) << 8)

    def push(self, val):
        self.wr(0x100 + self.sp, val)
        self.sp = (self.sp - 1) & 0xFF

    def reset(self):
        self.pc = self.read16(0xFFFC)
        self.sp = 0xFD
        self.set_status(0x24)
        self.cycles = 7

    def _interrupt(self, vector):
        pc = self.pc
        self.push(pc >> 8)
        self.push(pc & 0xFF)
        self.push(self.get_status(0x20))      # B flag clear
        self.fi = 1
        self.pc = self.read16(vector)
        self.cycles += 7

    def nmi(self):
        self._interrupt(0xFFFA)

    def irq(self):
        self._interrupt(0xFFFE)

    def step(self):
        op = self.rd(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        self.cycles += self.cyc[op]
        self.ops[op]()

    # ---- opcode table generation -------------------------------------------
    def _build(self):
        bus = self.bus
        ns = {'self': self, 'rd': self.rd, 'wr': self.wr,
              'ram': bus.ram, 'prg': bus.prg, 'poff': bus.prg_off}
        ops, cyc = self.ops, self.cyc

        def emit(op, code, cycles):
            if "#ZP" in code:       # effective address is always in zero page -> plain RAM access
                code = code.replace("rd(a)", "ram[a]")
                code = re.sub(r"wr\(a, (.*)\)\n", r"ram[a] = \1\n", code)
            src = "def h():\n" + "".join("    " + ln + "\n" for ln in code.split("\n") if ln.strip() != "" or ln.startswith(" "))
            exec(src, ns)
            ops[op] = ns['h']
            cyc[op] = cycles

        emit_default = "pass"
        for op in range(256):
            emit(op, emit_default, 2)

        def addr(mode, write=False):
            return MODE[WMODE.get(mode, mode)] if write and mode in WMODE else MODE[mode]

        # --- group 01: ORA AND EOR ADC STA LDA CMP SBC ---
        names = ['ORA', 'AND', 'EOR', 'ADC', 'STA', 'LDA', 'CMP', 'SBC']
        modes = ['izx', 'zp', 'imm', 'abs', 'izy', 'zpx', 'aby', 'abx']
        for aaa, name in enumerate(names):
            for bbb, mode in enumerate(modes):
                op = (aaa << 5) | (bbb << 2) | 1
                if name == 'STA':
                    if mode == 'imm':
                        continue
                    emit(op, addr(mode, True) + "wr(a, self.a)\n", CYC_WRITE[mode])
                else:
                    emit(op, addr(mode) + "v = rd(a)\n" + READ[name], CYC_READ[mode])

        # --- group 10: shifts, INC/DEC, LDX, STX ---
        for aaa, name in enumerate(['ASL', 'ROL', 'LSR', 'ROR', 'STX', 'LDX', 'DEC', 'INC']):
            for bbb, mode in ((0, 'imm'), (1, 'zp'), (2, 'acc'), (3, 'abs'), (5, 'zpx'), (7, 'abx')):
                op = (aaa << 5) | (bbb << 2) | 2
                if name in ('STX', 'LDX') and mode in ('zpx', 'abx'):
                    mode = 'zpy' if mode == 'zpx' else 'aby'
                if name == 'STX':
                    if mode in ('imm', 'acc', 'aby'):
                        continue
                    emit(op, addr(mode, True) + "wr(a, self.x)\n", CYC_WRITE[mode])
                elif name == 'LDX':
                    if mode == 'acc':
                        continue
                    emit(op, addr(mode) + "v = rd(a)\n" + READ['LDX'], CYC_READ[mode])
                else:
                    if mode == 'imm':
                        continue
                    if mode == 'acc':
                        if name in ('INC', 'DEC'):
                            continue
                        emit(op, "v = self.a\n" + SHIFT[name] + "self.a = r\n" + NZ, 2)
                    else:
                        emit(op, addr(mode, True) + "v = rd(a)\n" + SHIFT[name] + "wr(a, r)\n" + NZ, CYC_RMW[mode])

        # --- group 00: BIT STY LDY CPY CPX ---
        allowed = {'BIT': ('zp', 'abs'), 'STY': ('zp', 'abs', 'zpx'),
                   'LDY': ('imm', 'zp', 'abs', 'zpx', 'abx'), 'CPY': ('imm', 'zp', 'abs'),
                   'CPX': ('imm', 'zp', 'abs')}
        for aaa, name in ((1, 'BIT'), (4, 'STY'), (5, 'LDY'), (6, 'CPY'), (7, 'CPX')):
            for bbb, mode in ((0, 'imm'), (1, 'zp'), (3, 'abs'), (5, 'zpx'), (7, 'abx')):
                if mode not in allowed[name]:
                    continue
                op = (aaa << 5) | (bbb << 2)
                if name == 'STY':
                    emit(op, addr(mode, True) + "wr(a, self.y)\n", CYC_WRITE[mode])
                else:
                    emit(op, addr(mode) + "v = rd(a)\n" + READ[name], CYC_READ[mode])

        # --- implied / stack / jumps / BRK ---
        for op, code in IMPLIED.items():
            emit(op, code, IMPLIED_CYC[op])

        # --- branches ---
        for op, cond in BRANCH.items():
            emit(op, ("pc = self.pc\noff = prg[poff[(pc >> 13) & 3] + (pc & 0x1FFF)] if pc >= 0x8000 else rd(pc)\npc = (pc + 1) & 0xFFFF\n"
                      f"if {cond}:\n"
                      "    if off & 0x80:\n        off -= 256\n"
                      "    t = (pc + off) & 0xFFFF\n"
                      "    self.cycles += 1\n"
                      "    if (pc ^ t) & 0xFF00:\n        self.cycles += 1\n"
                      "    pc = t\n"
                      "self.pc = pc\n"), 2)

        # --- unofficial opcodes ---
        for op in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA):
            emit(op, "pass", 2)
        for op in (0x80, 0x82, 0x89, 0xC2, 0xE2):
            emit(op, MODE['imm'], 2)
        for op in (0x04, 0x44, 0x64):
            emit(op, MODE['zp'] + "rd(a)\n", 3)
        for op in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4):
            emit(op, MODE['zpx'] + "rd(a)\n", 4)
        emit(0x0C, MODE['abs'] + "rd(a)\n", 4)
        for op in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC):
            emit(op, MODE['abx'] + "rd(a)\n", 4)
        emit(0xEB, MODE['imm'] + "v = rd(a)\n" + READ['SBC'], 2)
        for base, mode in ((0xA7, 'zp'), (0xB7, 'zpy'), (0xAF, 'abs'), (0xBF, 'aby'), (0xA3, 'izx'), (0xB3, 'izy')):
            emit(base, addr(mode) + "v = rd(a)\n" + READ['LAX'], CYC_READ[mode])
        for base, mode in ((0x87, 'zp'), (0x97, 'zpy'), (0x8F, 'abs'), (0x83, 'izx')):
            emit(base, addr(mode, True) + "wr(a, self.a & self.x)\n", CYC_WRITE[mode])
        for name, (shift, after) in COMBO.items():
            for mode, low in COMBO_MODES:
                emit(COMBO_BASE[name] | low, addr(mode, True) + "v = rd(a)\n" + SHIFT[shift] +
                     "wr(a, r)\n" + after, CYC_RMW[mode])
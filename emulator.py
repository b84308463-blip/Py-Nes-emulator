from cpu import CPU
from ppu import PPU
from bus import Bus
from cartridge import Cartridge
from mappers import create_mapper
from regions import resolve_region


class Emulator:
    def __init__(self, rom_path, enable_logging=False, region_choice="AUTO", audio=True):
        self.cart = Cartridge(rom_path)
        self.region = resolve_region(region_choice, rom_path, self.cart)

        self.bus = Bus(self.cart, self.region, audio=audio)
        self.mapper = create_mapper(self.cart)
        self.bus.attach_mapper(self.mapper)

        self.cpu = CPU(self.bus, enable_logging=enable_logging)
        self.ppu = PPU(self.bus, self.region)
        self.ppu.attach()
        self.bus.cpu = self.cpu
        self.bus.ppu = self.ppu

        self.ratio = self.region.ppu_num / self.region.ppu_den
        self._acc = 0                       # fractional PPU dots (PAL = 3.2 dots / CPU cycle)
        self.ppu.reset()
        self.cpu.reset()

    def _dots(self, cycles):
        """CPU cycles -> whole PPU dots, carrying the remainder (no drift on PAL)."""
        r = self.region
        self._acc += cycles * r.ppu_num
        d, self._acc = divmod(self._acc, r.ppu_den)
        return d

    def _nmi(self):
        cpu = self.cpu
        cpu.push((cpu.pc >> 8) & 0xFF)
        cpu.push(cpu.pc & 0xFF)
        cpu.push((cpu.status | 0x20) & ~0x10)
        cpu.status |= cpu.I
        cpu.pc = cpu.read16(0xFFFA)
        cpu.cycles += 7
        self.ppu.run(self._dots(7))

    def step_frame(self):
        ppu, cpu, bus = self.ppu, self.cpu, self.bus
        mapper = self.mapper
        apu = bus.apu
        rd, ops, cyc = cpu.rd, cpu.ops, cpu.cyc
        prg, poff = bus.prg, bus.prg_off
        num, den = self.region.ppu_num, self.region.ppu_den
        acc = self._acc
        start_frame = ppu.frame
        guard = 0

        # Hot loop: CPU step, PPU and APU catch-up are inlined to avoid
        # several Python function calls per instruction.
        while ppu.frame == start_frame:
            guard += 1
            if guard > 300000:
                break

            if ppu.nmi_pending and (ppu.ctrl & 0x80):
                ppu.nmi_pending = False
                self._acc = acc
                self._nmi()
                acc = self._acc

            pc = cpu.pc
            op = prg[poff[(pc >> 13) & 3] + (pc & 0x1FFF)] if pc >= 0x8000 else rd(pc)
            cpu.pc = (pc + 1) & 0xFFFF
            before = cpu.cycles
            cpu.cycles = before + cyc[op]
            ops[op]()
            n = cpu.cycles - before          # includes DMA / page-cross penalties

            if den == 1:
                dots = n * num
            else:
                acc += n * num
                dots = acc // den
                acc -= dots * den
            c = ppu.cycle + dots
            if c < ppu.next_c:
                ppu.cycle = c
            else:
                ppu._run_slow(dots)

            apu.cyc += n
            if apu.cyc >= apu.next_fc:
                apu._frame_events()

            if mapper.has_cpu_clock:
                mapper.cpu_clock(n)
            if mapper.irq_pending and not cpu.fi:      # honour the CPU's interrupt-disable flag;
                mapper.irq_pending = False             # a masked IRQ stays pending until CLI/RTI
                cpu.irq()

        self._acc = acc
        apu.flush()                          # produce this frame's audio samples

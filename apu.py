"""NES APU: 2 pulse, triangle, noise and DMC channels.

Audio is synthesised with numpy in chunks. A chunk ends whenever the channel
state can change (register write, frame-counter tick, end of frame), so within
a chunk every channel is a pure function of time and can be computed as a
vector instead of looping per sample.
"""
import math
import numpy as np

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:                      # ImportError, or OSError if PortAudio is missing
    sd = None
    HAS_AUDIO = False

LEN_TABLE = [10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
             12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30]
DUTY = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0, 0, 0],
                 [1, 0, 0, 1, 1, 1, 1, 1]], dtype=np.int64)
TRI_SEQ = np.array(list(range(15, -1, -1)) + list(range(0, 16)), dtype=np.int64)
NOISE_NTSC = [4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068]
NOISE_PAL = [4, 8, 14, 30, 60, 88, 118, 148, 188, 236, 354, 472, 708, 944, 1890, 3778]
DMC_NTSC = [428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54]
DMC_PAL = [398, 354, 316, 298, 276, 236, 210, 198, 176, 148, 132, 118, 98, 78, 66, 50]

PULSE_TAB = np.array([0.0] + [95.52 / (8128.0 / n + 100.0) for n in range(1, 31)])
TND_TAB = np.array([0.0] + [163.67 / (24329.0 / n + 100.0) for n in range(1, 203)])


def _lfsr(short):
    reg, out, seen = 1, [], {}
    while reg not in seen:
        seen[reg] = len(out)
        out.append(0 if (reg & 1) else 1)        # output is enabled while bit 0 is clear
        fb = (reg & 1) ^ ((reg >> (6 if short else 1)) & 1)
        reg = (reg >> 1) | (fb << 14)
    return np.array(out[seen[reg]:], dtype=np.int64)


NOISE_SEQ = [_lfsr(False), _lfsr(True)]


def _full(x, n):
    return x if isinstance(x, np.ndarray) else np.full(n, x, dtype=np.int64)


# ------------------------------------------------------------------ channels
class Envelope:
    def __init__(self):
        self.start = False
        self.divider = 0
        self.decay = 0
        self.loop = False
        self.const = False
        self.period = 0

    def write(self, v):
        self.loop = bool(v & 0x20)
        self.const = bool(v & 0x10)
        self.period = v & 0x0F

    def clock(self):
        if self.start:
            self.start = False
            self.decay = 15
            self.divider = self.period
        elif self.divider == 0:
            self.divider = self.period
            if self.decay > 0:
                self.decay -= 1
            elif self.loop:
                self.decay = 15
        else:
            self.divider -= 1

    def volume(self):
        return self.period if self.const else self.decay


class Pulse:
    def __init__(self, ones_complement):
        self.ones = ones_complement
        self.env = Envelope()
        self.duty = 0
        self.timer = 0
        self.length = 0
        self.enabled = False
        self.phase = 0.0
        self.seq = 0
        self.sw_en = False
        self.sw_period = 0
        self.sw_neg = False
        self.sw_shift = 0
        self.sw_div = 0
        self.sw_reload = False

    def write(self, reg, v):
        if reg == 0:
            self.duty = v >> 6
            self.env.write(v)
        elif reg == 1:
            self.sw_en = bool(v & 0x80)
            self.sw_period = (v >> 4) & 7
            self.sw_neg = bool(v & 8)
            self.sw_shift = v & 7
            self.sw_reload = True
        elif reg == 2:
            self.timer = (self.timer & 0x700) | v
        else:
            self.timer = (self.timer & 0xFF) | ((v & 7) << 8)
            if self.enabled:
                self.length = LEN_TABLE[v >> 3]
            self.seq = 0
            self.env.start = True

    def target(self):
        change = self.timer >> self.sw_shift
        if self.sw_neg:
            change = -change - (1 if self.ones else 0)
        return self.timer + change

    def clock_quarter(self):
        self.env.clock()

    def clock_half(self):
        if self.length and not self.env.loop:
            self.length -= 1
        tgt = self.target()
        muted = self.timer < 8 or tgt > 0x7FF
        if self.sw_div == 0 and self.sw_en and self.sw_shift and not muted:
            self.timer = tgt
        if self.sw_div == 0 or self.sw_reload:
            self.sw_div = self.sw_period
            self.sw_reload = False
        else:
            self.sw_div -= 1

    def render(self, t):
        if self.length == 0 or self.timer < 8 or self.target() > 0x7FF:
            return None
        vol = self.env.volume()
        if vol == 0:
            return None
        step_len = 2 * (self.timer + 1)
        idx = ((self.phase + t) // step_len).astype(np.int64)
        return DUTY[self.duty][(idx + self.seq) & 7] * vol

    def advance(self, dc):
        step_len = 2 * (self.timer + 1)
        total = self.phase + dc
        n = int(total // step_len)
        self.phase = total - n * step_len
        self.seq = (self.seq + n) & 7


class Triangle:
    def __init__(self):
        self.control = False
        self.reload_val = 0
        self.counter = 0
        self.reload_flag = False
        self.timer = 0
        self.length = 0
        self.enabled = False
        self.phase = 0.0
        self.seq = 0
        self.held = 0

    def write(self, reg, v):
        if reg == 0:
            self.control = bool(v & 0x80)
            self.reload_val = v & 0x7F
        elif reg == 2:
            self.timer = (self.timer & 0x700) | v
        elif reg == 3:
            self.timer = (self.timer & 0xFF) | ((v & 7) << 8)
            if self.enabled:
                self.length = LEN_TABLE[v >> 3]
            self.reload_flag = True

    def clock_quarter(self):
        if self.reload_flag:
            self.counter = self.reload_val
        elif self.counter:
            self.counter -= 1
        if not self.control:
            self.reload_flag = False

    def clock_half(self):
        if self.length and not self.control:
            self.length -= 1

    def active(self):
        return self.length > 0 and self.counter > 0 and self.timer >= 2

    def render(self, t):
        if not self.active():
            return self.held
        step_len = self.timer + 1
        idx = ((self.phase + t) // step_len).astype(np.int64)
        return TRI_SEQ[(idx + self.seq) & 31]

    def advance(self, dc):
        if not self.active():
            return
        step_len = self.timer + 1
        total = self.phase + dc
        n = int(total // step_len)
        self.phase = total - n * step_len
        self.seq = (self.seq + n) & 31
        self.held = int(TRI_SEQ[self.seq])


class Noise:
    def __init__(self):
        self.env = Envelope()
        self.mode = 0
        self.period_idx = 0
        self.length = 0
        self.enabled = False
        self.phase = 0.0
        self.pos = 0

    def write(self, reg, v):
        if reg == 0:
            self.env.write(v)
        elif reg == 2:
            self.mode = v >> 7
            self.period_idx = v & 15
        elif reg == 3:
            if self.enabled:
                self.length = LEN_TABLE[v >> 3]
            self.env.start = True

    def clock_quarter(self):
        self.env.clock()

    def clock_half(self):
        if self.length and not self.env.loop:
            self.length -= 1

    def render(self, t, periods):
        if self.length == 0:
            return None
        vol = self.env.volume()
        if vol == 0:
            return None
        arr = NOISE_SEQ[self.mode]
        idx = ((self.phase + t) // periods[self.period_idx]).astype(np.int64)
        return arr[(idx + self.pos) % len(arr)] * vol

    def advance(self, dc, periods):
        per = periods[self.period_idx]
        total = self.phase + dc
        n = int(total // per)
        self.phase = total - n * per
        self.pos = (self.pos + n) % len(NOISE_SEQ[self.mode])


class DMC:
    def __init__(self, apu):
        self.apu = apu
        self.irq_en = False
        self.loop = False
        self.rate_idx = 0
        self.level = 0
        self.addr = 0xC000
        self.length = 1
        self.cur = 0xC000
        self.bytes_left = 0
        self.buffer = None
        self.shift = 0
        self.bits = 8
        self.silence = True
        self.acc = 0.0
        self.irq = False

    def write(self, reg, v):
        if reg == 0:
            self.irq_en = bool(v & 0x80)
            self.loop = bool(v & 0x40)
            self.rate_idx = v & 15
            if not self.irq_en:
                self.irq = False
        elif reg == 1:
            self.level = v & 0x7F
        elif reg == 2:
            self.addr = 0xC000 | (v << 6)
        else:
            self.length = (v << 4) | 1

    def restart(self):
        self.cur = self.addr
        self.bytes_left = self.length

    def set_enabled(self, on):
        if not on:
            self.bytes_left = 0
        elif self.bytes_left == 0:
            self.restart()
            self._fill()

    def _fill(self):
        if self.buffer is None and self.bytes_left > 0:
            self.buffer = self.apu.bus.cpu_read(self.cur)
            self.cur = ((self.cur + 1) & 0xFFFF) | 0x8000
            self.bytes_left -= 1
            if self.bytes_left == 0:
                if self.loop:
                    self.restart()
                elif self.irq_en:
                    self.irq = True

    def _bit(self):
        if not self.silence:
            if self.shift & 1:
                if self.level <= 125:
                    self.level += 2
            elif self.level >= 2:
                self.level -= 2
            self.shift >>= 1
        self.bits -= 1
        if self.bits == 0:
            self.bits = 8
            if self.buffer is None:
                self.silence = True
            else:
                self.silence = False
                self.shift = self.buffer
                self.buffer = None
                self._fill()

    def run(self, dc, t, rates):
        """Advance by dc CPU cycles. Returns the output level (int or array over t)."""
        start = self.level
        if self.bytes_left == 0 and self.buffer is None and self.silence:
            return start
        self._fill()
        rate = rates[self.rate_idx]
        times, levels = [], []
        pos = rate - self.acc
        while pos <= dc:
            self._bit()
            times.append(pos)
            levels.append(self.level)
            pos += rate
        self.acc = rate - (pos - dc)
        if t is None or not times:
            return start if not times else self.level
        lv = np.concatenate(([start], levels)).astype(np.int64)
        return lv[np.searchsorted(times, t, side='right')]


# ------------------------------------------------------------------ audio output
class AudioOut:
    """Lock-free ring buffer feeding a sounddevice stream."""

    def __init__(self, rate):
        self.rate = rate
        self.size = rate * 2
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.w = 0
        self.r = 0
        self.max_backlog = int(rate * 0.30)
        self.target = int(rate * 0.10)
        self.stream = sd.OutputStream(samplerate=rate, channels=1, dtype='float32',
                                      blocksize=1024, latency=0.1, callback=self._cb)
        self.stream.start()

    def write(self, arr):
        n = len(arr)
        if self.w - self.r > self.max_backlog:           # emulator running ahead: drop old audio
            self.r = self.w - self.target
        i = self.w % self.size
        end = i + n
        if end <= self.size:
            self.buf[i:end] = arr
        else:
            k = self.size - i
            self.buf[i:] = arr[:k]
            self.buf[:n - k] = arr[k:]
        self.w += n

    def _cb(self, outdata, frames, time, status):
        avail = self.w - self.r
        n = min(frames, avail)
        if n > 0:
            i = self.r % self.size
            end = i + n
            if end <= self.size:
                outdata[:n, 0] = self.buf[i:end]
            else:
                k = self.size - i
                outdata[:k, 0] = self.buf[i:]
                outdata[k:n, 0] = self.buf[:end - self.size]
            self.r += n
        if n < frames:
            outdata[max(n, 0):, 0] = 0.0

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# ------------------------------------------------------------------ APU
class APU:
    def __init__(self, bus, region, audio=True, sample_rate=44100):
        self.bus = bus
        self.region = region
        self.sample_rate = sample_rate
        self.step = region.cpu_hz / sample_rate        # CPU cycles per output sample
        self.noise_periods = NOISE_PAL if region.pal_tables else NOISE_NTSC
        self.dmc_rates = DMC_PAL if region.pal_tables else DMC_NTSC

        self.pulse = [Pulse(True), Pulse(False)]
        self.tri = Triangle()
        self.noise = Noise()
        self.dmc = DMC(self)

        self.cyc = 0
        self.last = 0
        self.next_sample = 0.0
        self.fc_mode = 4
        self.fc_inhibit = False
        self.fc_step = 0
        self.fc_start = 0
        self.frame_irq = False
        self.irq = False
        self.next_fc = region.frame_steps4[0]

        self.volume = 1.0
        self.muted = False
        self.dc = 0.0
        self.prev = 0.0
        self.out = None
        self.audio_error = None
        if audio:
            if not HAS_AUDIO:
                self.audio_error = "sounddevice / PortAudio not available (pip install sounddevice)"
            else:
                try:
                    self.out = AudioOut(sample_rate)
                except Exception as e:
                    self.audio_error = f"Could not open audio device: {e}"

    # ---- housekeeping -------------------------------------------------
    def close(self):
        if self.out:
            self.out.close()
            self.out = None

    def _upd_irq(self):
        self.irq = self.frame_irq or self.dmc.irq

    def tick(self, cycles):
        self.cyc += cycles
        if self.cyc >= self.next_fc:
            self._frame_events()

    def flush(self):
        """Call once per frame so queued samples are produced."""
        self._render(self.cyc)
        self._upd_irq()

    # ---- frame counter ------------------------------------------------------
    def _steps(self):
        r = self.region
        return r.frame_steps5 if self.fc_mode == 5 else r.frame_steps4

    def _quarter(self):
        self.pulse[0].clock_quarter()
        self.pulse[1].clock_quarter()
        self.tri.clock_quarter()
        self.noise.clock_quarter()

    def _half(self):
        self.pulse[0].clock_half()
        self.pulse[1].clock_half()
        self.tri.clock_half()
        self.noise.clock_half()

    def _frame_events(self):
        r = self.region
        while self.cyc >= self.next_fc:
            self._render(self.next_fc)
            s = self.fc_step
            if self.fc_mode == 4:
                self._quarter()
                if s in (1, 3):
                    self._half()
                if s == 3 and not self.fc_inhibit:
                    self.frame_irq = True
                count, period = 4, r.frame_period4
            else:
                if s != 3:
                    self._quarter()
                if s in (1, 4):
                    self._half()
                count, period = 5, r.frame_period5
            self.fc_step += 1
            if self.fc_step >= count:
                self.fc_step = 0
                self.fc_start += period
            self.next_fc = self.fc_start + self._steps()[self.fc_step]
        self._upd_irq()

    # ---- registers --------------------------------------------------------------
    def write_register(self, addr, val):
        if addr > 0x4017:
            return
        self._render(self.cyc)
        if addr < 0x4004:
            self.pulse[0].write(addr & 3, val)
        elif addr < 0x4008:
            self.pulse[1].write(addr & 3, val)
        elif addr < 0x400C:
            self.tri.write(addr & 3, val)
        elif addr < 0x4010:
            self.noise.write(addr & 3, val)
        elif addr < 0x4014:
            self.dmc.write(addr & 3, val)
        elif addr == 0x4015:
            for i in (0, 1):
                ch = self.pulse[i]
                ch.enabled = bool(val & (1 << i))
                if not ch.enabled:
                    ch.length = 0
            self.tri.enabled = bool(val & 4)
            if not self.tri.enabled:
                self.tri.length = 0
            self.noise.enabled = bool(val & 8)
            if not self.noise.enabled:
                self.noise.length = 0
            self.dmc.irq = False
            self.dmc.set_enabled(bool(val & 0x10))
            self._upd_irq()
        elif addr == 0x4017:
            self.fc_mode = 5 if val & 0x80 else 4
            self.fc_inhibit = bool(val & 0x40)
            if self.fc_inhibit:
                self.frame_irq = False
            self.fc_step = 0
            self.fc_start = self.cyc
            self.next_fc = self.fc_start + self._steps()[0]
            if self.fc_mode == 5:
                self._quarter()
                self._half()
            self._upd_irq()

    def read_status(self):
        self._render(self.cyc)
        v = 0
        if self.pulse[0].length:
            v |= 1
        if self.pulse[1].length:
            v |= 2
        if self.tri.length:
            v |= 4
        if self.noise.length:
            v |= 8
        if self.dmc.bytes_left:
            v |= 0x10
        if self.frame_irq:
            v |= 0x40
        if self.dmc.irq:
            v |= 0x80
        self.frame_irq = False
        self._upd_irq()
        return v

    # ---- synthesis ------------------------------------------------------------------
    def _render(self, end):
        ns = self.next_sample
        if ns >= end:
            return                       # no output sample falls in this interval yet
        dc = end - self.last
        step = self.step
        n = int(math.ceil((end - ns) / step))
        audio = self.out is not None and not self.muted

        t = None
        p = tri = nz = None
        if audio:
            t = (ns - self.last) + step * np.arange(n)
            p = 0
            for ch in self.pulse:
                a = ch.render(t)
                if a is not None:
                    p = p + a
            tri = self.tri.render(t)
            nz = self.noise.render(t, self.noise_periods)
        dm = self.dmc.run(dc, t, self.dmc_rates)
        self.pulse[0].advance(dc)
        self.pulse[1].advance(dc)
        self.tri.advance(dc)
        self.noise.advance(dc, self.noise_periods)

        if audio:
            pi = _full(p, n)
            ti = 3 * _full(tri, n) + 2 * _full(nz if nz is not None else 0, n) + _full(dm, n)
            s = PULSE_TAB[pi] + TND_TAB[ti]
            alpha = min(1.0, n / (self.sample_rate * 0.1))
            s -= self.dc
            self.dc += float(s.mean()) * alpha
            sm = s.copy()
            sm[1:] = (s[1:] + s[:-1]) * 0.5
            sm[0] = (s[0] + self.prev) * 0.5
            self.prev = float(s[-1])
            sm *= 1.6 * self.volume
            np.clip(sm, -1.0, 1.0, out=sm)
            self.out.write(sm.astype(np.float32))

        self.next_sample = ns + n * step
        self.last = end
        self._upd_irq()

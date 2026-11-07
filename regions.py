"""Region / console timing presets.

NTSC  - NES / Famicom (Japan, North America)          262 lines, ~60.10 Hz
PAL   - NES (Europe, Australia)                        312 lines, ~50.01 Hz
DENDY - Russian Famicom clone (PAL timing, long vblank) 312 lines, ~50.01 Hz
"""
import os
import re


class Region:
    def __init__(self, key, label, cpu_hz, lines, vblank_line, ppu_num, ppu_den,
                 frame_steps4, frame_period4, frame_steps5, frame_period5, pal_tables):
        self.key = key
        self.label = label
        self.cpu_hz = cpu_hz
        self.lines = lines                  # scanlines per frame
        self.vblank_line = vblank_line      # scanline where vblank starts
        self.pre_line = lines - 1           # pre-render scanline
        self.ppu_num = ppu_num              # PPU dots per CPU cycle = num/den
        self.ppu_den = ppu_den
        self.frame_steps4 = frame_steps4
        self.frame_period4 = frame_period4
        self.frame_steps5 = frame_steps5
        self.frame_period5 = frame_period5
        self.pal_tables = pal_tables        # use PAL noise/DMC period tables
        dots = lines * 341
        self.fps = cpu_hz * ppu_num / ppu_den / dots

    def __repr__(self):
        return f"<Region {self.key} {self.fps:.2f}fps>"


NTSC = Region("NTSC", "NTSC  -  NES / Famicom (Japan, North America)", 1789773.0, 262, 241, 3, 1,
              (7457, 14913, 22371, 29829), 29830, (7457, 14913, 22371, 29829, 37281), 37282, False)
PAL = Region("PAL", "PAL  -  NES (Europe, Australia)", 1662607.0, 312, 241, 16, 5,
             (8313, 16627, 24939, 33253), 33255, (8313, 16627, 24939, 33253, 41565), 41566, True)
DENDY = Region("DENDY", "Dendy  -  Russian Famicom clone", 1773448.0, 312, 291, 3, 1,
               (7457, 14913, 22371, 29829), 29830, (7457, 14913, 22371, 29829, 37281), 37282, False)

REGIONS = {"NTSC": NTSC, "PAL": PAL, "DENDY": DENDY}

# What the user sees in the Options dialog
CONSOLE_CHOICES = [
    ("AUTO", "Auto-detect (from ROM header / file name)"),
    ("NTSC", NTSC.label),
    ("PAL", PAL.label),
    ("DENDY", DENDY.label),
]

_PAL_TAGS = re.compile(r"\((e|europe|pal|australia|germany|france|spain|italy|sweden|netherlands|"
                       r"uk|united kingdom|scandinavia|eur)\)|\[pal\]", re.I)
_DENDY_TAGS = re.compile(r"dendy|\(russia\)", re.I)


def detect_region(path, cart):
    """Guess the region of a ROM. Returns a Region."""
    hint = getattr(cart, "region_hint", None)
    if hint in REGIONS:
        return REGIONS[hint]
    name = os.path.basename(path or "")
    if _DENDY_TAGS.search(name):
        return DENDY
    if _PAL_TAGS.search(name):
        return PAL
    return NTSC


def resolve_region(choice, path, cart):
    if choice in REGIONS:
        return REGIONS[choice]
    return detect_region(path, cart)

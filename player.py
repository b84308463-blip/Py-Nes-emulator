import sys
import os
import time
import gc
import pygame
import numpy as np
from emulator import Emulator


def build_pal32(surface, palette_rgb):
    """NES palette -> packed 32-bit pixels in the surface's own pixel format."""
    return np.array([surface.map_rgb((int(r), int(g), int(b))) for r, g, b in palette_rgb], dtype=np.uint32)


_fast_blit_ok = [True]

def draw_frame(emu, back, screen, pal32):
    """Palette-index frame -> screen. One uint32 gather (4x less data than RGB),
    then a single scale straight into the display surface (no temp surfaces)."""
    fb = emu.ppu.frame_buf
    if _fast_blit_ok[0]:
        try:
            pygame.surfarray.blit_array(back, pal32[fb.T])
            pygame.transform.scale(back, screen.get_size(), screen)
            return
        except Exception:
            _fast_blit_ok[0] = False          # unusual pixel format -> use the safe path
    rgb = emu.ppu.get_palette_array()[fb]
    pygame.surfarray.blit_array(back, rgb.swapaxes(0, 1))
    screen.blit(pygame.transform.scale(back, screen.get_size()), (0, 0))

def clean_title(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for ch in "_.":
        name = name.replace(ch, " ")
    for tag in ("(U)", "(E)", "(J)", "[!]", "(USA)", "(World)"):
        name = name.replace(tag, "")
    return " ".join(name.split()) or "UNKNOWN"

def main():
    if len(sys.argv) < 2:
        return

    rom_path = sys.argv[1]
    region = sys.argv[2] if len(sys.argv) > 2 else "AUTO"
    
    pygame.init()
    
    clock = pygame.time.Clock()
    WIDTH, HEIGHT = 256 * 3, 240 * 3
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    title_base = f"PY-NES | {clean_title(rom_path)} [{region}]"

    try:
        emu = Emulator(rom_path, enable_logging=False, region_choice=region, audio=True)
    except Exception as e:
        print(f"Error loading ROM: {e}")
        return

    key_map = {
        pygame.K_z: 0x01, pygame.K_x: 0x02, pygame.K_RSHIFT: 0x04, pygame.K_RETURN: 0x08,
        pygame.K_UP: 0x10, pygame.K_DOWN: 0x20, pygame.K_LEFT: 0x40, pygame.K_RIGHT: 0x80
    }

    running = True
    frames = 0
    fps_t0 = time.time()
    back_surface = pygame.Surface((256, 240), 0, screen)
    pal32 = build_pal32(back_surface, emu.ppu.get_palette_array())
    target_fps = int(round(emu.region.fps))
    emu_time = 0.0
    gc.collect()
    gc.disable()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_map:
                    emu.bus.controller[0] |= key_map[event.key]
            elif event.type == pygame.KEYUP:
                if event.key in key_map:
                    emu.bus.controller[0] &= ~key_map[event.key]

        t_emu = time.perf_counter()
        emu.step_frame()
        emu_time += time.perf_counter() - t_emu

        # draw EVERY frame (no frame skipping)
        draw_frame(emu, back_surface, screen, pal32)
        pygame.display.flip()

        frames += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps = frames / (now - fps_t0)
            emu_ms = emu_time / max(frames, 1) * 1000
            pygame.display.set_caption(f"{title_base} | {fps:.1f} FPS | emu {emu_ms:.1f} ms | ESC to quit")
            frames = 0
            emu_time = 0.0
            fps_t0 = now

        clock.tick_busy_loop(target_fps)

    if emu.bus.apu.out:
        emu.bus.apu.out.close()

    pygame.quit()

if __name__ == "__main__":
    main()
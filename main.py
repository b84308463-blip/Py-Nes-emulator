import sys
import os
import json
import time
import pickle
import gc
import pygame
import numpy as np
import tkinter as tk
from tkinter import filedialog
from emulator import Emulator

SETTINGS_FILE = "settings.json"
COVERS_DIR = "covers"
SNAPSHOTS_DIR = "snapshots"


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

class CheatManager:
    """Manages RAM patches and Game Genie / ROM overrides."""
    def __init__(self):
        self.ram_patches = {}  # { addr: value }
        self.rom_patches = {}  # { addr: value }
        self.enabled = False

    def add_ram_cheat(self, addr, val):
        self.ram_patches[addr] = val & 0xFF

    def apply_ram_cheats(self, ram_array):
        if not self.enabled:
            return
        for addr, val in self.ram_patches.items():
            ram_array[addr & 0x7FF] = val

def load_settings():
    default_settings = {"last_folder": "", "last_rom": "", "region": "AUTO", "audio": True, "cheats_enabled": False, "recent_roms": []}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return {**default_settings, **json.load(f)}
        except Exception:
            pass
    return default_settings

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

def add_to_recent(settings, rom_path):
    recents = settings.get("recent_roms", [])
    if rom_path in recents:
        recents.remove(rom_path)
    recents.insert(0, rom_path)
    settings["recent_roms"] = recents[:10]
    save_settings(settings)

def clean_title(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for ch in "_.":
        name = name.replace(ch, " ")
    for tag in ("(U)", "(E)", "(J)", "[!]", "(USA)", "(World)", "(Rev 1)"):
        name = name.replace(tag, "")
    return " ".join(name.split()) or "UNKNOWN"

def select_rom_file(initial_dir):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select NES ROM File",
        initialdir=initial_dir if initial_dir else os.getcwd(),
        filetypes=[("NES ROM files", "*.nes"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def select_rom_folder(initial_dir):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(
        title="Select ROMs Folder",
        initialdir=initial_dir if initial_dir else os.getcwd()
    )
    root.destroy()
    return folder_path

def get_rom_list(folder_path):
    roms = []
    if folder_path and os.path.exists(folder_path):
        for f in sorted(os.listdir(folder_path)):
            if f.lower().endswith(".nes"):
                roms.append(os.path.join(folder_path, f))
    return roms

def load_cover_art(rom_path, width=200, height=260):
    os.makedirs(COVERS_DIR, exist_ok=True)
    if rom_path:
        base_name = os.path.splitext(os.path.basename(rom_path))[0]
        for ext in [".png", ".jpg", ".jpeg"]:
            cover_path = os.path.join(COVERS_DIR, f"{base_name}{ext}")
            if os.path.exists(cover_path):
                try:
                    img = pygame.image.load(cover_path)
                    return pygame.transform.smoothscale(img, (width, height))
                except Exception:
                    pass

    surf = pygame.Surface((width, height))
    surf.fill((28, 28, 35))
    pygame.draw.rect(surf, (50, 50, 65), surf.get_rect(), 2)
    
    font = pygame.font.SysFont("Arial", 16, bold=True)
    lbl = font.render("NO COVER ART", True, (120, 120, 140))
    lbl_rect = lbl.get_rect(center=(width // 2, height // 2))
    surf.blit(lbl, lbl_rect)
    return surf

def take_screenshot(surface, rom_path):
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    game_title = clean_title(rom_path).replace(" ", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{game_title}_{timestamp}.png"
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    try:
        pygame.image.save(surface, filepath)
        print(f"Screenshot saved: {filepath}")
    except Exception as e:
        print(f"Error saving screenshot: {e}")

def get_state_path(rom_path):
    return f"{os.path.splitext(rom_path)[0]}.state"

def save_emulator_state(emu, rom_path):
    state_path = get_state_path(rom_path)
    try:
        state_data = {
            "cpu": emu.cpu.__dict__ if hasattr(emu, 'cpu') else {},
            "ppu": emu.ppu.__dict__ if hasattr(emu, 'ppu') else {},
            "bus": {"ram": emu.bus.ram.copy() if hasattr(emu.bus, 'ram') else None}
        }
        with open(state_path, "wb") as f:
            pickle.dump(state_data, f)
        print(f"State saved: {state_path}")
    except Exception as e:
        print(f"Error saving state: {e}")

def load_emulator_state(emu, rom_path):
    state_path = get_state_path(rom_path)
    if os.path.exists(state_path):
        try:
            with open(state_path, "rb") as f:
                state_data = pickle.load(f)
            if "cpu" in state_data and hasattr(emu, 'cpu'):
                emu.cpu.__dict__.update(state_data["cpu"])
            if "ppu" in state_data and hasattr(emu, 'ppu'):
                emu.ppu.__dict__.update(state_data["ppu"])
            if "bus" in state_data and "ram" in state_data["bus"] and hasattr(emu.bus, 'ram'):
                emu.bus.ram[:] = state_data["bus"]["ram"]
            print(f"State loaded: {state_path}")
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
    return False

def get_sram_path(rom_path):
    return f"{os.path.splitext(rom_path)[0]}.sav"

def load_sram(bus, rom_path):
    sram_path = get_sram_path(rom_path)
    if hasattr(bus, 'sram') and os.path.exists(sram_path):
        try:
            with open(sram_path, "rb") as f:
                data = f.read()
                bus.sram[:len(data)] = np.frombuffer(data, dtype=np.uint8)
            print(f"Loaded SRAM: {sram_path}")
        except Exception as e:
            print(f"Error loading SRAM: {e}")

def save_sram(bus, rom_path):
    sram_path = get_sram_path(rom_path)
    if hasattr(bus, 'sram'):
        try:
            with open(sram_path, "wb") as f:
                f.write(bus.sram.tobytes())
            print(f"Saved SRAM: {sram_path}")
        except Exception as e:
            print(f"Error saving SRAM: {e}")

def run_pause_menu(screen, game_w, game_h):
    font_title = pygame.font.SysFont("Arial", 26, bold=True)
    font_btn = pygame.font.SysFont("Arial", 16, bold=True)
    
    box_w, box_h = 280, 250
    box_x = (game_w - box_w) // 2
    box_y = (game_h - box_h) // 2
    box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
    
    buttons = [
        ("Resume Game", "RESUME"),
        ("Save State (F5)", "SAVE"),
        ("Load State (F8)", "LOAD"),
        ("Quit to Launcher", "QUIT")
    ]
    
    while True:
        mouse_pos = pygame.mouse.get_pos()
        overlay = pygame.Surface((game_w, game_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        
        pygame.draw.rect(screen, (26, 26, 34), box_rect, border_radius=10)
        pygame.draw.rect(screen, (231, 29, 54), box_rect, width=2, border_radius=10)
        
        title_surf = font_title.render("GAME PAUSED", True, (240, 240, 245))
        screen.blit(title_surf, (box_x + (box_w - title_surf.get_width()) // 2, box_y + 20))
        
        btn_rects = []
        for i, (label, action) in enumerate(buttons):
            brect = pygame.Rect(box_x + 20, box_y + 65 + i * 42, box_w - 40, 34)
            btn_rects.append((brect, action))
            
            is_hover = brect.collidepoint(mouse_pos)
            bg = (190, 25, 45) if (is_hover and action == "QUIT") else ((50, 50, 65) if is_hover else (35, 35, 45))
            
            pygame.draw.rect(screen, bg, brect, border_radius=6)
            txt_surf = font_btn.render(label, True, (240, 240, 245))
            screen.blit(txt_surf, (brect.x + (brect.width - txt_surf.get_width()) // 2, brect.y + 8))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "RESUME"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for brect, action in btn_rects:
                    if brect.collidepoint(event.pos):
                        return action

def run_menu(screen, settings):
    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont("Arial", 28, bold=True)
    font_item = pygame.font.SysFont("Arial", 16)
    font_small = pygame.font.SysFont("Arial", 14)
    font_btn = pygame.font.SysFont("Arial", 15, bold=True)

    BG_COLOR = (18, 18, 22)
    PANEL_BG = (26, 26, 34)
    HOVER_BG = (42, 42, 56)
    SELECTED_BG = (190, 25, 45)
    ACCENT_COLOR = (231, 29, 54)
    TEXT_COLOR = (240, 240, 245)
    MUTED_COLOR = (140, 140, 160)

    rom_folder = settings.get("last_folder", "")
    all_rom_list = get_rom_list(rom_folder)
    
    current_tab = "ALL"
    selected_idx = 0
    scroll_offset = 0

    while True:
        screen.fill(BG_COLOR)
        mouse_pos = pygame.mouse.get_pos()

        if current_tab == "ALL":
            active_list = all_rom_list
        else:
            active_list = [r for r in settings.get("recent_roms", []) if os.path.exists(r)]

        selected_idx = max(0, min(selected_idx, len(active_list) - 1)) if active_list else -1
        selected_rom = active_list[selected_idx] if 0 <= selected_idx < len(active_list) else ""

        # Top Header & Tabs
        title_surf = font_title.render("PY-NES GAME LIBRARY", True, ACCENT_COLOR)
        screen.blit(title_surf, (30, 20))

        tab_all_rect = pygame.Rect(350, 22, 90, 30)
        tab_rec_rect = pygame.Rect(450, 22, 100, 30)

        pygame.draw.rect(screen, PANEL_BG if current_tab != "ALL" else ACCENT_COLOR, tab_all_rect, border_radius=5)
        pygame.draw.rect(screen, PANEL_BG if current_tab != "RECENT" else ACCENT_COLOR, tab_rec_rect, border_radius=5)

        screen.blit(font_small.render("All ROMs", True, TEXT_COLOR), (tab_all_rect.x + 15, tab_all_rect.y + 6))
        screen.blit(font_small.render("🕒 Recent", True, TEXT_COLOR), (tab_rec_rect.x + 18, tab_rec_rect.y + 6))

        folder_text = f"Folder: {rom_folder if rom_folder else 'No folder selected'}"
        screen.blit(font_small.render(folder_text, True, MUTED_COLOR), (30, 60))

        # Left Panel: Game List
        list_rect = pygame.Rect(30, 95, 440, 395)
        pygame.draw.rect(screen, PANEL_BG, list_rect, border_radius=8)

        visible_count = 13
        max_scroll = max(0, len(active_list) - visible_count)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        for i in range(scroll_offset, min(len(active_list), scroll_offset + visible_count)):
            y_pos = list_rect.y + 10 + (i - scroll_offset) * 28
            item_rect = pygame.Rect(list_rect.x + 8, y_pos, list_rect.width - 16, 26)
            
            is_hover = item_rect.collidepoint(mouse_pos)
            is_selected = (i == selected_idx)

            if is_selected:
                pygame.draw.rect(screen, SELECTED_BG, item_rect, border_radius=5)
            elif is_hover:
                pygame.draw.rect(screen, HOVER_BG, item_rect, border_radius=5)
            
            game_name = clean_title(active_list[i])
            if len(game_name) > 38:
                game_name = game_name[:35] + "..."
            
            screen.blit(font_item.render(game_name, True, TEXT_COLOR), (item_rect.x + 10, item_rect.y + 4))

        if not active_list:
            msg = "No recent games found." if current_tab == "RECENT" else "No .nes ROMs found. Click 'Folder'."
            screen.blit(font_item.render(msg, True, MUTED_COLOR), (list_rect.x + 40, list_rect.y + 180))

        # Right Panel: Cover Art & Game Info
        right_x = 490
        cover_surf = load_cover_art(selected_rom, width=200, height=260)
        screen.blit(cover_surf, (right_x + 40, 95))

        info_box = pygame.Rect(right_x, 375, 280, 115)
        pygame.draw.rect(screen, PANEL_BG, info_box, border_radius=8)
        
        info_title = clean_title(selected_rom) if selected_rom else "Select a Game"
        if len(info_title) > 24: info_title = info_title[:22] + "..."
        screen.blit(font_btn.render(info_title, True, TEXT_COLOR), (info_box.x + 15, info_box.y + 15))
        screen.blit(font_small.render(f"Region: {settings['region']}", True, MUTED_COLOR), (info_box.x + 15, info_box.y + 50))
        screen.blit(font_small.render(f"Cheats: {'ON' if settings['cheats_enabled'] else 'OFF'}", True, MUTED_COLOR), (info_box.x + 15, info_box.y + 75))

        # Bottom Control Buttons (Load File, Folder, Region, Cheats, Audio, PLAY)
        btn_y = 510
        buttons = [
            ("Load File", 20, 95),
            ("Folder", 120, 75),
            (f"{settings['region']}", 200, 75),
            (f"Cheats", 280, 75),
            (f"Audio", 360, 75),
            ("PLAY", 445, 325)
        ]
        btn_rects = []
        for label, bx, bw in buttons:
            brect = pygame.Rect(bx, btn_y, bw, 40)
            btn_rects.append(brect)
            is_hover = brect.collidepoint(mouse_pos)
            bg = ACCENT_COLOR if label == "PLAY" else (HOVER_BG if is_hover else PANEL_BG)
            pygame.draw.rect(screen, bg, brect, border_radius=6)
            
            # Center text inside buttons
            txt_surf = font_btn.render(label, True, TEXT_COLOR)
            screen.blit(txt_surf, (brect.x + (brect.width - txt_surf.get_width()) // 2, brect.y + 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, settings
            elif event.type == pygame.MOUSEWHEEL:
                scroll_offset -= event.y
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if tab_all_rect.collidepoint(event.pos):
                    current_tab = "ALL"
                    selected_idx = 0
                    scroll_offset = 0
                elif tab_rec_rect.collidepoint(event.pos):
                    current_tab = "RECENT"
                    selected_idx = 0
                    scroll_offset = 0
                elif list_rect.collidepoint(event.pos) and active_list:
                    clicked_idx = scroll_offset + (event.pos[1] - (list_rect.y + 10)) // 28
                    if 0 <= clicked_idx < len(active_list):
                        selected_idx = clicked_idx
                elif btn_rects[0].collidepoint(event.pos): # Load File button
                    file_path = select_rom_file(rom_folder)
                    if file_path and os.path.exists(file_path):
                        add_to_recent(settings, file_path)
                        return file_path, settings
                elif btn_rects[1].collidepoint(event.pos): # Folder button
                    folder = select_rom_folder(rom_folder)
                    if folder:
                        rom_folder = folder
                        settings["last_folder"] = folder
                        save_settings(settings)
                        all_rom_list = get_rom_list(rom_folder)
                        selected_idx = 0 if all_rom_list else -1
                elif btn_rects[2].collidepoint(event.pos): # Region toggle
                    regions = ["AUTO", "NTSC", "PAL", "DENDY"]
                    curr_idx = regions.index(settings["region"]) if settings["region"] in regions else 0
                    settings["region"] = regions[(curr_idx + 1) % len(regions)]
                    save_settings(settings)
                elif btn_rects[3].collidepoint(event.pos): # Cheats toggle
                    settings["cheats_enabled"] = not settings["cheats_enabled"]
                    save_settings(settings)
                elif btn_rects[4].collidepoint(event.pos): # Audio toggle
                    settings["audio"] = not settings["audio"]
                    save_settings(settings)
                elif btn_rects[5].collidepoint(event.pos): # PLAY button
                    if selected_rom and os.path.exists(selected_rom):
                        add_to_recent(settings, selected_rom)
                        return selected_rom, settings
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN and active_list:
                    selected_idx = min(len(active_list) - 1, selected_idx + 1)
                    if selected_idx >= scroll_offset + visible_count:
                        scroll_offset += 1
                elif event.key == pygame.K_UP and active_list:
                    selected_idx = max(0, selected_idx - 1)
                    if selected_idx < scroll_offset:
                        scroll_offset -= 1
                elif event.key == pygame.K_RETURN and selected_rom:
                    if os.path.exists(selected_rom):
                        add_to_recent(settings, selected_rom)
                        return selected_rom, settings

        clock.tick(60)

def main():
    settings = load_settings()
    
    pygame.init()
    WIDTH, HEIGHT = 800, 560
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("PY-NES Launcher")

    running_app = True
    while running_app:
        rom_path, settings = run_menu(screen, settings)
        if not rom_path:
            break

        game_w, game_h = 256 * 3, 240 * 3
        screen = pygame.display.set_mode((game_w, game_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
        title_base = f"PY-NES | {clean_title(rom_path)} [{settings['region']}]"
        clock = pygame.time.Clock()

        try:
            emu = Emulator(rom_path, enable_logging=False, region_choice=settings["region"], audio=settings["audio"])
            load_sram(emu.bus, rom_path)
        except Exception as e:
            print(f"Error loading ROM: {e}")
            screen = pygame.display.set_mode((800, 560), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("PY-NES Launcher")
            continue

        cheat_manager = CheatManager()
        cheat_manager.enabled = settings.get("cheats_enabled", False)

        key_map = {
            pygame.K_z: 0x01, pygame.K_x: 0x02, pygame.K_RSHIFT: 0x04, pygame.K_RETURN: 0x08,
            pygame.K_UP: 0x10, pygame.K_DOWN: 0x20, pygame.K_LEFT: 0x40, pygame.K_RIGHT: 0x80
        }

        in_game = True
        frames = 0
        fps_t0 = time.time()
        back_surface = pygame.Surface((256, 240), 0, screen)       # same pixel format as the window
        pal32 = build_pal32(back_surface, emu.ppu.get_palette_array())
        target_fps = int(round(emu.region.fps))                    # 60 NTSC / 50 PAL
        emu_time = 0.0
        gc.collect()
        gc.disable()                                               # no GC pauses mid-frame

        while in_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    in_game = False
                    running_app = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        action = run_pause_menu(screen, game_w, game_h)
                        if action == "QUIT":
                            in_game = False
                        elif action == "SAVE":
                            save_emulator_state(emu, rom_path)
                        elif action == "LOAD":
                            load_emulator_state(emu, rom_path)
                    elif event.key == pygame.K_F5:
                        save_emulator_state(emu, rom_path)
                    elif event.key == pygame.K_F8:
                        load_emulator_state(emu, rom_path)
                    elif event.key == pygame.K_F12:
                        take_screenshot(back_surface, rom_path)
                    elif event.key in key_map:
                        emu.bus.controller[0] |= key_map[event.key]
                elif event.type == pygame.KEYUP:
                    if event.key in key_map:
                        emu.bus.controller[0] &= ~key_map[event.key]

            if not in_game:
                break

            if cheat_manager.enabled and hasattr(emu.bus, 'ram'):
                cheat_manager.apply_ram_cheats(emu.bus.ram)

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
                cheat_status = " [CHEATS ON]" if cheat_manager.enabled else ""
                emu_ms = emu_time / max(frames, 1) * 1000
                pygame.display.set_caption(f"{title_base} | {fps:.1f} FPS | emu {emu_ms:.1f} ms{cheat_status} | ESC for menu")
                frames = 0
                emu_time = 0.0
                fps_t0 = now

            clock.tick_busy_loop(target_fps)

        gc.enable()
        if hasattr(emu, 'bus'):
            save_sram(emu.bus, rom_path)
        if hasattr(emu.bus, 'apu') and emu.bus.apu and emu.bus.apu.out:
            emu.bus.apu.out.close()

        screen = pygame.display.set_mode((800, 560), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("PY-NES Launcher")

    pygame.quit()

if __name__ == "__main__":
    main()
# PY-NES 🎮

A high-performance Nintendo Entertainment System (NES) emulator written in pure Python.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pygame](https://img.shields.io/badge/Pygame--CE-2.5%2B-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange)

PY-NES features a modern, easy-to-use game launcher, full audio synthesis, support for over 30 memory mappers, and uses NumPy to vectorize PPU rendering and APU audio generation for smooth 60 FPS gameplay.

<!-- 
![Launcher](launcher.png)
![Gameplay](gameplay.png)
-->

## ✨ Features

- **Modern Launcher:** A built-in game library UI with cover art support, recent games, and quick toggles for region, audio, and cheats.
- **Full CPU Emulation:** 2A03 (6502 without decimal mode) core. Opcodes are dynamically generated at startup for maximum execution speed. Supports common unofficial opcodes.
- **Vectorized PPU:** Scanline-accurate rendering of backgrounds and sprites, including sprite-0 hit and palette mirroring, accelerated using NumPy.
- **Vectorized APU:** Synthesizes 2 pulse channels, triangle, noise, and DMC audio. Audio chunks are calculated mathematically instead of per-sample looping.
- **30+ Mappers:** Supports a huge variety of classic cartridge boards (NROM, MMC1/2/3/4, UxROM, CNROM, AxROM, VRC2/4/6, VRC1/3, Sunsoft, Irem, Namco, Taito, Camerica, and more).
- **Region Support:** NTSC, PAL, and Dendy timings with auto-detection.
- **Quality of Life:** Save states, battery-backed SRAM auto-saving, screenshots, and Game Genie/RAM cheating.

## 🕹️ Controls

| Key | Action |
| :--- | :--- |
| **Arrow Keys** | D-Pad (Up, Down, Left, Right) |
| **Z** | A Button |
| **X** | B Button |
| **Right Shift** | Select |
| **Enter** | Start |
| **ESC** | Open Pause Menu |
| **F5** | Save State |
| **F8** | Load State |
| **F12** | Take Screenshot (saved to `/snapshots`) |

## 🚀 Getting Started

### Prerequisites
- **Python 3.10** or higher.
- **Tkinter** (Usually bundled with the official Python installer for Windows. Used for the file selection dialogs).

### Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/b84308463-blip/PY-NES.git
cd PY-NES
pip install -r requirements.txt

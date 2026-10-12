"""Run the report's commands and save terminal-style images of their real output.

    python -m Lab2.tools.screenshots              # every screen
    python -m Lab2.tools.screenshots tests bits   # only the named ones

For every screen: Lab2/results/screens/<name>.png (rendered console) and <name>.txt
(the exact captured text). Commands are executed for real; nothing is typed in by hand.
Same look as tools/screenshots.py of lab 1.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT = _ROOT / "Lab2" / "results" / "screens"
_PROMPT = "PS D:" + chr(92) + "..." + chr(92) + "Lab1> "   # full path: D:\Uni\...\Lab1
_FONT = r"C:\Windows\Fonts\consola.ttf"
_BG = (12, 12, 12)
_FG = (204, 204, 204)
_PROMPT_FG = (249, 241, 165)
_CMD_FG = (255, 255, 255)


def _python(*args: str) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, *args], cwd=str(_ROOT), capture_output=True, env=env)
    return (r.stdout + r.stderr).decode("utf-8").replace("\r\n", "\n")


def _get_content(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _main(shown: str) -> tuple[str, callable]:
    """A run of the program; `shown` is the argument string exactly as typed."""
    args = shown.replace(chr(92), "/").split()
    return f"python -m Lab2.bloom.main {shown}", lambda: _python("-m", "Lab2.bloom.main", *args)


def _tool(module: str, shown: str) -> tuple[str, callable]:
    args = shown.replace(chr(92), "/").split()
    return f"python -m {module} {shown}", lambda: _python("-m", module, *args)


def _file(rel: str) -> tuple[str, callable]:
    return f"Get-Content {rel}", lambda: _get_content(rel.replace(chr(92), "/"))


# name -> (characters per image line, [(shown command, how to produce its output)])
SCREENS: dict[str, tuple[int, list[tuple[str, callable]]]] = {
    "small": (84, [
        _file(r"Lab2\demo\demo_small.txt"),
        _main(r"--input Lab2\demo\demo_small.txt --seed 1 --stats"),
    ]),
    "bits": (84, [
        _file(r"Lab2\demo\demo_bits.txt"),
        _main(r"--input Lab2\demo\demo_bits.txt --n 10 --p 0.1 --seed 6 --trace --show-bits"),
        _main(r"--input Lab2\demo\demo_bits.txt --seed 6"),
    ]),
    "delete": (84, [
        _file(r"Lab2\demo\demo_delete.txt"),
        _main(r"--input Lab2\demo\demo_delete.txt --n 10 --p 0.1 --seed 1 --trace --show-bits"),
    ]),
    "malformed": (84, [
        _file(r"Lab2\demo\demo_malformed.txt"),
        _main(r"--input Lab2\demo\demo_malformed.txt --seed 1"),
    ]),
    "tests": (84, [
        ("python -m pytest Lab2 -q", lambda: _python("-m", "pytest", "Lab2", "-q", "-p", "no:cacheprovider")),
    ]),
    "large_runs": (100, [
        _main(rf"--input Lab2\data\n{n}_default_s0.txt --output Lab2\results\scale_{n}_answers.txt "
              r"--timing --stats --seed 1")
        for n in (10_000, 100_000, 1_000_000)
    ]),
    "benchmark": (104, [
        _tool("Lab2.tools.benchmark", r"--summary-only --variants bloom counting --out Lab2\results\bench.csv"),
    ]),
    "fp_fill": (84, [
        _tool("Lab2.tools.fp_experiment", r"--experiment fill --out Lab2\results\fp_fill.csv"),
    ]),
}


def render(name: str, width: int, steps: list[tuple[str, callable]]) -> str:
    def wrap(line: str, limit: int = width) -> list[str]:
        return textwrap.wrap(line, limit, break_on_hyphens=False) or [""]

    lines: list[tuple[str, str]] = []   # (kind, text)
    transcript = []
    for shown, produce in steps:
        output = produce().rstrip("\n")
        transcript.append(_PROMPT + shown + "\n" + output)
        for i, part in enumerate(wrap(shown, width - len(_PROMPT))):
            lines.append(("cmd" if i else "prompt", part))
        for line in output.split("\n"):
            lines.extend(("out", part) for part in wrap(line))
        lines.append(("out", ""))

    font = ImageFont.truetype(_FONT, 18)
    char_w = font.getbbox("M")[2]
    line_h = 24
    pad = 16
    cols = max(len(_PROMPT) + len(text) if kind in ("prompt", "cmd") else len(text) for kind, text in lines)
    img = Image.new("RGB", (pad * 2 + char_w * cols, pad * 2 + line_h * len(lines)), _BG)
    draw = ImageDraw.Draw(img)
    for row, (kind, text) in enumerate(lines):
        y = pad + row * line_h
        if kind == "prompt":
            draw.text((pad, y), _PROMPT, font=font, fill=_PROMPT_FG)
            draw.text((pad + char_w * len(_PROMPT), y), text, font=font, fill=_CMD_FG)
        elif kind == "cmd":
            draw.text((pad + char_w * len(_PROMPT), y), text, font=font, fill=_CMD_FG)
        else:
            draw.text((pad, y), text, font=font, fill=_FG)
    _OUT.mkdir(parents=True, exist_ok=True)
    img.save(_OUT / f"{name}.png")
    text = "\n\n".join(transcript) + "\n"
    (_OUT / f"{name}.txt").write_text(text, encoding="utf-8")
    print(f"[{name}] columns={cols} lines={len(lines)}")
    return text


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(SCREENS)
    for name in names:
        width, steps = SCREENS[name]
        print(f"== {name}")
        print(render(name, width, steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

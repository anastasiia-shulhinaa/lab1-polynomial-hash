"""End-to-end CLI checks: Y/N format, terminator, validation, filter choice, demo files."""

import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _cli(data: bytes | None, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "Lab2.bloom.main", "--seed", "1", *args],
                          input=data, capture_output=True, cwd=str(_ROOT))


@pytest.mark.parametrize("kind", ["auto", "bloom", "counting"])
def test_example_from_the_task(kind):
    r = _cli(b"+ apple\n+ pear\n? apple\n? grape\n#\n", "--filter", kind)
    assert r.returncode == 0
    assert r.stdout == b"Y\nN\n"
    assert r.stderr == b""


def test_duplicate_add_is_allowed():
    assert _cli(b"+ a\n+ a\n? a\n#\n").stdout == b"Y\n"


def test_deletion_switches_to_counters():
    r = _cli(b"+ apple\n+ pear\n? apple\n- pear\n? pear\n? apple\n#\n", "--stats")
    assert r.stdout == b"Y\nN\nY\n"
    assert b"'filter': 'counting'" in r.stderr and b"'minus': 1" in r.stderr


def test_bit_filter_is_used_without_deletions():
    r = _cli(b"+ apple\n? apple\n#\n", "--stats")
    assert r.stdout == b"Y\n"
    assert b"'filter': 'bloom'" in r.stderr
    assert b"'m': 9592955" in r.stderr and b"'l': 7" in r.stderr and b"'bytes': 1199120" in r.stderr


def test_bit_filter_rejects_deletion():
    r = _cli(b"+ pear\n- pear\n? pear\n#\n", "--filter", "bloom")
    assert r.returncode == 0
    assert r.stdout == b"Y\n"
    assert r.stderr.count(b"warning") == 1


def test_duplicate_add_then_single_remove():
    assert _cli(b"+ a\n+ a\n- a\n? a\n#\n").stdout == b"N\n"


def test_lines_after_terminator_are_ignored():
    assert _cli(b"+ aa\n? aa\n#\n? aa\n+ bb\n").stdout == b"Y\n"


def test_missing_terminator():
    r = _cli(b"+ abba\n? abba\n")
    assert r.returncode == 0 and r.stdout == b"Y\n"


def test_only_terminator():
    r = _cli(b"#\n")
    assert r.returncode == 0 and r.stdout == b""


def test_crlf_line_endings():
    assert _cli(b"+ abba\r\n? abba\r\n#\r\n").stdout == b"Y\n"


def test_both_cases_are_accepted_and_distinct():
    r = _cli(b"+ Apple\n? apple\n? Apple\n? APPLE\n#\n")
    assert r.stdout == b"N\nY\nN\n" and r.stderr == b""


def test_length_limits():
    r = _cli(b"+ a\n+ abcdefghijklmno\n+ abcdefghijklmnop\n? a\n? abcdefghijklmno\n#\n")
    assert r.stdout == b"Y\nY\n"
    assert r.stderr.count(b"warning") == 1


def test_malformed_lines_are_skipped_without_shifting_the_rest():
    r = _cli(b"+ abba\n* abc\n+ ab1\n+\n+ two words\n? abba\n? abc\n#\n")
    assert r.returncode == 0
    assert r.stdout == b"Y\nN\n"
    assert r.stderr.count(b"warning") == 4
    assert b"line 4" in r.stderr


def test_strict_mode_fails_on_malformed():
    assert _cli(b"+ abba\n+ ab1\n#\n", "--strict").returncode == 1


def test_input_output_files(tmp_path):
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    src.write_bytes(b"? aa\n+ aa\n? aa\n#\n")
    r = _cli(None, "--input", str(src), "--output", str(dst))
    assert r.returncode == 0 and r.stdout == b""
    assert dst.read_bytes() == b"N\nY\n"


def test_small_parameters_and_bit_dump(tmp_path):
    bits = tmp_path / "bits.txt"
    r = _cli(b"+ apple\n? apple\n#\n", "--n", "10", "--p", "0.1", "--show-bits", str(bits), "--stats")
    assert r.stdout == b"Y\n"
    assert b"'m': 49" in r.stderr and b"'l': 4" in r.stderr
    lines = bits.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("T (bloom): m=49, l=4")
    assert len("".join(lines[1].split()[1:])) == 49


def test_trace_table(tmp_path):
    trace = tmp_path / "trace.md"
    r = _cli(None, "--input", "Lab2/demo/demo_delete.txt", "--trace", str(trace))
    assert r.stdout == b"Y\nN\nY\nY\n"
    rows = trace.read_text(encoding="utf-8").splitlines()
    assert rows[2].startswith("| 1 | `+ apple` |") and rows[2].endswith("| лічильники +1 | — |")
    assert rows[4].endswith("| уже є, без змін | — |")
    assert rows[6].endswith("| лічильники -1 | — |")
    assert rows[7].startswith("| 6 | `? apple` |") and rows[7].endswith("| N |")
    assert rows[9].endswith("| немає, без змін | — |")
    assert rows[-1] == "| 10 | `#` | | | завершення |"
    assert len(rows[2].split("|")[3].split(",")) == 7      # l positions


def test_invalid_parameters_are_rejected():
    assert _cli(b"#\n", "--p", "1.5").returncode == 2
    assert _cli(b"#\n", "--n", "0").returncode == 2


# ---- demo files ----

def test_demo_small():
    r = _cli(None, "--input", "Lab2/demo/demo_small.txt")
    assert r.stdout == b"N\nY\nY\nN\n" and r.stderr == b""


def test_demo_delete():
    r = _cli(None, "--input", "Lab2/demo/demo_delete.txt")
    assert r.stdout == b"Y\nN\nY\nY\n"


def test_demo_malformed():
    r = _cli(None, "--input", "Lab2/demo/demo_malformed.txt")
    assert r.stdout == b"Y\nN\n"
    assert r.stderr.count(b"warning") == 4


def test_demo_bits_shows_a_false_positive():
    """10 strings in a 49-bit filter: with seed 6 the absent 'banana' answers Y."""
    args = ("--input", "Lab2/demo/demo_bits.txt", "--n", "10", "--p", "0.1")
    r = subprocess.run([sys.executable, "-m", "Lab2.bloom.main", "--seed", "6", *args],
                       capture_output=True, cwd=str(_ROOT))
    assert r.stdout == b"Y\nY\nN\nN\n"
    # the same input in the full-size filter gives the exact answer
    r = subprocess.run([sys.executable, "-m", "Lab2.bloom.main", "--seed", "6", args[0], args[1]],
                       capture_output=True, cwd=str(_ROOT))
    assert r.stdout == b"Y\nN\nN\nN\n"

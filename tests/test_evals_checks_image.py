"""Objective checks for a generated image.

Engine-independent: whatever produced the file, these are the ways it can be
worthless. A solid grey square is a real failure mode of a broken pipeline and
it is invisible to "did the command exit 0".
"""
import struct
import zlib
from pathlib import Path

import pytest

from harness.checks import image as image_check


def png(width, height, pixel=(200, 30, 30), noise=False):
    """Minimal valid PNG, optionally with varied pixels."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter byte per scanline
        for x in range(width):
            if noise:
                raw += bytes(((x * 37 + y * 91) % 256,
                              (x * 13 + y * 7) % 256,
                              (x + y) % 256))
            else:
                raw += bytes(pixel)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw)))
            + chunk(b"IEND", b""))


@pytest.fixture
def img(tmp_path):
    def make(**kw):
        p = tmp_path / "out.png"
        p.write_bytes(png(**kw))
        return p
    return make


def test_valid_varied_image_passes(img):
    r = image_check.check(img(width=64, height=64, noise=True))
    assert r.ok
    assert r.width == 64 and r.height == 64


def test_missing_file_fails(tmp_path):
    r = image_check.check(tmp_path / "nope.png")
    assert not r.ok
    assert "no output" in r.reason.lower()


def test_empty_file_fails(tmp_path):
    p = tmp_path / "e.png"
    p.write_bytes(b"")
    assert not image_check.check(p).ok


def test_not_an_image_fails(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"this is not a png")
    r = image_check.check(p)
    assert not r.ok
    assert "not a" in r.reason.lower() or "decode" in r.reason.lower()


def test_blank_image_fails(img):
    """A uniform canvas is what a broken pipeline emits, and it exits 0."""
    r = image_check.check(img(width=64, height=64, noise=False))
    assert not r.ok
    assert "uniform" in r.reason.lower() or "blank" in r.reason.lower()


def test_expected_dimensions_are_enforced(img):
    p = img(width=64, height=64, noise=True)
    assert image_check.check(p, expect=(64, 64)).ok
    r = image_check.check(p, expect=(512, 512))
    assert not r.ok
    assert "512" in r.reason


def test_tiny_image_is_flagged(img):
    r = image_check.check(img(width=8, height=8, noise=True))
    assert any("small" in w.lower() for w in r.warnings)

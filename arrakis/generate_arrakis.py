#!/usr/bin/env python3
"""Build a 3D model of the north polar region of Arrakis, as set up in Dune, Part One.

Writes two files next to this script:
  arrakis.glb   terrain mesh with vertex colours (glTF 2.0 binary)
  arrakis.html  standalone viewer with the GLB embedded

Standard library only. Horizontal units are km. Heights are stored as
km * VERTICAL_EXAGGERATION so the Shield Wall reads at a glance.

Map north is -z and east is +x. The sources checked give three facts: the Shield
Wall rings the north pole, Arrakeen sits inside that ring, and Carthag is about
200 km north-east of Arrakeen. Everything else here is a schematic choice.
"""

import array
import base64
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

GRID = 255                      # vertices per side; GRID * GRID < 65536 so indices fit in uint16
TILE_KM = 320.0                 # width of the square tile
VERTICAL_EXAGGERATION = 6.0
SEED = 1965

WALL_RADIUS_KM = 105.0          # Shield Wall: a ring of rock around the north pole
WALL_HALF_WIDTH_KM = 9.0
ARRAKEEN_KM = (-38.0, 30.0)     # inside the ring, sheltered by it
CARTHAG_DISTANCE_KM = 200.0
NORTH_EAST = (math.sqrt(0.5), -math.sqrt(0.5))
CARTHAG_KM = (ARRAKEEN_KM[0] + CARTHAG_DISTANCE_KM * NORTH_EAST[0],
              ARRAKEEN_KM[1] + CARTHAG_DISTANCE_KM * NORTH_EAST[1])

DUNE_ANGLE = 0.35               # radians; crest direction, roughly across the prevailing wind
DUNE_SPACING_KM = 8.0

SAND_LIGHT = (0.87, 0.69, 0.45)
SAND_DARK = (0.66, 0.46, 0.28)
ROCK_LIGHT = (0.50, 0.38, 0.31)
ROCK_DARK = (0.33, 0.25, 0.21)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def lerp3(a, b, t):
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))


def _hash(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 4294967295.0


def value_noise(x, y, seed):
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)
    a = _hash(x0, y0, seed)
    b = _hash(x0 + 1, y0, seed)
    c = _hash(x0, y0 + 1, seed)
    d = _hash(x0 + 1, y0 + 1, seed)
    top = a + (b - a) * sx
    bottom = c + (d - c) * sx
    return top + (bottom - top) * sy


def fbm(x, y, octaves, seed):
    """Fractal value noise in [0, 1]."""
    total, amp, freq, norm = 0.0, 0.5, 1.0, 0.0
    for o in range(octaves):
        total += amp * value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


def terrain_height_km(x, z):
    r = math.hypot(x, z)
    theta = math.atan2(z, x)

    # Shield Wall: 1 at the crest of the ring, falling to 0 a few km either side
    ring = max(0.0, 1.0 - abs(r - WALL_RADIUS_KM) / WALL_HALF_WIDTH_KM) ** 0.5

    # Broad plain, with a shallow sheltered basin just inside the wall
    inside = clamp((WALL_RADIUS_KM - WALL_HALF_WIDTH_KM - r) / 12.0)
    base = 0.25 + 0.10 * fbm(x * 0.012, z * 0.012, 3, SEED) - 0.12 * inside

    # Transverse dunes, crests across the prevailing wind
    u = x * math.cos(DUNE_ANGLE) + z * math.sin(DUNE_ANGLE)
    warp = 1.5 * (fbm(x * 0.04, z * 0.04, 3, SEED + 7) - 0.5)
    crest = 0.5 + 0.5 * math.sin(2 * math.pi * u / DUNE_SPACING_KM + warp)
    amp = 0.02 + 0.08 * fbm(x * 0.03, z * 0.03, 3, SEED + 13)
    # keep both settlements on flat ground
    d_arrakeen = math.hypot(x - ARRAKEEN_KM[0], z - ARRAKEEN_KM[1])
    d_carthag = math.hypot(x - CARTHAG_KM[0], z - CARTHAG_KM[1])
    calm = clamp(min(d_arrakeen, d_carthag) / 25.0, 0.1, 1.0)
    dunes = amp * crest ** 1.6 * calm * (1.0 - ring)

    # Wall crest height varies a little around the ring
    peak = 4.1 + 0.1 * fbm(math.cos(theta) * 1.5 + 4, math.sin(theta) * 1.5 + 4, 3, SEED + 21)
    rough = 0.15 * (fbm(x * 0.12, z * 0.12, 4, SEED + 31) - 0.5)
    wall = (peak + rough) * ring

    return base + dunes + wall


def surface_color(x, z, h, slope):
    sand = lerp3(SAND_LIGHT, SAND_DARK, fbm(x * 0.05, z * 0.05, 3, SEED + 41))
    rock = lerp3(ROCK_LIGHT, ROCK_DARK, fbm(x * 0.3, z * 0.3, 3, SEED + 51))
    t = max(smoothstep(0.18, 0.5, slope), smoothstep(1.2, 2.6, h))
    return tuple(int(round(255 * clamp(c))) for c in lerp3(sand, rock, t))


def write_glb(path, positions, normals, colors, indices):
    if sys.byteorder != "little":
        raise SystemExit("this writer assumes a little-endian host")

    blob = bytearray()
    views = []

    def add_view(data, target):
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    vertex_count = len(positions) // 3
    pos_view = add_view(positions.tobytes(), 34962)
    nrm_view = add_view(normals.tobytes(), 34962)
    col_view = add_view(bytes(colors), 34962)
    idx_view = add_view(indices.tobytes(), 34963)

    accessors = [
        {"bufferView": pos_view, "componentType": 5126, "count": vertex_count, "type": "VEC3",
         "min": [min(positions[0::3]), min(positions[1::3]), min(positions[2::3])],
         "max": [max(positions[0::3]), max(positions[1::3]), max(positions[2::3])]},
        {"bufferView": nrm_view, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
        {"bufferView": col_view, "componentType": 5121, "normalized": True, "count": vertex_count, "type": "VEC4"},
        {"bufferView": idx_view, "componentType": 5123, "count": len(indices), "type": "SCALAR",
         "min": [min(indices)], "max": [max(indices)]},
    ]

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_arrakis.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Arrakis_north_polar_region"}],
        "meshes": [{"name": "Arrakis", "primitives": [
            {"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}
        ]}],
        "materials": [{"name": "Sand_and_rock", "doubleSided": True, "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.95}}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    bin_bytes = bytes(blob) + b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        f.write(json_bytes)
        f.write(struct.pack("<I4s", len(bin_bytes), b"BIN\0"))
        f.write(bin_bytes)


def main():
    cell = TILE_KM / (GRID - 1)
    half = TILE_KM / 2
    coords = [-half + k * cell for k in range(GRID)]
    height = [[terrain_height_km(x, z) for z in coords] for x in coords]

    positions = array.array("f")
    normals = array.array("f")
    colors = bytearray()
    for i, x in enumerate(coords):
        i0, i1 = max(i - 1, 0), min(i + 1, GRID - 1)
        for j, z in enumerate(coords):
            j0, j1 = max(j - 1, 0), min(j + 1, GRID - 1)
            h = height[i][j]
            dhdx = (height[i1][j] - height[i0][j]) / ((i1 - i0) * cell)
            dhdz = (height[i][j1] - height[i][j0]) / ((j1 - j0) * cell)
            positions.extend((x, VERTICAL_EXAGGERATION * h, z))
            nx, ny, nz = -VERTICAL_EXAGGERATION * dhdx, 1.0, -VERTICAL_EXAGGERATION * dhdz
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            normals.extend((nx / length, ny / length, nz / length))
            colors.extend(surface_color(x, z, h, math.hypot(dhdx, dhdz)) + (255,))

    indices = array.array("H")
    for i in range(GRID - 1):
        for j in range(GRID - 1):
            v00 = i * GRID + j
            v10 = (i + 1) * GRID + j
            v01 = i * GRID + j + 1
            v11 = (i + 1) * GRID + j + 1
            indices.extend((v00, v01, v11, v00, v11, v10))

    def marker(name, desc, color, x, z):
        return {"name": name, "desc": desc, "color": color, "x": x, "z": z,
                "y": VERTICAL_EXAGGERATION * terrain_height_km(x, z)}

    wall_theta = math.radians(-150)
    markers = [
        marker("Arrakeen", "Atreides capital, sheltered inside the Shield Wall", "#ffd27a", *ARRAKEEN_KM),
        marker("Carthag", "Former capital, about 200 km north-east of Arrakeen", "#d9534f", *CARTHAG_KM),
        marker("Shield Wall", "Rock ring around the north pole, sheltering Arrakeen from coriolis storms",
               "#9fb3c8", WALL_RADIUS_KM * math.cos(wall_theta), WALL_RADIUS_KM * math.sin(wall_theta)),
    ]

    glb_path = os.path.join(HERE, "arrakis.glb")
    write_glb(glb_path, positions, normals, colors, indices)

    with open(glb_path, "rb") as f:
        glb_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(os.path.join(HERE, "viewer.template.html"), encoding="utf-8") as f:
        template = f.read()
    html = template.replace("__GLB_BASE64__", glb_b64).replace("__MARKERS__", json.dumps(markers))
    with open(os.path.join(HERE, "arrakis.html"), "w", encoding="utf-8") as f:
        f.write(html)

    all_h = [h for row in height for h in row]
    print(f"vertices: {GRID * GRID}  triangles: {(GRID - 1) ** 2 * 2}")
    print(f"height range: {min(all_h):.2f} km to {max(all_h):.2f} km")
    print(f"wrote {glb_path} ({os.path.getsize(glb_path) / 1e6:.2f} MB)")
    print(f"wrote {os.path.join(HERE, 'arrakis.html')} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

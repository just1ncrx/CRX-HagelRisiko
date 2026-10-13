#!/usr/bin/env python3
"""VII-Composite (DWD, ODIM-HDF5) -> Hagelrisiko im NATIVEN DWD-Raster.

Kein Warp nach Web Mercator, keine Einfaerbung. Ausgabe: rohe uint8-Bytes,
keine Dateiendung.
Layout: ysize Zeilen x xsize Spalten, row-major, Zeile 0 = Norden (wie im HDF5).
Dateigroesse immer exakt xsize * ysize Bytes.

Code-Bedeutung in der Ausgabedatei:
    0       = kein Hagelrisiko (unter MIN_VISIBLE) -> transparent
    1..N    = Stufe i (1-basiert) aus HAIL_TABLE; Farbe/Schwelle siehe Client-Metadaten
    255     = kein Daten / ausserhalb
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/vii")
OUT_DIR = Path("output/vii")

# composite_VII_20261009_1020-hd5  (Vorhersageschritt _LLL optional)
FILENAME_RE = re.compile(
    r"composite_vii_(\d{8})_(\d{4})(?:_(\d{3}))?-hd5", re.IGNORECASE
)

# Fallbacks, falls Attribute in der Datei fehlen
DEFAULT_GAIN = 1.0
DEFAULT_OFFSET = 0.0
DEFAULT_NODATA = 65535

# Darunter: Code 0 (transparent)
MIN_VISIBLE = 0.005

NODATA_BYTE = 255

# Jeder Wert bekommt die Stufe der letzten Schwelle <= Wert.
# (Schwelle, Hex-Farbe) - Einheit = Einheit des Datasets nach gain/offset
# Die Farben werden nur noch fuer die Client-Metadaten gebraucht.
HAIL_TABLE: list[tuple[float, str]] = [
    #GRAU
    (0.005,  "#B1B1B1"),
    (0.04,   "#C2C2C2"),
    (0.075,  "#D3D3D3"),
    (0.1625, "#EBEBEB"),
    #GRÜN
    (0.25,   "#C8FFBE"),
    (0.375,  "#B4FAAA"),
    (0.5,    "#96F58C"),
    (0.625,  "#78F573"),
    (0.75,   "#50F050"),
    (0.875,  "#37D23C"),
    (1.0,    "#1EB41E"),
    (1.25,   "#0FA00F"),
    #BLAU
    (1.5,    "#1E6EEB"),
    (1.75,   "#2882F0"),
    (2.0,    "#3C96F5"),
    (2.5,    "#50A5F5"),
    (3.0,    "#78B9FA"),
    (3.5,    "#96D2FA"),
    (4.0,    "#B4F0FA"),
    #GELB/ORANGE
    (4.5,    "#D9ECB9"),
    (5.0,    "#FFE878"),
    (6.0,    "#FFC03C"),
    (7.0,    "#FFA000"),
    (8.5,    "#FF6000"),
    (10,     "#F53200"),
    (12.5,   "#E11400"),
    (15,     "#C00000"),
    (17.5,   "#A50000"),
    (20,     "#91001E"),
    (25,     "#7B0064"),
    (30,     "#6400AA"),
    (35,     "#AA1ED2"),
    (40,     "#CD50EB"),
    (50,     "#EB8CFF"),
    (60,     "#F5AAFF"),
    (70,     "#FFC8FF")
]

BERLIN = ZoneInfo("Europe/Berlin")


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_filename(filename: str) -> tuple[datetime, int]:
    """Liefert (Basiszeit UTC, Vorhersageschritt in Minuten; 0 falls keiner)."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            "Dateiname passt nicht zum Schema "
            f"'composite_VII_yyyymmdd_HHMM[_LLL]-hd5': {filename}"
        )
    date_str, time_str, lead_str = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc), int(lead_str or 0)


def _attr_str(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def find_dataset(h5file: h5py.File) -> h5py.Dataset:
    """Erstes 2D-'data'-Dataset (bei Composites i.d.R. dataset1/data1/data)."""
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError("Kein 2D-Datensatz in der VII-Datei gefunden.")
    return candidates[0]


def read_values(ds: h5py.Dataset) -> np.ndarray:
    """Liest das Dataset, wendet gain/offset an (NaN = kein Datum)."""
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    attrs = what.attrs if what is not None else {}

    gain = float(_scalar(attrs.get("gain", DEFAULT_GAIN)))
    offset = float(_scalar(attrs.get("offset", DEFAULT_OFFSET)))
    nodata = _scalar(attrs.get("nodata", DEFAULT_NODATA))
    undetect = _scalar(attrs["undetect"]) if "undetect" in attrs else None
    quantity = _attr_str(attrs.get("quantity", ""))

    raw = ds[()]
    values = raw.astype(np.float64) * gain + offset
    values[raw == nodata] = np.nan
    if undetect is not None and undetect != nodata:
        values[raw == undetect] = 0.0

    print(
        f"Dataset: quantity='{quantity}', gain={gain}, offset={offset}, "
        f"nodata={nodata}, undetect={undetect}"
    )
    valid = values[np.isfinite(values)]
    if valid.size:
        print(f"Wertebereich: {valid.min():.2f} .. {valid.max():.2f}")
    return values


def _parse_odim_time(date_s, time_s) -> datetime | None:
    try:
        d = _attr_str(date_s).strip()
        t = _attr_str(time_s).strip().ljust(6, "0")[:6]
        return datetime.strptime(d + t, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def read_metadata(h5file: h5py.File, ds: h5py.Dataset) -> dict:
    """Sammelt Zeit-/Produktmetadaten aus den ODIM-'what'-Gruppen."""
    whats: list[h5py.Group] = []
    for grp in (ds.parent, ds.parent.parent, h5file):
        w = grp.get("what")
        if w is not None:
            whats.append(w)

    def first(key: str):
        for w in whats:
            if key in w.attrs:
                return w.attrs[key]
        return None

    start = _parse_odim_time(first("startdate"), first("starttime")) \
        if first("startdate") is not None else None
    end = _parse_odim_time(first("enddate"), first("endtime")) \
        if first("enddate") is not None else None
    nominal = _parse_odim_time(first("date"), first("time")) \
        if first("date") is not None else None

    return {
        "product": _attr_str(first("product")) if first("product") is not None else None,
        "quantity": _attr_str(first("quantity")) if first("quantity") is not None else None,
        "nominal_time": nominal,
        "start": start,
        "end": end,
    }


def find_where_group(h5file: h5py.File) -> h5py.Group | None:
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found: list[h5py.Group] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_str(key: str) -> str:
        return _attr_str(where.attrs[key])

    def as_float(key: str) -> float:
        return float(_scalar(where.attrs[key]))

    return {
        "projdef": as_str("projdef"),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


# --------------------------------------------------------------------------- #
# Kodieren (statt Einfaerben)
# --------------------------------------------------------------------------- #
def encode_levels(values: np.ndarray) -> np.ndarray:
    """Wert -> uint8-Code (0 = nichts, 1..N = Stufe, 255 = nodata)."""
    thresholds = np.array([t for t, _ in HAIL_TABLE], dtype=np.float64)

    codes = np.zeros(values.shape, dtype=np.uint8)
    codes[~np.isfinite(values)] = NODATA_BYTE

    visible = np.isfinite(values) & (values >= max(MIN_VISIBLE, thresholds[0]))
    if visible.any():
        idx = np.searchsorted(thresholds - 1e-6, values[visible], side="right") - 1
        idx = np.clip(idx, 0, len(thresholds) - 1)
        codes[visible] = idx + 1  # 1-basiert, 0 bleibt "nichts"
    return codes


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    candidates = sorted(
        (p for p in SRC_DIR.iterdir() if FILENAME_RE.match(p.name)),
        key=lambda p: parse_filename(p.name),
    ) if SRC_DIR.exists() else []
    if not candidates:
        sys.exit(f"Keine VII-Datei (composite_VII_*-hd5) in {SRC_DIR} gefunden.")
    src_path = candidates[-1]
    base_ts, lead_min = parse_filename(src_path.name)
    print(f"Quelle: {src_path.name}")

    with h5py.File(src_path, "r") as f:
        ds = find_dataset(f)
        values_native = read_values(ds)
        meta = read_metadata(f, ds)
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden.")
        grid = extract_grid_info(where)

    # Fallback: Zeiten aus Dateiname, falls nicht in der Datei
    if meta["start"] is None:
        meta["start"] = base_ts
    if meta["end"] is None:
        meta["end"] = meta["nominal_time"] or base_ts
    if meta["start"] > meta["end"]:
        meta["start"], meta["end"] = meta["end"], meta["start"]
    ts = meta["end"]

    print(
        f"Daten von {meta['start']:%Y-%m-%d %H:%M:%S} UTC "
        f"bis {meta['end']:%Y-%m-%d %H:%M:%S} UTC "
        f"({meta['start'].astimezone(BERLIN):%H:%M} - "
        f"{meta['end'].astimezone(BERLIN):%H:%M} Berlin)"
    )

    h, w = values_native.shape
    if (h, w) != (grid["ysize"], grid["xsize"]):
        sys.exit(f"Rastergröße {values_native.shape} passt nicht zu where-Info "
                 f"({grid['ysize']} x {grid['xsize']}).")
    print(f"Natives Raster: {w} x {h} px, {grid['xscale']} m / px")

    codes = encode_levels(values_native)

    # Speichern: rohe Bytes, keine Dateiendung, Zeile 0 = Norden
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"vii_{ts.astimezone(BERLIN):%Y%m%d_%H%M}"
    np.ascontiguousarray(codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({w}x{h}, {codes.nbytes} Bytes)")

    # Metadaten fuer den Client (aendern sich nicht -> einmalig kopieren)
    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    client_meta = {
        "width": w,
        "height": h,
        "projdef": grid["projdef"],
        "xscale": grid["xscale"],
        "yscale": grid["yscale"],
        "ll_x": ll_x,
        "ll_y": ll_y,
        "rowOrder": "north-to-south",
        "nodataCode": NODATA_BYTE,
        "emptyCode": 0,
        "levels": [
            {"code": i + 1, "min": t, "color": c}
            for i, (t, c) in enumerate(HAIL_TABLE)
        ],
    }
    print("Client-Metadaten:", json.dumps(client_meta))


if __name__ == "__main__":
    main()
"""
pymims_metadata.py — Extended metadata extraction for NanoSIMS .im files.

The .im file's binary header carries far more than the basic acquisition
geometry that pymims.py reads by default. Cameca's analyser tool generates
HTML-format reports listing every analytical parameter (Cs HV, lens
voltages, slit widths, detector HV, dead times, etc.). This module
extracts the same information programmatically, so any of those fields
can be queried in code rather than read from a paper print-out.

The offsets below were determined by reverse-engineering a known .im
file against its companion HTML report. They are believed to be stable
across NS50/50L acquisitions but may vary on different instrument
generations or firmware versions. Cross-validate against an HTML report
on your own data the first time you use this on a new instrument; the
``parse_full_metadata`` function returns a structured dict that is easy
to compare row-by-row.

Limitations
-----------
- This is not a complete decode of the header. The fields extracted here
  are the ones that appear in the HTML report and were validated against
  a 2026 NS50L acquisition with a 6-detector + SE configuration.
- Detector slot indexing follows the Cameca convention: slots #1, #2, #3,
  #6, #7 are commonly populated (matching the multicollector geometry);
  #4 and #5 are typically unused. The HTML reflects this and so does
  this parser.
- Mass/radius/plate fields at slot offsets 4 and 5 (if present) will
  contain whatever happens to be in those bytes — they may be junk if
  no detector is installed there.

Author : G. McMahon (with AI-assisted development)
Created: May 2026 (v0.8 work)

Usage
-----
    from pymims import MimsImage
    img = MimsImage('myfile.im')
    meta = img.read_full_metadata()
    print(meta['mass_table'])
    print(meta['detectors'])
    img.print_metadata()    # convenience pretty-printer
"""

import struct
from collections import OrderedDict


# ── Header offsets (validated against 2026 NS50L .im + companion HTML) ──────

# Acquisition basics — known and parsed by pymims.py already.
# Re-listed here for documentation and self-contained parsing.
OFFSETS_BASIC = {
    'duration_s'      : (140, '<I'),    # total acquisition time in seconds
    'cycles_planned'  : (144, '<I'),    # number of cycles SET in analysis spec
    'stage_x_um'      : (20,  '<i'),    # sample stage X (HTML units: µm)
    'stage_y_um'      : (24,  '<i'),    # sample stage Y (µm)
    'stage_z_um'      : (76,  '<i'),    # sample stage Z (µm)
    'dwell_us'        : (3968, '<I'),   # pixel dwell time in microseconds
                                         # (OpenMIMS divides by 1000 for "ms/xy")
}

# Magnet trim DAC values, from the "Bfield" row in HTML
OFFSETS_BFIELD = {
    'bfield_dac' : (3960, '<i'),
    'q_dac'      : (3988, '<i'),
    'lf4_dac'    : (3992, '<i'),
    'hex_dac'    : (3996, '<i'),
}

# Mass table: one row per detector slot. Slot n (1..7) starts at
# DETECTOR_BASE + (n-1) * DETECTOR_STRIDE.
DETECTOR_BASE   = 4252       # row 0 (slot #1) starts here
DETECTOR_STRIDE = 208        # bytes per row

# Within each detector row:
DETECTOR_FIELDS = {
    'mass_amu'        : (32,  '<d'),    # nominal mass in amu
    'radius_mm'       : (40,  '<d'),    # ion-optical radius in mm
    'neg_plate_dac'   : (48,  '<i'),    # negative plate DAC
    'pos_plate_dac'   : (52,  '<i'),    # positive plate DAC
    'exit_slit_um'    : (120, '<d'),    # exit slit width in µm
}

# Primary column parameters (Cs source, lenses)
OFFSETS_PRIMARY = {
    # Cs source
    'ionizer_mA'      : (24084, '<d'),  # double precision
    'reservoir_mA'    : (24092, '<d'),  # double precision
    'cs_hv_v'         : (7600,  '<i'),  # Cs HV in volts (int)
    # Primary lenses (DACs are int32; voltages are derived but not always stored)
    'l1_v'            : (7392,  '<i'),  # L1 in volts
    'oct45_v'         : (7540,  '<d'),  # Oct45 in volts (double; DAC at oct45_dac)
    'oct90_v'         : (7548,  '<d'),  # Oct90 in volts
    'oct45_dac'       : (7676,  '<i'),  # Oct45 in DAC counts
    'oct90_dac'       : (7680,  '<i'),  # Oct90 in DAC counts
    'e0p_v'           : (7556,  '<d'),  # Primary focalization (E0P) in volts
    # Be / E-gun
    'be_dac'          : (24068, '<i'),
}

# Secondary column parameters (sample HV, transfer optics)
OFFSETS_SECONDARY = {
    'e0w_dac'         : (24000, '<i'),  # sample HV DAC
    'e0s_dac'         : (24004, '<i'),  # secondary focalization DAC
    'lf5_dac'         : (24020, '<i'),
    'p2_dac'          : (24024, '<i'),
    'p3_dac'          : (24028, '<i'),
}


# Diaphragm and slit position selectors. Each is a single i32 holding
# the chosen position, followed by an array of 5 i32 values giving the
# available physical sizes (diameters in µm for diaphragms, widths/heights
# in µm for slits).
OFFSETS_SELECTORS = {
    'd1_position'         : (7484, '<i'),    # primary diaphragm D1
    'es_position'         : (7932, '<i'),    # entrance slit
    'as_position'         : (8016, '<i'),    # aperture slit
}

# Address ranges of the available-position arrays. Each is 5 × i32
# starting at the given offset.
OFFSETS_SELECTOR_TABLES = {
    'd1_diameters_um'     : (7492, 5, '<i'),  # 750/300/200/150/100 µm slots
    'es_widths_um'        : (7940, 5, '<i'),  # 50/40/30/20/10 µm slots
    'es_heights_um'       : (7980, 5, '<i'),  # 220/220/180/140/100 µm slots
    'as_widths_um'        : (8024, 5, '<i'),  # 350/200/150/80/40 µm slots
    'as_heights_um'       : (8064, 5, '<i'),  # 250/200/150/80/40 µm slots
}

# Detector parameters table — one row per detector slot, tightly packed.
# These are AT FIXED OFFSETS not the per-detector mass-table offsets.
DETECTOR_PARAMS_BASE_HV   = 24116   # int32, slot n at HV_BASE + (n-1)*4
DETECTOR_PARAMS_BASE_THR  = 24156   # f64,   slot n at THR_BASE + (n-1)*8
# Dead time and yield are observed to be uniform across detectors in
# practice (44 ns / 100% in the test file). They live in the same area
# but vary less. We don't yet have validated offsets for them per-slot.


# ── Pending offsets — awaiting cross-validation against OpenMIMS or HTML ────
#
# These are fields known to exist in the Cameca .im header but whose exact
# binary offset has not been validated against a known reference. They live
# here as placeholders so users can fill them in once a target value is
# available (e.g. from running OpenMIMS on the same file and reading off
# the value).
#
# To validate: open this file's `260304_S5_E3A1D5-4um.im` in OpenMIMS,
# read off the field's value, then run scripts/find_offset.py with the
# value to locate the binary offset, then update the entry below.
#
# Format: 'name' : (offset, fmt, note). Set offset to None to skip; once
# validated, set the offset to the located byte position.
OFFSETS_PENDING = {
    # FCP = Faraday Cup Primary current.
    # 2017 file: at offset 7092/7096 (validated).
    # 2025/2026 files: those bytes are reused for the section label
    # "Anal_param_nano" string. Despite cross-checking offsets across
    # multiple files using value-range searches (15-30 for nA, 20000-30000
    # for pA, both i32 and f64 encodings), no field with a pattern
    # consistent with FCP storage was found in the new header layout.
    # Tentative conclusion: Cameca may no longer write FCP to the .im
    # header in 2025+ firmware; the values may live in cur_setup.set
    # (referenced at offset 8677) or be excluded entirely from .im.
    # If a known T0/END pair from the NanoSIMS computer becomes available
    # for a 2025+ acquisition, run scripts/find_offset.py to settle this.
    'fcp_current_t0' : (None, '<i',
                        'FCP T0: located at offset 7092 in 2017-format files; '
                        'not findable in 2025+ Anal_param_nano header layout'),
    'fcp_current_end': (None, '<i',
                        'FCP END: located at offset 7096 in 2017-format files; '
                        'not findable in 2025+ Anal_param_nano header layout'),
    # Prim L0 — primary lens 0. Sits adjacent to L1 (offset 7392) but
    # the value is 0 in our validation file so we can't disambiguate
    # between candidates 7384 and 7388.
    'prim_l0_v'      : (None, '<i',
                        'Primary lens L0; candidates at offsets 7384 or 7388'),
    # Boolean correction flags from OpenMIMS. Both false in our 2026
    # validation file so we can't search by value (0 matches everywhere).
    'dead_time_corrected': (None, '<i',
                            'Dead-time correction applied flag (bool)'),
    'qsa_corrected'      : (None, '<i',
                            'Quasi-Simultaneous Arrival correction flag (bool)'),
}


def _read(data, offset, fmt):
    """Read a single value from ``data`` at ``offset`` using struct ``fmt``.
    Returns None if the read goes past end-of-data."""
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        return None
    try:
        return struct.unpack_from(fmt, data, offset)[0]
    except struct.error:
        return None


def _read_pending(data, offset_dict, warn=True):
    """
    Read pending fields. Each entry has (offset, fmt, note); if offset
    is None, returns None for that entry. If offset is set but the read
    fails (e.g. out of bounds), returns None and prints a warning.
    """
    out = OrderedDict()
    for name, (off, fmt, note) in offset_dict.items():
        if off is None:
            out[name] = None
            continue
        val = _read(data, off, fmt)
        if val is None and warn:
            print(f"  [pending] {name}: offset {off} out of bounds")
        out[name] = val
    return out


def _read_dict(data, offset_dict):
    """Read every (offset, fmt) entry in ``offset_dict`` and return a
    new dict of name → value. None is returned for fields that fall
    past end-of-data."""
    out = OrderedDict()
    for name, (off, fmt) in offset_dict.items():
        out[name] = _read(data, off, fmt)
    return out


def parse_mass_table(data, nominal_masses=None, n_slots_search=20):
    """
    Parse the per-detector mass table.

    The Cameca .im file stores the mass table as a dense list of rows
    (one per available mass position in the multicollector geometry, not
    one per actually-installed detector). To find the row corresponding
    to each installed detector we match the row's mass against the
    image's known nominal-mass list.

    Parameters
    ----------
    data : bytes
        The raw header bytes.
    nominal_masses : list of float or None
        The nominal masses (amu) of the channels actually present in
        the image — pass ``MimsImage.nom_masses``. If None, we return
        rows from row indices 0 through n_slots_search-1, regardless of
        whether they correspond to populated channels (useful for
        first-time inspection of unfamiliar files).
    n_slots_search : int, default 20
        How many rows to scan when matching by nominal mass, or how many
        rows to return when ``nominal_masses`` is None.

    Returns
    -------
    OrderedDict mapping channel index (0-based, matching pymims.masses
    order) → row dict. The dict has the standard fields ``mass_amu``,
    ``radius_mm``, ``neg_plate_dac``, ``pos_plate_dac``, ``exit_slit_um``,
    plus a ``row_index`` field recording which physical slot the data
    came from.

    If ``nominal_masses`` is None, returns rows by physical row index
    instead.
    """
    if nominal_masses is None:
        # Return rows in physical order
        out = OrderedDict()
        for row_idx in range(n_slots_search):
            row_start = DETECTOR_BASE + row_idx * DETECTOR_STRIDE
            if row_start + DETECTOR_STRIDE > len(data):
                break
            row = OrderedDict({'row_index': row_idx})
            for name, (off_in_row, fmt) in DETECTOR_FIELDS.items():
                row[name] = _read(data, row_start + off_in_row, fmt)
            out[row_idx] = row
        return out

    # Match each known nominal mass to its row in the table
    out = OrderedDict()
    # Cache all row masses so we can match
    row_masses = []
    for row_idx in range(n_slots_search):
        row_start = DETECTOR_BASE + row_idx * DETECTOR_STRIDE
        if row_start + 8 > len(data):
            break
        row_masses.append(_read(data, row_start + DETECTOR_FIELDS['mass_amu'][0],
                                 DETECTOR_FIELDS['mass_amu'][1]))

    for ch_idx, nom_mass in enumerate(nominal_masses):
        if nom_mass is None or nom_mass == 0:
            # SE channels have nominal mass 0; no analytical row to match.
            continue
        # Find best-matching row
        best_row = None
        best_diff = float('inf')
        for row_idx, row_mass in enumerate(row_masses):
            if row_mass is None or row_mass == 0:
                continue
            diff = abs(row_mass - nom_mass)
            if diff < best_diff and diff < 0.05:   # tight tolerance
                best_diff = diff
                best_row = row_idx
        if best_row is None:
            continue
        # Read the full row
        row_start = DETECTOR_BASE + best_row * DETECTOR_STRIDE
        row = OrderedDict({'row_index': best_row})
        for name, (off_in_row, fmt) in DETECTOR_FIELDS.items():
            row[name] = _read(data, row_start + off_in_row, fmt)
        out[ch_idx] = row
    return out


def parse_detector_params(data, populated_slots=(1, 2, 3, 6, 7)):
    """
    Parse the detector parameters table (HV, threshold).

    Parameters
    ----------
    data : bytes
        Raw header bytes.
    populated_slots : tuple of int

    Returns
    -------
    dict {slot → {'hv_v': int, 'threshold_mv': float}}
    """
    out = OrderedDict()
    for slot in populated_slots:
        hv = _read(data, DETECTOR_PARAMS_BASE_HV + (slot - 1) * 4, '<i')
        thr = _read(data, DETECTOR_PARAMS_BASE_THR + (slot - 1) * 8, '<d')
        out[slot] = {'hv_v': hv, 'threshold_mv': thr}
    return out


def parse_selectors(data):
    """
    Parse the diaphragm and slit selector positions plus their
    available-position tables.

    Returns
    -------
    OrderedDict with the position values plus, for each selector, a
    list of available physical sizes and the size at the chosen
    position.
    """
    out = OrderedDict()
    # Selected positions
    for name, (off, fmt) in OFFSETS_SELECTORS.items():
        out[name] = _read(data, off, fmt)
    # Available-position arrays
    for name, (off, n, fmt) in OFFSETS_SELECTOR_TABLES.items():
        size = struct.calcsize(fmt)
        vals = []
        for i in range(n):
            v = _read(data, off + i * size, fmt)
            if v is not None:
                vals.append(v)
        out[name] = vals
    # Convenience: actual size at the chosen position
    if out.get('d1_position') is not None and out.get('d1_diameters_um'):
        idx = out['d1_position'] - 1   # positions are 1-indexed
        if 0 <= idx < len(out['d1_diameters_um']):
            out['d1_diameter_um']  = out['d1_diameters_um'][idx]
    if out.get('es_position') is not None and out.get('es_widths_um'):
        idx = out['es_position'] - 1
        if 0 <= idx < len(out['es_widths_um']):
            out['es_width_um']  = out['es_widths_um'][idx]
            out['es_height_um'] = out['es_heights_um'][idx]
    if out.get('as_position') is not None and out.get('as_widths_um'):
        idx = out['as_position'] - 1
        if 0 <= idx < len(out['as_widths_um']):
            out['as_width_um']  = out['as_widths_um'][idx]
            out['as_height_um'] = out['as_heights_um'][idx]
    return out


def detect_header_layout(header_bytes):
    """
    Best-effort detection of whether the .im header layout matches the
    one this parser was validated against (a 2026 NS50L acquisition).

    Strategy: read a few values at key offsets and check whether they
    fall in physically plausible ranges. If too many fields look wrong,
    the file probably comes from a different instrument generation or
    firmware version, and our offsets won't apply.

    Returns
    -------
    dict with:
        'layout_supported' : bool — True if the file looks compatible
        'reasons'          : list of strings, one per detected issue
        'sample_values'    : the values that were tested

    A warning string is returned in the 'reasons' list when issues are
    found; an empty list means full compatibility was detected.
    """
    reasons = []
    sample = {}

    # Cs HV — typically 5000-10000 V on NS50/50L
    cs_hv = _read(header_bytes, 7600, '<i')
    sample['cs_hv_at_7600'] = cs_hv
    if cs_hv is None or not (1000 <= cs_hv <= 15000):
        reasons.append(
            f"Cs HV at offset 7600 reads {cs_hv}; expected 1000-15000 V."
        )

    # Ionizer current — typically 1-3 mA
    ionizer = _read(header_bytes, 24084, '<d')
    sample['ionizer_at_24084'] = ionizer
    if ionizer is None or not (0.001 <= ionizer <= 10.0):
        reasons.append(
            f"Ionizer current at offset 24084 reads {ionizer}; "
            f"expected 0.001-10 mA."
        )

    # Mass table row 0 (det #1) — should be a real mass value 1-300 amu
    m0 = _read(header_bytes, DETECTOR_BASE + 32, '<d')
    sample['mass0_at_4284'] = m0
    if m0 is None or not (1.0 <= m0 <= 300.0):
        reasons.append(
            f"Mass-table row 0 at offset {DETECTOR_BASE+32} reads {m0}; "
            f"expected a mass value 1-300 amu."
        )

    return {
        'layout_supported': len(reasons) == 0,
        'reasons': reasons,
        'sample_values': sample,
    }


def parse_full_metadata(header_bytes, nominal_masses=None,
                         populated_slots=(1, 2, 3, 6, 7)):
    """
    Top-level entry point: read every documented field from the .im header.

    Parameters
    ----------
    header_bytes : bytes
        The .im file header. Pass the first ~30 KB of the file (or the
        full file — slicing is handled internally).
    nominal_masses : list of float or None
        Nominal masses (amu) of the actual channels present in the image,
        used to match mass-table rows to channel indices. Pass
        ``MimsImage.nom_masses``. If None, the mass table is returned
        keyed by physical row index.
    populated_slots : tuple of int
        Detector slot numbers physically installed. Used for the
        detector-parameters table only (HV, threshold), which IS indexed
        by slot number. Default ``(1, 2, 3, 6, 7)`` matches NS50L geometry.

    Returns
    -------
    OrderedDict with sub-dicts:
        'basic'     : duration, cycles_planned, stage XYZ
        'bfield'    : magnet trim DACs
        'primary'   : Cs source + primary lens parameters
        'secondary' : sample HV + secondary transfer optics
        'mass_table': {ch_idx → row} when nominal_masses given,
                      {row_idx → row} when not
        'detectors' : {slot → {HV, threshold}}
    """
    out = OrderedDict()
    layout = detect_header_layout(header_bytes)
    out['layout'] = layout
    out['basic']      = _read_dict(header_bytes, OFFSETS_BASIC)
    out['bfield']     = _read_dict(header_bytes, OFFSETS_BFIELD)
    out['primary']    = _read_dict(header_bytes, OFFSETS_PRIMARY)
    out['secondary']  = _read_dict(header_bytes, OFFSETS_SECONDARY)
    out['selectors']  = parse_selectors(header_bytes)
    out['mass_table'] = parse_mass_table(header_bytes,
                                          nominal_masses=nominal_masses)
    out['detectors']  = parse_detector_params(header_bytes,
                                                populated_slots=populated_slots)
    out['pending']    = _read_pending(header_bytes, OFFSETS_PENDING, warn=False)
    return out


# ── Pretty printing ─────────────────────────────────────────────────────────

def format_metadata(meta, masses=None):
    """
    Return a nicely-formatted multi-line string of ``meta``.

    Parameters
    ----------
    meta : dict
        Output of parse_full_metadata().
    masses : list[str] or None
        Optional list of mass labels, one per populated slot, used to
        annotate the mass-table and detector-table rows. If provided,
        its length must match the number of slots in ``meta['mass_table']``.

    Returns
    -------
    str
    """
    lines = []
    lines.append('=' * 70)
    lines.append('NanoSIMS .im file — full metadata')
    lines.append('=' * 70)

    # Layout warning if the header doesn't match our reference layout
    layout = meta.get('layout')
    if layout and not layout.get('layout_supported', True):
        lines.append('')
        lines.append('⚠ HEADER LAYOUT WARNING')
        lines.append('-' * 70)
        lines.append('This file\'s header does not match the layout that')
        lines.append('PyMIMS was validated against (2026 NS50L acquisition).')
        lines.append('Many of the analytical parameters below may be incorrect.')
        lines.append('')
        lines.append('Issues detected:')
        for r in layout.get('reasons', []):
            lines.append(f'  • {r}')
        lines.append('')
        lines.append('Fields known to work across header versions:')
        lines.append('  • Acquisition basics (duration, cycles, stage XYZ)')
        lines.append('')
        lines.append('Other fields are reported below for diagnostic purposes')
        lines.append('but should not be trusted on this file. To validate')
        lines.append('offsets for your instrument, use scripts/find_offset.py')
        lines.append('with known values from the Cameca tool or OpenMIMS.')

    # Basic
    b = meta.get('basic', {})
    lines.append('\n## Acquisition')
    if b.get('duration_s') is not None:
        lines.append(f"  Total acquisition time : {b['duration_s']} s")
    if b.get('cycles_planned') is not None:
        lines.append(f"  Cycles (planned)       : {b['cycles_planned']}")
    if b.get('dwell_us') is not None:
        lines.append(f"  Pixel dwell time       : {b['dwell_us']/1000:.3f} ms"
                     f"  ({b['dwell_us']} µs)")
    if b.get('stage_x_um') is not None:
        lines.append(f"  Sample stage X / Y / Z : "
                     f"{b['stage_x_um']} / "
                     f"{b['stage_y_um']} / "
                     f"{b['stage_z_um']}  µm")

    # Bfield / magnet trims
    bf = meta.get('bfield', {})
    lines.append('\n## Bfield and magnet trim DACs')
    for name, val in bf.items():
        if val is not None:
            lines.append(f"  {name:18s}: {val}")

    # Primary column
    p = meta.get('primary', {})
    lines.append('\n## Primary column (Cs source + lenses)')
    if p.get('cs_hv_v') is not None:
        lines.append(f"  Cs HV              : {p['cs_hv_v']} V")
    if p.get('ionizer_mA') is not None:
        lines.append(f"  Ionizer current    : {p['ionizer_mA']:.3f} mA")
    if p.get('reservoir_mA') is not None:
        lines.append(f"  Reservoir current  : {p['reservoir_mA']:.3f} mA")
    if p.get('l1_v') is not None:
        lines.append(f"  L1                 : {p['l1_v']} V")
    if p.get('oct45_v') is not None:
        lines.append(f"  Oct45              : {p['oct45_v']:.4f} V "
                     f"({p.get('oct45_dac')} DAC)")
    if p.get('oct90_v') is not None:
        lines.append(f"  Oct90              : {p['oct90_v']:.4f} V "
                     f"({p.get('oct90_dac')} DAC)")
    if p.get('e0p_v') is not None:
        lines.append(f"  E0P (primary focal): {p['e0p_v']:.2f} V")
    if p.get('be_dac') is not None:
        lines.append(f"  Be DAC             : {p['be_dac']}")

    # Secondary
    s = meta.get('secondary', {})
    lines.append('\n## Secondary column (sample HV + transfer optics)')
    for name, val in s.items():
        if val is not None:
            lines.append(f"  {name:18s}: {val}")

    # Selectors — diaphragms and slits
    sel = meta.get('selectors', {})
    if sel:
        lines.append('\n## Diaphragms and slits')
        if sel.get('d1_diameter_um') is not None:
            lines.append(f"  D1 (primary diaphragm) : pos {sel['d1_position']} → "
                         f"{sel['d1_diameter_um']} µm  "
                         f"(available: {sel['d1_diameters_um']} µm)")
        if sel.get('es_width_um') is not None:
            lines.append(f"  ES (entrance slit)     : pos {sel['es_position']} → "
                         f"{sel['es_width_um']} × {sel['es_height_um']} µm  "
                         f"(W: {sel['es_widths_um']}, "
                         f"H: {sel['es_heights_um']})")
        if sel.get('as_width_um') is not None:
            lines.append(f"  AS (aperture slit)     : pos {sel['as_position']} → "
                         f"{sel['as_width_um']} × {sel['as_height_um']} µm  "
                         f"(W: {sel['as_widths_um']}, "
                         f"H: {sel['as_heights_um']})")

    # Mass table — keys are channel indices (0-based), or row indices if
    # nominal_masses wasn't given
    mt = meta.get('mass_table', {})
    if mt:
        lines.append('\n## Mass table')
        header_row = (f"  {'symbol':<10}  {'mass':>9}  "
                      f"{'radius':>10}  {'neg':>6}  {'pos':>6}  {'exit':>8}")
        lines.append(header_row)
        lines.append('  ' + '-' * (len(header_row) - 2))
        for key, row in mt.items():
            sym = '-'
            if masses is not None and isinstance(key, int) and key < len(masses):
                sym = masses[key]
            mass = row.get('mass_amu')
            radius = row.get('radius_mm')
            neg = row.get('neg_plate_dac')
            pos = row.get('pos_plate_dac')
            exitw = row.get('exit_slit_um')
            if mass is not None:
                lines.append(
                    f"  {sym:<10}  "
                    f"{mass:>9.3f}  {radius:>10.3f}  {neg:>6}  {pos:>6}  "
                    f"{exitw:>8.1f}"
                )
            else:
                lines.append(f"  {sym:<10}  (missing)")

    # Detector params
    dp = meta.get('detectors', {})
    if dp:
        lines.append('\n## Detector parameters')
        header_row = (f"  {'slot':>4}  {'symbol':<10}  {'HV':>5}  "
                      f"{'threshold':>10}")
        lines.append(header_row)
        lines.append('  ' + '-' * (len(header_row) - 2))
        for i, (slot, row) in enumerate(dp.items()):
            sym = '-'
            if masses is not None and i < len(masses):
                sym = masses[i]
            hv = row.get('hv_v')
            thr = row.get('threshold_mv')
            if hv is not None:
                lines.append(f"  #{slot:<3}  {sym:<10}  "
                             f"{hv:>5} V  {thr:>8.4f} mV")
            else:
                lines.append(f"  #{slot:<3}  (missing)")

    # Pending fields — shown as a clear "awaiting validation" footer
    pending = meta.get('pending', {})
    if pending:
        any_pending = any(p[0] is None for p in OFFSETS_PENDING.values())
        if any_pending:
            lines.append('\n## Pending fields (offsets not yet validated)')
            for name, (off, fmt, note) in OFFSETS_PENDING.items():
                if off is None:
                    lines.append(f"  {name:18s}: not yet located — {note}")
                else:
                    val = pending.get(name)
                    lines.append(f"  {name:18s}: {val} (offset {off}, "
                                 f"unverified) — {note}")

    return '\n'.join(lines)

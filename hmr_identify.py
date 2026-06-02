"""Species identification for NanoSIMS HMR peaks.

Ported from nanosims_picker_v2.html so the picker and the PyMIMS HMR tab share
one isotope table and one candidate-enumeration model. Given the elements
expected in the sample, a nominal mass and polarity, ``enumerate_candidates``
returns plausible molecular ions with their exact m/z and a plausibility score
(isotope-abundance product x 0.3^(n-1) size penalty). ``match_by_separation``
labels deconvolved peaks against the *separations* between candidates anchored
on a central peak — no absolute mass calibration needed (the HMR amu axis is a
nominal instrument scale).
"""
from __future__ import annotations
from dataclasses import dataclass, field

ME = 0.00054858          # electron mass (u)
MAX_H = 2                # hydrogen cap per cluster (MH-, M(OH)2- common; MH3- not)
SIZE_PENALTY_PER_EXTRA_ATOM = 0.3

ELEMENTS = [{'sym': 'H', 'z': 1, 'row': 1, 'col': 1, 'isotopes': [[1, 1.00782503, 0.99985], [2, 2.01410178, 0.00015]]}, {'sym': 'He', 'z': 2, 'row': 1, 'col': 18, 'isotopes': [[4, 4.00260325, 1.0]]}, {'sym': 'Li', 'z': 3, 'row': 2, 'col': 1, 'isotopes': [[6, 6.01512279, 0.0759], [7, 7.01600344, 0.9241]]}, {'sym': 'Be', 'z': 4, 'row': 2, 'col': 2, 'isotopes': [[9, 9.01218307, 1.0]]}, {'sym': 'B', 'z': 5, 'row': 2, 'col': 13, 'isotopes': [[10, 10.01293695, 0.199], [11, 11.00930536, 0.801]]}, {'sym': 'C', 'z': 6, 'row': 2, 'col': 14, 'isotopes': [[12, 12.0, 0.9893], [13, 13.00335484, 0.0107]]}, {'sym': 'N', 'z': 7, 'row': 2, 'col': 15, 'isotopes': [[14, 14.00307401, 0.99636], [15, 15.0001089, 0.00364]]}, {'sym': 'O', 'z': 8, 'row': 2, 'col': 16, 'isotopes': [[16, 15.99491462, 0.99757], [17, 16.9991317, 0.00038], [18, 17.99915961, 0.00205]]}, {'sym': 'F', 'z': 9, 'row': 2, 'col': 17, 'isotopes': [[19, 18.99840316, 1.0]]}, {'sym': 'Ne', 'z': 10, 'row': 2, 'col': 18, 'isotopes': [[20, 19.99244018, 0.9048], [22, 21.99138511, 0.0925]]}, {'sym': 'Na', 'z': 11, 'row': 3, 'col': 1, 'isotopes': [[23, 22.98976928, 1.0]]}, {'sym': 'Mg', 'z': 12, 'row': 3, 'col': 2, 'isotopes': [[24, 23.9850417, 0.7899], [25, 24.98583692, 0.1], [26, 25.98259297, 0.1101]]}, {'sym': 'Al', 'z': 13, 'row': 3, 'col': 13, 'isotopes': [[27, 26.98153853, 1.0]]}, {'sym': 'Si', 'z': 14, 'row': 3, 'col': 14, 'isotopes': [[28, 27.97692653, 0.92223], [29, 28.97649466, 0.04685], [30, 29.97377017, 0.03092]]}, {'sym': 'P', 'z': 15, 'row': 3, 'col': 15, 'isotopes': [[31, 30.97376199, 1.0]]}, {'sym': 'S', 'z': 16, 'row': 3, 'col': 16, 'isotopes': [[32, 31.97207117, 0.9499], [33, 32.97145876, 0.0075], [34, 33.967867, 0.0425]]}, {'sym': 'Cl', 'z': 17, 'row': 3, 'col': 17, 'isotopes': [[35, 34.96885268, 0.7576], [37, 36.96590258, 0.2424]]}, {'sym': 'Ar', 'z': 18, 'row': 3, 'col': 18, 'isotopes': [[36, 35.96754511, 0.00337], [40, 39.96238312, 0.996]]}, {'sym': 'K', 'z': 19, 'row': 4, 'col': 1, 'isotopes': [[39, 38.96370648, 0.93258], [41, 40.96182525, 0.0673]]}, {'sym': 'Ca', 'z': 20, 'row': 4, 'col': 2, 'isotopes': [[40, 39.96259098, 0.96941], [42, 41.95861783, 0.00647], [44, 43.95548156, 0.02086]]}, {'sym': 'Sc', 'z': 21, 'row': 4, 'col': 3, 'isotopes': [[45, 44.95590828, 1.0]]}, {'sym': 'Ti', 'z': 22, 'row': 4, 'col': 4, 'isotopes': [[46, 45.95262772, 0.0825], [47, 46.95175879, 0.0744], [48, 47.94794198, 0.7372], [49, 48.94786568, 0.0541], [50, 49.94478689, 0.0518]]}, {'sym': 'V', 'z': 23, 'row': 4, 'col': 5, 'isotopes': [[50, 49.94715602, 0.0025], [51, 50.94395704, 0.9975]]}, {'sym': 'Cr', 'z': 24, 'row': 4, 'col': 6, 'isotopes': [[50, 49.94604184, 0.04345], [52, 51.94050623, 0.83789], [53, 52.94064815, 0.09501], [54, 53.93887916, 0.02365]]}, {'sym': 'Mn', 'z': 25, 'row': 4, 'col': 7, 'isotopes': [[55, 54.93804451, 1.0]]}, {'sym': 'Fe', 'z': 26, 'row': 4, 'col': 8, 'isotopes': [[54, 53.93960899, 0.05845], [56, 55.93493633, 0.91754], [57, 56.93539283, 0.02119], [58, 57.93327443, 0.00282]]}, {'sym': 'Co', 'z': 27, 'row': 4, 'col': 9, 'isotopes': [[59, 58.93319429, 1.0]]}, {'sym': 'Ni', 'z': 28, 'row': 4, 'col': 10, 'isotopes': [[58, 57.93534241, 0.68077], [60, 59.93078588, 0.26223], [61, 60.93105557, 0.0114], [62, 61.92834537, 0.03634], [64, 63.92796682, 0.00926]]}, {'sym': 'Cu', 'z': 29, 'row': 4, 'col': 11, 'isotopes': [[63, 62.92959772, 0.6915], [65, 64.9277897, 0.3085]]}, {'sym': 'Zn', 'z': 30, 'row': 4, 'col': 12, 'isotopes': [[64, 63.92914201, 0.4917], [66, 65.92603381, 0.2773], [67, 66.92712775, 0.0404], [68, 67.92484455, 0.1845], [70, 69.9253192, 0.0061]]}, {'sym': 'Ga', 'z': 31, 'row': 4, 'col': 13, 'isotopes': [[69, 68.92557353, 0.60108], [71, 70.92470258, 0.39892]]}, {'sym': 'Ge', 'z': 32, 'row': 4, 'col': 14, 'isotopes': [[70, 69.92424875, 0.2057], [72, 71.92207583, 0.2745], [73, 72.92345896, 0.0775], [74, 73.92117776, 0.365], [76, 75.92140273, 0.0773]]}, {'sym': 'As', 'z': 33, 'row': 4, 'col': 15, 'isotopes': [[75, 74.92159457, 1.0]]}, {'sym': 'Se', 'z': 34, 'row': 4, 'col': 16, 'isotopes': [[74, 73.92247593, 0.0089], [76, 75.9192137, 0.0937], [77, 76.91991415, 0.0763], [78, 77.91730928, 0.2377], [80, 79.9165218, 0.4961], [82, 81.91669952, 0.0873]]}, {'sym': 'Br', 'z': 35, 'row': 4, 'col': 17, 'isotopes': [[79, 78.91833715, 0.5069], [81, 80.91628962, 0.4931]]}, {'sym': 'Kr', 'z': 36, 'row': 4, 'col': 18, 'isotopes': [[84, 83.91149773, 0.57]]}, {'sym': 'Rb', 'z': 37, 'row': 5, 'col': 1, 'isotopes': [[85, 84.91178974, 0.7217], [87, 86.90918054, 0.2783]]}, {'sym': 'Sr', 'z': 38, 'row': 5, 'col': 2, 'isotopes': [[88, 87.90561225, 0.8258]]}, {'sym': 'Y', 'z': 39, 'row': 5, 'col': 3, 'isotopes': [[89, 88.9058403, 1.0]]}, {'sym': 'Zr', 'z': 40, 'row': 5, 'col': 4, 'isotopes': [[90, 89.90469877, 0.5145], [91, 90.90564022, 0.1122], [92, 91.90503533, 0.1715], [94, 93.90631252, 0.1738], [96, 95.90827757, 0.028]]}, {'sym': 'Nb', 'z': 41, 'row': 5, 'col': 5, 'isotopes': [[93, 92.90637303, 1.0]]}, {'sym': 'Mo', 'z': 42, 'row': 5, 'col': 6, 'isotopes': [[92, 91.90680796, 0.1453], [94, 93.9050849, 0.0915], [95, 94.90583877, 0.1584], [96, 95.90467612, 0.1667], [97, 96.90601812, 0.096], [98, 97.90540482, 0.2439], [100, 99.90747718, 0.0982]]}, {'sym': 'Tc', 'z': 43, 'row': 5, 'col': 7, 'isotopes': [[98, 97.9072124, 1.0]]}, {'sym': 'Ru', 'z': 44, 'row': 5, 'col': 8, 'isotopes': [[102, 101.9043441, 0.3155]]}, {'sym': 'Rh', 'z': 45, 'row': 5, 'col': 9, 'isotopes': [[103, 102.905498, 1.0]]}, {'sym': 'Pd', 'z': 46, 'row': 5, 'col': 10, 'isotopes': [[106, 105.9034804, 0.2733]]}, {'sym': 'Ag', 'z': 47, 'row': 5, 'col': 11, 'isotopes': [[107, 106.9050916, 0.51839], [109, 108.9047553, 0.48161]]}, {'sym': 'Cd', 'z': 48, 'row': 5, 'col': 12, 'isotopes': [[114, 113.9033585, 0.2873]]}, {'sym': 'In', 'z': 49, 'row': 5, 'col': 13, 'isotopes': [[115, 114.9038787, 0.9572]]}, {'sym': 'Sn', 'z': 50, 'row': 5, 'col': 14, 'isotopes': [[120, 119.9022016, 0.3258]]}, {'sym': 'Sb', 'z': 51, 'row': 5, 'col': 15, 'isotopes': [[121, 120.903812, 0.5721], [123, 122.9042132, 0.4279]]}, {'sym': 'Te', 'z': 52, 'row': 5, 'col': 16, 'isotopes': [[130, 129.9062227, 0.3408]]}, {'sym': 'I', 'z': 53, 'row': 5, 'col': 17, 'isotopes': [[127, 126.9044719, 1.0]]}, {'sym': 'Xe', 'z': 54, 'row': 5, 'col': 18, 'isotopes': [[132, 131.904155, 0.2689]]}, {'sym': 'Cs', 'z': 55, 'row': 6, 'col': 1, 'isotopes': [[133, 132.9054519, 1.0]]}, {'sym': 'Ba', 'z': 56, 'row': 6, 'col': 2, 'isotopes': [[138, 137.9052472, 0.717]]}, {'sym': 'La', 'z': 57, 'row': 9, 'col': 3, 'isotopes': [[139, 138.9063587, 0.9991]]}, {'sym': 'Ce', 'z': 58, 'row': 9, 'col': 4, 'isotopes': [[140, 139.9054387, 0.8845]]}, {'sym': 'Pr', 'z': 59, 'row': 9, 'col': 5, 'isotopes': [[141, 140.9076528, 1.0]]}, {'sym': 'Nd', 'z': 60, 'row': 9, 'col': 6, 'isotopes': [[142, 141.9077233, 0.2715]]}, {'sym': 'Pm', 'z': 61, 'row': 9, 'col': 7, 'isotopes': [[145, 144.9127559, 1.0]]}, {'sym': 'Sm', 'z': 62, 'row': 9, 'col': 8, 'isotopes': [[152, 151.9197324, 0.2675]]}, {'sym': 'Eu', 'z': 63, 'row': 9, 'col': 9, 'isotopes': [[153, 152.9212303, 0.5219]]}, {'sym': 'Gd', 'z': 64, 'row': 9, 'col': 10, 'isotopes': [[158, 157.9241039, 0.2484]]}, {'sym': 'Tb', 'z': 65, 'row': 9, 'col': 11, 'isotopes': [[159, 158.9253468, 1.0]]}, {'sym': 'Dy', 'z': 66, 'row': 9, 'col': 12, 'isotopes': [[164, 163.9291748, 0.2826]]}, {'sym': 'Ho', 'z': 67, 'row': 9, 'col': 13, 'isotopes': [[165, 164.9303221, 1.0]]}, {'sym': 'Er', 'z': 68, 'row': 9, 'col': 14, 'isotopes': [[166, 165.9302931, 0.3361]]}, {'sym': 'Tm', 'z': 69, 'row': 9, 'col': 15, 'isotopes': [[169, 168.9342133, 1.0]]}, {'sym': 'Yb', 'z': 70, 'row': 9, 'col': 16, 'isotopes': [[174, 173.9388621, 0.3183]]}, {'sym': 'Lu', 'z': 71, 'row': 9, 'col': 17, 'isotopes': [[175, 174.9407718, 0.9741]]}, {'sym': 'Hf', 'z': 72, 'row': 6, 'col': 4, 'isotopes': [[180, 179.94655, 0.3508]]}, {'sym': 'Ta', 'z': 73, 'row': 6, 'col': 5, 'isotopes': [[181, 180.9479958, 0.9999]]}, {'sym': 'W', 'z': 74, 'row': 6, 'col': 6, 'isotopes': [[184, 183.9509312, 0.3064]]}, {'sym': 'Re', 'z': 75, 'row': 6, 'col': 7, 'isotopes': [[187, 186.9557501, 0.626]]}, {'sym': 'Os', 'z': 76, 'row': 6, 'col': 8, 'isotopes': [[192, 191.961477, 0.4078]]}, {'sym': 'Ir', 'z': 77, 'row': 6, 'col': 9, 'isotopes': [[193, 192.9629238, 0.6287]]}, {'sym': 'Pt', 'z': 78, 'row': 6, 'col': 10, 'isotopes': [[195, 194.9647911, 0.3378]]}, {'sym': 'Au', 'z': 79, 'row': 6, 'col': 11, 'isotopes': [[197, 196.9665687, 1.0]]}, {'sym': 'Hg', 'z': 80, 'row': 6, 'col': 12, 'isotopes': [[202, 201.9706434, 0.2986]]}, {'sym': 'Tl', 'z': 81, 'row': 6, 'col': 13, 'isotopes': [[205, 204.9744275, 0.7048]]}, {'sym': 'Pb', 'z': 82, 'row': 6, 'col': 14, 'isotopes': [[208, 207.9766525, 0.524]]}, {'sym': 'Bi', 'z': 83, 'row': 6, 'col': 15, 'isotopes': [[209, 208.9803991, 1.0]]}, {'sym': 'U', 'z': 92, 'row': 10, 'col': 6, 'isotopes': [[238, 238.0507882, 0.9927]]}]

_BY_SYM = {e["sym"]: e for e in ELEMENTS}


@dataclass
class Candidate:
    formula: str
    mz: float
    total_atoms: int
    charge: int
    iso_abund: float
    size_penalty: float
    plausibility: float


def enumerate_candidates(elements, nominal_mass, max_atoms=4, polarity=-1,
                         mz_tol=0.05, nominal_tol=0.6):
    """Return plausible ions near ``nominal_mass`` built from ``elements``.

    polarity: -1 anion (m/z = neutral + me), +1 cation (m/z = neutral - me,
    charge 2 allowed). Faithful port of the picker's enumerate(): H capped at
    MAX_H, each isotope capped at min(remaining, 4). ``mz_tol`` sets the accept
    window around the nominal mass (picker default 0.05; widen for HMR).
    """
    atom_list = []
    for sym in elements:
        el = _BY_SYM.get(sym)
        if not el:
            continue
        for a, mass, abund in el["isotopes"]:
            atom_list.append({"sym": sym, "a": a, "mass": mass, "abund": abund})
    n_types = len(atom_list)
    counts = [0] * n_types
    charges = [1] if polarity == -1 else [1, 2]
    results = []

    def recurse(idx, total, hcount, nominal_sum, mass_sum):
        if total > max_atoms:
            return
        if idx == n_types:
            if total == 0:
                return
            for z in charges:
                mz = (mass_sum - polarity * z * ME) / z
                if abs(nominal_sum / z - nominal_mass) < nominal_tol and \
                   abs(mz - nominal_mass) < mz_tol:
                    parts = []
                    for i, at in enumerate(atom_list):
                        if counts[i] > 0:
                            suff = counts[i] if counts[i] > 1 else ""
                            parts.append(f"{at['a']}{at['sym']}{suff}")
                    sign = "-" if polarity == -1 else "+"
                    zsuf = z if z > 1 else ""
                    iso_abund = 1.0
                    for i, at in enumerate(atom_list):
                        if counts[i] > 0:
                            iso_abund *= at["abund"] ** counts[i]
                    size_pen = SIZE_PENALTY_PER_EXTRA_ATOM ** max(0, total - 1)
                    results.append(Candidate(
                        formula=f"{''.join(parts)}{zsuf}{sign}",
                        mz=mz, total_atoms=total, charge=z,
                        iso_abund=iso_abund, size_penalty=size_pen,
                        plausibility=iso_abund * size_pen,
                    ))
            return
        at = atom_list[idx]
        is_h = at["sym"] == "H"
        cap = min(MAX_H - hcount, max_atoms - total) if is_h else (max_atoms - total)
        max_of_this = min(cap, 4)
        for n in range(max_of_this + 1):
            counts[idx] = n
            recurse(idx + 1, total + n, hcount + (n if is_h else 0),
                    nominal_sum + n * at["a"], mass_sum + n * at["mass"])
        counts[idx] = 0

    recurse(0, 0, 0, 0, 0)
    seen = {}
    for r in results:
        if r.formula not in seen or seen[r.formula].plausibility < r.plausibility:
            seen[r.formula] = r
    out = list(seen.values())
    out.sort(key=lambda c: -c.plausibility)
    return out


@dataclass
class SeparationMatch:
    anchor: Candidate
    labels: list = field(default_factory=list)   # one dict per observed peak
    rms_mau: float = 0.0


def match_by_separation(observed_mz, expected_elements, polarity=-1,
                        max_atoms=4, top_anchors=6):
    """Label deconvolved peaks by relative separation (no absolute calib).

    Enumerate candidates around the median observed nominal mass, then for each
    plausible candidate taken as the *anchor* on the central observed peak,
    shift the candidate set onto the data and score how well every observed
    peak lines up with a candidate exact-mass position. Returns the best
    (lowest-RMS) anchoring.
    """
    observed = sorted(float(x) for x in observed_mz)
    if not observed:
        return None
    nominal = round(sum(observed) / len(observed))
    cands = enumerate_candidates(expected_elements, nominal, max_atoms=max_atoms,
                                 polarity=polarity, mz_tol=0.6)
    if not cands:
        return None
    anchor_obs = observed[len(observed) // 2]
    best = None
    for anchor in cands[:max(top_anchors, 1)]:
        shift = anchor_obs - anchor.mz
        labels, sq = [], 0.0
        for o in observed:
            target = o - shift
            nearest = min(cands, key=lambda c: abs(c.mz - target))
            err = (nearest.mz - target) * 1000.0   # mau
            sq += err * err
            labels.append({"observed_mz": o, "candidate": nearest,
                           "error_mau": err})
        rms = (sq / len(observed)) ** 0.5
        if best is None or rms < best.rms_mau:
            best = SeparationMatch(anchor=anchor, labels=labels, rms_mau=rms)
    return best

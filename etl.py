from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "circuit.json")

EXCITATORY = {"ACH", "ACETYLCHOLINE", "GLUTAMATE_ACH", "DA", "OA", "5HT", "SEROTONIN"}
INHIBITORY = {"GABA", "GLUTAMATE", "GLU", "GABA_GLU", "HISTAMINE"}

CORE_GROUPS = {
    "loom_lc4": ["LC4"],
    "loom_lplc2": ["LPLC2"],
    "escape_gf": ["DNp01"],
    "steer_left": ["DNa01"],
    "steer_right": ["DNa02"],
    "forward": ["DNp09"],
    "groom": ["DNg11"],
    "backward": ["MDN"],
    "wing_a": ["DNp02"],
    "wing_b": ["DNp04"],
    "wing_c": ["DNp11"],
}

PARTNERS_PER_GROUP = 30
ASCENDING_LIMIT = 40
WIND_LIMIT = 20
GAIN_E = 3.2
GAIN_I = 1.8
ALPHA = 1.0
SPECIFICITY = 0.20


def _open(name):
    return gzip.open(os.path.join(RAW, name), "rt", encoding="utf-8", newline="")


def _norm(k):
    return k.strip().lower().replace(" ", "_")


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0


def load_cell_types():
    path = os.path.join(RAW, "consolidated_cell_types.csv.gz")
    if not os.path.exists(path):
        return {}
    types = {}
    with _open("consolidated_cell_types.csv.gz") as f:
        rd = csv.reader(f)
        header = [_norm(h) for h in next(rd)]
        i_id = header.index("root_id")
        i_ct = None
        for cand in ("primary_type", "cell_type", "type", "celltype", "name"):
            if cand in header:
                i_ct = header.index(cand)
                break
        if i_ct is None:
            print("  ! 找不到类型列，表头:", header)
            return {}
        for row in rd:
            if len(row) <= max(i_id, i_ct):
                continue
            ct = row[i_ct].strip()
            if ct:
                types[row[i_id]] = ct
    return types


def load_classification():
    meta = {}
    with _open("classification.csv.gz") as f:
        rd = csv.DictReader(f)
        for row in rd:
            meta[row["root_id"]] = {
                "super_class": row.get("super_class", ""),
                "class": row.get("class", ""),
                "side": row.get("side", ""),
            }
    return meta


def load_positions():
    pos = {}
    with _open("coordinates.csv.gz") as f:
        rd = csv.DictReader(f)
        for row in rd:
            s = row["position"].strip("[]")
            parts = s.split()
            if len(parts) == 3:
                try:
                    pos[row["root_id"]] = [int(p) for p in parts]
                except ValueError:
                    pass
    return pos


def iter_connections():
    with _open("connections.csv.gz") as f:
        rd = csv.reader(f)
        header = [_norm(h) for h in next(rd)]
        try:
            i_pre = header.index("pre_root_id")
            i_post = header.index("post_root_id")
        except ValueError:
            i_pre = header.index("pre")
            i_post = header.index("post")
        i_syn = None
        for cand in ("syn_count", "synapses", "count", "weight"):
            if cand in header:
                i_syn = header.index(cand)
                break
        i_nt = None
        for cand in ("nt_type", "neurotransmitter", "nt"):
            if cand in header:
                i_nt = header.index(cand)
                break
        if i_syn is None:
            raise SystemExit(f"找不到突触数字段，表头: {header}")
        print(f"  connections 列: {header}")
        for row in rd:
            try:
                yield (row[i_pre], row[i_post],
                       int(float(row[i_syn])),
                       (row[i_nt].strip() if i_nt is not None else ""))
            except (ValueError, IndexError):
                continue


def sign_of(nt):
    nt = (nt or "").upper()
    if nt in INHIBITORY:
        return -1.0
    if nt in EXCITATORY:
        return 1.0
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--partners", type=int, default=PARTNERS_PER_GROUP)
    ap.add_argument("--ascending", type=int, default=ASCENDING_LIMIT)
    ap.add_argument("--wind", type=int, default=WIND_LIMIT)
    ap.add_argument("--gain-e", type=float, default=GAIN_E,
                    help="每个神经元的总兴奋输入归一化到该值（阈值=1.0）")
    ap.add_argument("--gain-i", type=float, default=GAIN_I,
                    help="每个神经元的总抑制输入归一化到该值")
    ap.add_argument("--specificity", type=float, default=SPECIFICITY,
                    help="伙伴神经元对目标群的输出占比下限，用于剔除共享枢纽")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="归一化幂次：1=纯一致性检测（丢失强度差异），0=保留原始突触数")
    ap.add_argument("--no-jo-down", action="store_true",
                    help="不补 JO 的直接下游（用于对比 wind 群孤立时的行为）")
    a = ap.parse_args()

    print("[1/5] 读取细胞类型")
    types = load_cell_types()
    print(f"  {len(types)} 个有命名类型")
    if not types:
        raise SystemExit("缺少 consolidated_cell_types.csv.gz")

    print("[2/5] 读取分类与坐标")
    meta = load_classification()
    pos = load_positions()
    print(f"  {len(meta)} 个神经元，{len(pos)} 个有坐标")

    print("[3/5] 定位核心群")
    by_type = defaultdict(list)
    for rid, ct in types.items():
        by_type[ct].append(rid)

    core = {}
    for role, names in CORE_GROUPS.items():
        found = []
        for nm in names:
            found.extend(by_type.get(nm, []))
        if found:
            core[role] = found
            print(f"  {role:<12} {len(found):>4}  ({'/'.join(names)})")
        else:
            print(f"  {role:<12}    0  ({'/'.join(names)})  未找到")
    if not core:
        raise SystemExit("一个核心群都没找到，检查 cell_type 命名")

    core_ids = set()
    for v in core.values():
        core_ids.update(v)

    print("[4/5] 统计核心群的伙伴强度（第一遍扫描）")
    role_of_core = {}
    for role, ids in core.items():
        for rid in ids:
            role_of_core[rid] = role

    strength = defaultdict(int)
    per_role = defaultdict(lambda: defaultdict(int))
    out_total = defaultdict(int)
    total_edges = 0
    for pre, post, syn, _nt in iter_connections():
        total_edges += 1
        out_total[pre] += syn
        if pre in role_of_core and post not in core_ids:
            r = role_of_core[pre]
            strength[post] += syn
            per_role[r][post] += syn
        elif post in role_of_core and pre not in core_ids:
            r = role_of_core[post]
            strength[pre] += syn
            per_role[r][pre] += syn
    print(f"  扫描 {total_edges} 条边，{len(strength)} 个候选伙伴")

    partners = []
    claimed = set(core_ids)
    for role in core:
        ranked = sorted(per_role[role].items(), key=lambda kv: -kv[1])
        picked = []
        for rid, wsum in ranked:
            if rid in claimed:
                continue
            share = wsum / max(1, out_total.get(rid, 0))
            if share < a.specificity:
                continue
            claimed.add(rid)
            picked.append(rid)
            if len(picked) >= a.partners:
                break
        partners.extend(picked)
        print(f"    {role:<12} 取 {len(picked)} 个特异伙伴（≥{a.specificity:.0%} 输出）")
    print(f"  伙伴神经元 {len(partners)}（互不重叠）")

    asc = [r for r, m in meta.items() if m["super_class"] == "ascending"]
    asc.sort(key=lambda r: -strength.get(r, 0))
    asc = asc[:a.ascending]
    wind = [r for r, ct in types.items() if ct.upper().startswith("JO-")]
    wind = [r for r in wind if r not in core_ids][:a.wind]
    print(f"  上行 {len(asc)}，风感(JO) {len(wind)}")

    selected = list(dict.fromkeys(list(core_ids) + partners + asc + wind))
    sel = set(selected)
    print(f"  合计 {len(selected)} 个神经元")


    jo_down = []
    if wind and not a.no_jo_down:
        print("[4.5/5] 补 JO 直接下游")
        wind_set = set(wind)
        seen = set()
        for pre, post, _syn, _nt in iter_connections():
            if pre in wind_set and post not in sel and post not in seen:
                seen.add(post)
                jo_down.append(post)
        for rid in jo_down:
            sel.add(rid)
            selected.append(rid)
        print(f"  JO 下游 {len(jo_down)} 个，合计 {len(selected)} 个神经元")

    print("[5/5] 抽取子图（第二遍扫描）")
    edges = []
    for pre, post, syn, nt in iter_connections():
        if pre in sel and post in sel:
            edges.append((pre, post, syn, sign_of(nt)))
    print(f"  {len(edges)} 条突触")

    index = {rid: i for i, rid in enumerate(selected)}
    role_of = {}
    for role, ids in core.items():
        for rid in ids:
            role_of[rid] = role
    for rid in partners:
        role_of.setdefault(rid, "partner")
    for rid in asc:
        role_of.setdefault(rid, "ascending")
    for rid in wind:
        role_of.setdefault(rid, "wind")
    for rid in jo_down:
        role_of.setdefault(rid, "jo_down")

    syns = [e[2] for e in edges]
    syns.sort()
    p99 = syns[int(len(syns) * 0.99)] if syns else 1

    n = len(selected)
    exc_in = defaultdict(int)
    inh_in = defaultdict(int)
    for _p, q, s, sc in edges:
        if sc > 0:
            exc_in[q] += s
        else:
            inh_in[q] += s

    alpha = a.alpha
    ref_e = _median([v for v in exc_in.values() if v > 0]) or 1.0
    ref_i = _median([v for v in inh_in.values() if v > 0]) or 1.0

    def _denom(tot, ref):
        if tot <= 0:
            return 0.0
        return (tot ** alpha) * (ref ** (1.0 - alpha))

    def weight_of(p, q, s, sc):
        if sc > 0:
            d = _denom(exc_in.get(q, 0), ref_e)
            return s * (a.gain_e / d) if d else 0.0
        d = _denom(inh_in.get(q, 0), ref_i)
        return -s * (a.gain_i / d) if d else 0.0

    neurons = []
    for rid in selected:
        p = pos.get(rid)
        neurons.append({
            "id": int(rid),
            "type": types.get(rid, ""),
            "role": role_of.get(rid, ""),
            "side": meta.get(rid, {}).get("side", ""),
            "pos": p,
        })

    out_edges = []
    for p, q, s, sc in edges:
        w = weight_of(p, q, s, sc)
        if w:
            out_edges.append([index[p], index[q], round(w, 6)])

    data = {
        "meta": {
            "source": "FlyWire FAFB v783",
            "n_neurons": n,
            "n_edges": len(out_edges),
            "gain_e": a.gain_e,
            "gain_i": a.gain_i,
            "alpha": a.alpha,
            "p99_syn": p99,
            "core_groups": {k: len(v) for k, v in core.items()},
        },
        "neurons": neurons,
        "edges": out_edges,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"已写出 {a.out}  ({os.path.getsize(a.out) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

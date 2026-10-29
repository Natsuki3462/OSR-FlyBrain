
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from etl import iter_connections, load_cell_types
from brain.circuit import Circuit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="wind")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    c = Circuit()
    idx2role = {i: nb.get("role", "") for i, nb in enumerate(c.neurons)}
    ids = {str(c.neurons[i]["id"]) for i in c.by_role.get(a.role, [])}
    inside = {str(nb["id"]) for nb in c.neurons}
    print(f"{a.role} 群 {len(ids)} 个神经元，子图共 {len(inside)} 个")

    types = load_cell_types()
    print("扫 connections ...")
    out_w = defaultdict(int)
    in_w = defaultdict(int)
    out_edges = in_edges = 0
    for pre, post, syn, _nt in iter_connections():
        if pre in ids:
            out_w[post] += syn
            out_edges += 1
        if post in ids:
            in_w[pre] += syn
            in_edges += 1

    def report(d, n_edges, label):
        print(f"\n{label}：{n_edges} 条边，{len(d)} 个对象")
        by_type = Counter()
        for rid, s in d.items():
            by_type[types.get(rid, "（无类型）")] += s
        for t, s in by_type.most_common(a.top):
            n_cells = sum(1 for r in d if types.get(r, "（无类型）") == t)
            mark = "  ← 已在子图" if any(r in inside for r in d
                                        if types.get(r, "（无类型）") == t) else ""
            print(f"  {t:<24}{n_cells:>5} 个 / {s:>7} 突触{mark}")
        hit = sum(1 for r in d if r in inside)
        print(f"  其中已在子图内：{hit} / {len(d)}")

    report(out_w, out_edges, "出边（它投射给谁）")
    report(in_w, in_edges, "入边（谁投射给它）")


    d1 = set(out_w.keys())
    print(f"\n第二跳：{len(d1)} 个直接下游的输出去向")
    core_ids = {}
    for nb in c.neurons:
        r = nb.get("role", "")
        if r and r not in ("partner", "ascending", "wind",
                           "loom_lc4", "loom_lplc2"):
            core_ids[str(nb["id"])] = r
    hit_core = Counter()
    by_type2 = Counter()
    d2 = defaultdict(int)
    for pre, post, syn, _nt in iter_connections():
        if pre in d1:
            d2[post] += syn
            by_type2[types.get(post, "（无类型）")] += syn
            if post in core_ids:
                hit_core[core_ids[post]] += syn
    for t, s in by_type2.most_common(a.top):
        print(f"  {t:<24}{s:>7} 突触")
    if hit_core:
        print("  命中现有命令群：" + ", ".join(
            f"{k}×{v}" for k, v in hit_core.most_common()))
    else:
        print("  没有命中任何现有命令群")
    return 0


if __name__ == "__main__":
    sys.exit(main())

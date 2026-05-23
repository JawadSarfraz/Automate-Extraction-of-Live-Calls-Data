"""Build the v2-vs-v3 DISAGREEMENT labeling queue.

The highest-information calls to human-label are the ones where the rule-based v2
and the learned v3 disagree (~35% of calls). This page shows only those, with
BOTH labels side by side, v3's confidence + per-class probabilities, the platform
category, the isolated customer words, the full transcript, an inline player, and
one-click relabel buttons. Export -> my_labels_disagree_<date>.json, which carries
v2_label + v3_label + weak_label alongside your my_label so a later step can
retarget v3 to OUR taxonomy.

Sorted to adjudicate systematic disagreements in batches: grouped by the
(v2 -> v3) transition, most-confident v3 first. Filter chips per transition + class.

Usage:
  .venv/bin/python ml/build_review_disagree.py --date 2026-05-14
  .venv/bin/python ml/build_review_disagree.py --date 2026-05-14 --audio remote
"""
from __future__ import annotations
import argparse
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
COLOR = {"fair": "#388E3C", "simple_vm": "#F57C00", "smart_vm": "#C2185B",
         "no_contact": "#90A4AE", "uncertain": "#9E9E9E"}


def parse_file(fn: str):
    parts = fn.split("-")
    t, phone = "", ""
    if len(parts) >= 3:
        hms = parts[1]
        if len(hms) == 6 and hms.isdigit():
            t = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}"
        phone = parts[2]
    return t, phone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    ap.add_argument("--audio", choices=["local", "remote"], default="local",
                    help="local = play files next to the HTML; remote = public URLs (shareable single file)")
    args = ap.parse_args()
    date = args.date

    v3 = json.load(open(os.path.join(DATA, "analysis", f"classified_v3_{date}.json")))
    v2 = {r["file"]: r for r in json.load(open(os.path.join(DATA, "analysis", f"classified_v2_{date}.json")))}

    url_map = {}
    if args.audio == "remote":
        for r in json.load(open(os.path.join(DATA, f"recordings_{date}.json")))["recordings"]:
            url_map[os.path.basename(r["file_url"].split("?")[0])] = r["file_url"]

    def audio_src(fn):
        return url_map.get(fn, fn) if args.audio == "remote" else fn

    # Disagreements only.
    rows = [r for r in v3 if r["v2_label"] != r["v3_label"]]
    prob_keys = sorted(k for k in (rows[0].keys() if rows else []) if k.startswith("p_"))

    # Stable id in chronological order, then display sort: group by (v2->v3), conf desc.
    rows.sort(key=lambda r: r["file"])
    for i, r in enumerate(rows, 1):
        r["_id"] = i
        r["_time"], r["_phone"] = parse_file(r["file"])
        r["_pair"] = f'{r["v2_label"]}→{r["v3_label"]}'
        v2r = v2.get(r["file"], {})
        r["_full"] = v2r.get("full_text", "")
        r["_v2reasons"] = v2r.get("v2_reasons", "")
        r["_dur"] = v2r.get("duration", "")
    rows.sort(key=lambda r: (r["_pair"], -float(r.get("v3_conf", 0) or 0)))

    def esc(s):
        return html.escape(str(s or ""))

    def probbar(r):
        cells = []
        for k in prob_keys:
            cls = k[2:]
            p = float(r.get(k, 0) or 0)
            cells.append(f'<span class="pb" title="{cls} {p:.2f}"><i style="width:{int(p*46)}px;background:{COLOR.get(cls,"#bbb")}"></i>{cls[:2]}</span>')
        return "".join(cells)

    def row(r):
        c2, c3 = COLOR.get(r["v2_label"], "#bbb"), COLOR.get(r["v3_label"], "#bbb")
        return f"""<tr data-pair="{r['_pair']}" data-v2="{r['v2_label']}" data-v3="{r['v3_label']}" data-id="{r['_id']}" data-phone="{r['_phone']}" data-time="{r['_time']}" data-platform="{esc(r['category'])}" data-weak="{esc(r['weak_label'])}" data-file="{r['file']}">
  <td class="id">#{r['_id']}</td>
  <td class="ph">{r['_phone']}<div class="tm">{r['_time']}</div></td>
  <td><span class="pill" style="background:{c2}">{r['v2_label']}</span></td>
  <td><span class="pill" style="background:{c3}">{r['v3_label']}</span><div class="cf">conf {r.get('v3_conf','')}</div></td>
  <td class="pbs">{probbar(r)}</td>
  <td>{esc(r['category'])}<div class="tm">weak: {esc(r['weak_label'])}</div></td>
  <td class="cust"><b>{esc(r['customer_text']) or '<i>(no customer speech)</i>'}</b></td>
  <td class="full" title="{esc(r['_full'])}">{esc(r['_full'][:90])}…</td>
  <td>{esc(r['_dur'])}s</td>
  <td><audio controls preload="none" src="{audio_src(r['file'])}"></audio></td>
  <td class="rl">
    <button type="button" onclick="lbl(this,'fair')" title="Fair">F</button><button type="button" onclick="lbl(this,'simple_vm')" title="Simple VM">Si</button>
    <button type="button" onclick="lbl(this,'smart_vm')" title="Smart VM">Sm</button><button type="button" onclick="lbl(this,'no_contact')" title="No-contact">N</button>
  </td>
</tr>"""

    pair_counts = {}
    for r in rows:
        pair_counts[r["_pair"]] = pair_counts.get(r["_pair"], 0) + 1
    chips = " ".join(f'<button onclick="filt(\'{p}\')">{esc(p)} ({n})</button>'
                     for p, n in sorted(pair_counts.items(), key=lambda x: -x[1]))
    body = "\n".join(row(r) for r in rows)
    total = len(v3)
    agree = sum(1 for r in v3 if r["v2_label"] == r["v3_label"])

    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>v2 vs v3 disagreements — {date}</title>
<style>
 body{{font:13px system-ui,Arial;margin:0;background:#f4f5f7}}
 header{{position:sticky;top:0;background:#0f172a;color:#fff;padding:10px 16px;z-index:5}}
 .legend{{font-size:11px;opacity:.85;margin-top:4px}} .legend b{{color:#fff}}
 .bar{{margin:8px 16px}} .bar button{{margin:2px;padding:5px 9px;border:2px solid #ccc;border-radius:14px;background:#fff;cursor:pointer;font-size:11px}}
 #q{{margin:6px 16px;padding:7px 9px;width:300px;border:1px solid #ccc;border-radius:6px}}
 table{{border-collapse:collapse;width:calc(100% - 32px);margin:8px 16px 60px;background:#fff}}
 th,td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}}
 th{{position:sticky;top:62px;background:#fafbfc;font-size:11px;text-transform:uppercase;color:#555}}
 .id{{font-weight:700;color:#0f172a;white-space:nowrap}} .ph{{font-family:monospace;color:#334155;white-space:nowrap}} .tm{{font-size:10px;color:#94a3b8}}
 .pill{{color:#fff;padding:2px 8px;border-radius:10px;font-size:11px}} .cf{{font-size:10px;color:#888;margin-top:3px}}
 .pbs{{white-space:nowrap}} .pb{{display:inline-block;font-size:9px;color:#555;margin-right:4px}} .pb i{{display:block;height:5px;border-radius:3px;margin-bottom:1px}}
 .cust{{max-width:260px;color:#0f172a}} .full{{max-width:170px;font-size:11px;color:#999}}
 audio{{height:30px}}
 .rl button{{margin:2px;padding:6px 10px;font-size:13px;font-weight:600;cursor:pointer;border:1px solid #bbb;border-radius:5px;background:#fff}}
 .rl button:hover{{background:#e2e8f0}} .rl button.on{{background:#0f172a;color:#fff;border-color:#0f172a}}
 tr.lbl-fair{{background:#e8f5e9;box-shadow:inset 5px 0 #388E3C}} tr.lbl-simple_vm{{background:#fff3e0;box-shadow:inset 5px 0 #F57C00}}
 tr.lbl-smart_vm{{background:#fce4ec;box-shadow:inset 5px 0 #C2185B}} tr.lbl-no_contact{{background:#eceff1;box-shadow:inset 5px 0 #90A4AE}}
</style></head><body>
<header><b>⚖️ v2 vs v3 disagreements — {date}</b> &nbsp; {len(rows)} of {total} calls disagree ({100*agree/total:.0f}% agree)
 &nbsp;<button onclick="exp()" style="padding:5px 10px">⬇ Export my labels (<b id="cnt">0</b>)</button>
 <div class="legend">These are the calls where rule-based <b>v2</b> and learned <b>v3</b> differ — the most informative to verify. Listen, then click the TRUE class: <b>F</b>=Fair &nbsp;<b>Si</b>=Simple VM &nbsp;<b>Sm</b>=Smart VM &nbsp;<b>N</b>=No-contact. The bars show v3's probability per class. Export carries both model labels so we can retarget v3 to your taxonomy.</div></header>
<div class="bar"><button onclick="filt('all')" style="border-color:#333"><b>all ({len(rows)})</b></button> {chips}</div>
<input id="q" placeholder="search ID / phone / customer text / transcript…" oninput="qf(this.value)">
<table id="t"><thead><tr><th>ID</th><th>phone / time</th><th>v2</th><th>v3</th><th>v3 probs</th><th>platform</th><th>customer words</th><th>full transcript</th><th>dur</th><th>listen</th><th>TRUE label</th></tr></thead>
<tbody>{body}</tbody></table>
<script>
 const my={{}};
 function lbl(b,c){{const tr=b.closest('tr');
   tr.classList.remove('lbl-fair','lbl-simple_vm','lbl-smart_vm','lbl-no_contact');tr.classList.add('lbl-'+c);
   tr.querySelectorAll('.rl button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
   const d=tr.dataset;
   my[d.id]={{id:+d.id,phone:d.phone,time:d.time,file:d.file,platform:d.platform,weak_label:d.weak,v2_label:d.v2,v3_label:d.v3,my_label:c}};
   document.getElementById('cnt').textContent=Object.keys(my).length;}}
 function filt(p){{for(const tr of document.querySelectorAll('#t tbody tr'))tr.style.display=(p==='all'||tr.dataset.pair===p)?'':'none';}}
 function qf(v){{v=v.toLowerCase();for(const tr of document.querySelectorAll('#t tbody tr')){{const t=(tr.dataset.id+' '+tr.dataset.phone+' '+tr.children[6].textContent+' '+tr.children[7].textContent).toLowerCase();tr.style.display=t.includes(v)?'':'none';}}}}
 function exp(){{const arr=Object.values(my).sort((a,b)=>a.id-b.id);
   const b=new Blob([JSON.stringify(arr,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='my_labels_disagree_{date}.json';a.click();}}
</script></body></html>"""
    name = "review_disagree_shareable.html" if args.audio == "remote" else "review_disagree.html"
    path = os.path.join(DATA, "audio", date, name)
    open(path, "w").write(out)
    note = " (audio streams from public URLs — send this single file)" if args.audio == "remote" else ""
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} disagreement rows){note}")
    print("Top disagreement transitions (v2 → v3):")
    for p, n in sorted(pair_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"  {p:32} {n}")


if __name__ == "__main__":
    main()

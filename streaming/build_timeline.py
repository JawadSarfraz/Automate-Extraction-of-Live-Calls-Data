"""Build an HTML timeline viewer for the streaming stopwatch.

For each call it shows, second by second, what the detector saw and decided — a
colored strip (one cell per step, colored by the class v2 assigned at that moment),
the exact moment it FLAGGED the call (★), the revealed customer words, the outcome,
and an inline audio player so you can listen and verify the timing.

Writes to data/audio/<date>/timeline_<date>.html (served by the local HTTP server,
audio sits alongside). Default shows the informative calls (anything flagged + all
true VM/bot + all Fair false-flags); use --all for every call.

Usage:
  .venv/bin/python -m streaming.build_timeline --date 2026-05-14
  .venv/bin/python -m streaming.build_timeline --date 2026-05-14 --all
"""
from __future__ import annotations
import argparse
import html
import json
import os

from . import _paths
from .sources import OracleTranscriptSource
from .classifiers import V2RulesClassifier
from .trigger import ConfidenceTrigger
from .detector import StreamingDetector
from .report import CLASSES

from separate import learn_bot_bank  # noqa: E402

COLOR = {"fair": "#388E3C", "simple_vm": "#F57C00", "smart_vm": "#C2185B",
         "no_contact": "#90A4AE", "uncertain": "#CFD8DC"}
OUTCOME_COLOR = {"caught": "#2e7d32", "missed": "#b0bec5", "wrong": "#8e24aa",
                 "FALSE-FLAG": "#c62828", "fair-ok": "#2e7d32", "fair-noflag": "#90a4ae",
                 "false-pos": "#c62828", "ok-silent": "#90a4ae"}


def parse_file(fn: str):
    parts = fn.split("-")
    t, phone = "", ""
    if len(parts) >= 3 and len(parts[1]) == 6 and parts[1].isdigit():
        hms = parts[1]
        t = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}"
        phone = parts[2]
    return t, phone


def outcome_of(true_c, fired, fired_c):
    if true_c in ("simple_vm", "smart_vm"):
        if not fired:
            return "missed"
        return "caught" if fired_c == true_c else "wrong"
    if true_c == "fair":
        if not fired:
            return "fair-noflag"
        return "fair-ok" if fired_c == "fair" else "FALSE-FLAG"
    # no_contact
    return "false-pos" if fired else "ok-silent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--all", action="store_true", help="include every call (default: only informative ones)")
    ap.add_argument("--audio", choices=["local", "remote"], default="local",
                    help="local = play files next to the HTML (needs the server); "
                         "remote = bake in public URLs -> a single standalone file you just open")
    args = ap.parse_args()
    date = args.date

    tdir = os.path.join(_paths.DATA, "transcripts", date)
    v1 = {r["file"]: r for r in json.load(open(os.path.join(_paths.DATA, "analysis", f"classified_{date}.json")))}

    # filename -> public URL (for the standalone, shareable file)
    url_map = {}
    if args.audio == "remote":
        recs = json.load(open(os.path.join(_paths.DATA, f"recordings_{date}.json")))["recordings"]
        for r in recs:
            url_map[os.path.basename(r["file_url"].split("?")[0])] = r["file_url"]

    def audio_src(fn):
        return url_map.get(fn, fn) if args.audio == "remote" else fn
    bank, _, _ = learn_bot_bank(tdir)
    detector = StreamingDetector(V2RulesClassifier(),
                                 lambda: ConfidenceTrigger(tau=args.tau, k=args.k))

    rows = []
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".json"):
            continue
        tj = json.load(open(os.path.join(tdir, fn)))
        wav = tj.get("file", "")
        true_c = v1.get(wav, {}).get("weak_label", "")
        if true_c not in CLASSES:
            continue
        res = detector.run(OracleTranscriptSource(tj, bank, step=args.step))
        oc = outcome_of(true_c, res.fired, res.fired_label)
        if not args.all and not (res.fired or true_c in ("simple_vm", "smart_vm")):
            continue
        rows.append((wav, true_c, res, oc))

    # order: false-flags first, then caught VMs/bots, then the rest — most informative on top
    pri = {"FALSE-FLAG": 0, "wrong": 1, "caught": 2, "missed": 3, "false-pos": 4,
           "fair-ok": 5, "fair-noflag": 6, "ok-silent": 7}
    rows.sort(key=lambda r: (pri.get(r[3], 9), (r[2].fire_time if r[2].fire_time is not None else 1e9)))

    def esc(s):
        return html.escape(str(s or ""))

    def strip_html(res):
        cells = []
        for st in res.trajectory:
            fired_here = res.fired and res.fire_time is not None and abs(st.t - res.fire_time) < 1e-6
            star = "★" if fired_here else ""
            bd = "border:2px solid #000;font-weight:700" if fired_here else "border:1px solid #fff"
            tip = f"t={st.t:.0f}s  {st.label} {st.conf:.2f}\\ncust: {st.customer_text[:120]}"
            cells.append(f'<span class="cell" style="background:{COLOR.get(st.label,"#eee")};{bd}" title="{esc(tip)}">{star}</span>')
        return "".join(cells)

    def row_html(i, wav, true_c, res, oc):
        tm, ph = parse_file(wav)
        fired_txt = (f'<b style="color:{COLOR.get(res.fired_label,"#333")}">{res.fired_label}</b> '
                     f'@ <b>{res.fire_time:.0f}s</b>' if res.fired else '<i>no flag</i>')
        lead = f"{res.call_len - res.fire_time:.0f}s" if res.fired else "-"
        cust_at_fire = ""
        if res.fired:
            for st in res.trajectory:
                if res.fire_time is not None and abs(st.t - res.fire_time) < 1e-6:
                    cust_at_fire = st.customer_text[:160]
                    break
        return f"""<tr data-true="{true_c}" data-oc="{oc}">
  <td class="id">#{i}</td>
  <td class="ph">{ph}<div class="tm">{tm}</div></td>
  <td><span class="pill" style="background:{COLOR.get(true_c,'#bbb')}">{true_c}</span></td>
  <td>{fired_txt}</td>
  <td><span class="oc" style="background:{OUTCOME_COLOR.get(oc,'#777')}">{oc}</span></td>
  <td class="num">{res.call_len:.0f}s</td>
  <td class="num">{lead}</td>
  <td class="aud"><audio controls preload="none" src="{esc(audio_src(wav))}"></audio></td>
  <td class="cust">{esc(cust_at_fire)}</td>
  <td class="strip"><div class="sw">{strip_html(res)}</div></td>
</tr>"""

    body = "\n".join(row_html(i + 1, *r) for i, r in enumerate(rows))
    from collections import Counter
    oc_counts = Counter(r[3] for r in rows)
    chips = " ".join(f'<button onclick="filt(\'{o}\')" style="border-color:{OUTCOME_COLOR.get(o,"#777")}">{o} ({n})</button>'
                     for o, n in sorted(oc_counts.items(), key=lambda x: -x[1]))
    legend = " ".join(f'<span class="cell" style="background:{c};border:1px solid #ccc"></span>{k}'
                      for k, c in COLOR.items())

    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>Streaming timeline — {date}</title>
<style>
 body{{font:13px system-ui,Arial;margin:0;background:#f4f5f7}}
 header{{position:sticky;top:0;background:#0f172a;color:#fff;padding:10px 16px;z-index:5}}
 .bar{{margin:8px 16px}} .bar button{{margin:2px;padding:5px 9px;border:2px solid #ccc;border-radius:14px;background:#fff;cursor:pointer;font-size:11px}}
 .leg{{margin:4px 16px;font-size:11px;color:#fff}} .leg .cell{{margin:0 3px 0 10px}}
 table{{border-collapse:collapse;width:calc(100% - 32px);margin:8px 16px 60px;background:#fff}}
 th,td{{padding:5px 7px;border-bottom:1px solid #eee;text-align:left;vertical-align:middle}}
 th{{position:sticky;top:60px;background:#fafbfc;font-size:11px;text-transform:uppercase;color:#555}}
 .id{{font-weight:700;color:#0f172a}} .ph{{font-family:monospace;color:#334155;white-space:nowrap}} .tm{{font-size:10px;color:#94a3b8}}
 .pill{{color:#fff;padding:2px 8px;border-radius:10px;font-size:11px}}
 .oc{{color:#fff;padding:2px 7px;border-radius:6px;font-size:10px;font-weight:700}}
 .num{{text-align:right;font-variant-numeric:tabular-nums;color:#475569}}
 .aud{{width:420px}} audio{{height:42px;width:410px}}
 .sw{{max-width:780px;overflow-x:auto;white-space:nowrap;line-height:0;padding-bottom:4px}}
 .cell{{display:inline-block;width:12px;height:15px;text-align:center;font-size:9px;color:#fff;line-height:15px}}
 .cust{{max-width:240px;font-size:11px;color:#0f172a}}
</style></head><body>
<header><b>⏱ Streaming detection timeline — {date}</b> &nbsp; {len(rows)} calls &nbsp;|&nbsp;
 v2 brain, step={args.step}s, tau={args.tau}, k={args.k} &nbsp;|&nbsp; each square = 1s, colored by what v2 said; ★ = moment flagged
 <div class="leg">colors: {legend}</div></header>
<div class="bar"><button onclick="filt('all')" style="border-color:#333"><b>all ({len(rows)})</b></button> {chips}</div>
<table id="t"><thead><tr><th>#</th><th>phone/time</th><th>true</th><th>flagged</th><th>outcome</th><th>len</th><th>saved</th><th>🔊 listen</th><th>customer words @ flag</th><th>per-second timeline (hover a square) →</th></tr></thead>
<tbody>{body}</tbody></table>
<script>
 function filt(o){{for(const tr of document.querySelectorAll('#t tbody tr'))tr.style.display=(o==='all'||tr.dataset.oc===o)?'':'none';}}
</script></body></html>"""

    name = f"timeline_{date}_shareable.html" if args.audio == "remote" else f"timeline_{date}.html"
    out_path = os.path.join(_paths.DATA, "audio", date, name)
    open(out_path, "w").write(out)
    note = " (standalone — audio streams from public URLs; just download + open)" if args.audio == "remote" else ""
    print(f"wrote {os.path.relpath(out_path, _paths.ROOT)} ({len(rows)} rows){note}")
    print("outcome counts:", dict(oc_counts))


if __name__ == "__main__":
    main()

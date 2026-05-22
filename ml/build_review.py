"""Build a review/labeling HTML page from classified_<date>.json.

Shows every recording with: platform-derived class, category, transcript snippet,
key acoustic features, our audio-only prediction, an inline player, and
Fair/Simple/Smart/No-contact relabel buttons that export your corrections to JSON
(for bootstrapping a supervised model later).
"""
from __future__ import annotations
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CLASS_COLOR = {"fair": "#388E3C", "simple_vm": "#F57C00", "smart_vm": "#C2185B",
               "no_contact": "#90A4AE", "uncertain": "#9E9E9E", "unknown": "#bbb", "unmatched": "#ddd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    args = ap.parse_args()
    date = args.date
    rows = json.load(open(os.path.join(DATA, "analysis", f"classified_{date}.json")))

    def cell(r):
        c = CLASS_COLOR.get(r["weak_label"], "#bbb")
        flags = []
        if r["beep"]:
            flags.append("🔔beep")
        if r["bg_ratio"] and r["bg_ratio"] > 0.5:
            flags.append("🎵bg")
        agree = "✓" if r["weak_label"] == r["audio_pred"] else "✗"
        return f"""<tr data-class="{r['weak_label']}">
  <td><span class="pill" style="background:{c}">{r['weak_label']}</span></td>
  <td>{r['category']}</td>
  <td class="tx">{(r['snippet'] or '')}</td>
  <td>{r['duration']}s</td>
  <td>{r['n_segments']}</td>
  <td>{r['speech_ratio']}</td>
  <td>{' '.join(flags)}</td>
  <td class="ap">{r['audio_pred']} {agree}</td>
  <td><audio controls preload="none" src="{r['file']}"></audio></td>
  <td class="relabel">
    <button onclick="lbl(this,'fair')">F</button>
    <button onclick="lbl(this,'simple_vm')">Si</button>
    <button onclick="lbl(this,'smart_vm')">Sm</button>
    <button onclick="lbl(this,'no_contact')">N</button>
  </td>
</tr>"""

    body = "\n".join(cell(r) for r in rows)
    counts = {}
    for r in rows:
        counts[r["weak_label"]] = counts.get(r["weak_label"], 0) + 1
    chips = " ".join(f'<button onclick="filt(\'{k}\')" style="border-color:{CLASS_COLOR.get(k,"#bbb")}">{k} ({v})</button>'
                     for k, v in sorted(counts.items(), key=lambda x: -x[1]))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Review {date} — Fair / Simple / Smart</title>
<style>
 body{{font:13px system-ui,Arial;margin:0;background:#f4f5f7;color:#222}}
 header{{position:sticky;top:0;background:#1f2937;color:#fff;padding:10px 16px;z-index:5}}
 header b{{font-size:15px}} .bar{{margin:8px 16px}} .bar button{{margin:2px;padding:5px 9px;border:2px solid #ccc;border-radius:14px;background:#fff;cursor:pointer}}
 #q{{margin:6px 16px;padding:7px 9px;width:240px;border:1px solid #ccc;border-radius:6px}}
 table{{border-collapse:collapse;width:calc(100% - 32px);margin:8px 16px 60px;background:#fff}}
 th,td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:middle}}
 th{{position:sticky;top:46px;background:#fafbfc;font-size:11px;text-transform:uppercase;color:#555}}
 .pill{{color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;white-space:nowrap}}
 .tx{{max-width:280px;color:#374151}} .ap{{font-size:11px;color:#666;white-space:nowrap}}
 audio{{height:30px}} .relabel button{{margin:1px;padding:3px 6px;cursor:pointer;border:1px solid #bbb;border-radius:4px;background:#fff}}
 tr.lbl-fair{{box-shadow:inset 4px 0 #388E3C}} tr.lbl-simple_vm{{box-shadow:inset 4px 0 #F57C00}}
 tr.lbl-smart_vm{{box-shadow:inset 4px 0 #C2185B}} tr.lbl-no_contact{{box-shadow:inset 4px 0 #90A4AE}}
</style></head><body>
<header><b>🔎 Call Review — {date}</b> &nbsp; campaign 270 &nbsp; {len(rows)} recordings
 &nbsp;|&nbsp; class = platform-derived (your taxonomy); ✓/✗ = audio-only model agreement
 &nbsp; <button onclick="exportLabels()" style="padding:5px 10px">⬇ Export my labels</button></header>
<div class="bar"><button onclick="filt('all')" style="border-color:#333"><b>all ({len(rows)})</b></button> {chips}</div>
<input id="q" placeholder="filter transcript/category…" oninput="qfilt(this.value)">
<table id="t"><thead><tr>
 <th>class</th><th>category</th><th>snippet</th><th>dur</th><th>segs</th><th>speech</th><th>flags</th><th>audio-pred</th><th>listen</th><th>relabel</th>
</tr></thead><tbody>
{body}
</tbody></table>
<script>
 const my={{}};
 function lbl(btn,c){{const tr=btn.closest('tr');tr.className='lbl-'+c;my[tr.querySelector('audio').getAttribute('src')]=c;}}
 function filt(c){{for(const tr of document.querySelectorAll('#t tbody tr'))tr.style.display=(c==='all'||tr.dataset.class===c)?'':'none';}}
 function qfilt(v){{v=v.toLowerCase();for(const tr of document.querySelectorAll('#t tbody tr')){{const t=(tr.children[1].textContent+' '+tr.children[2].textContent).toLowerCase();tr.style.display=t.includes(v)?'':'none';}}}}
 function exportLabels(){{const blob=new Blob([JSON.stringify(my,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='my_labels_{date}.json';a.click();}}
</script></body></html>"""

    out = os.path.join(DATA, "audio", date, "review.html")
    open(out, "w").write(html)
    print(f"wrote {os.path.relpath(out, ROOT)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

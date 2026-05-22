"""Build the v2 review/labeling page: shows our content-based class, the
isolated CUSTOMER words, the full transcript, reasons, an inline player, and
one-click relabel buttons that export corrections to JSON.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    args = ap.parse_args()
    date = args.date
    rows = json.load(open(os.path.join(DATA, "analysis", f"classified_v2_{date}.json")))
    rows.sort(key=lambda r: (r["v2_label"], -float(r.get("v2_conf", 0) or 0)))

    def esc(s):
        return html.escape(str(s or ""))

    def row(r):
        c = COLOR.get(r["v2_label"], "#bbb")
        return f"""<tr data-class="{r['v2_label']}">
  <td><span class="pill" style="background:{c}">{r['v2_label']}</span><div class="cf">{r.get('v2_conf','')}</div></td>
  <td>{esc(r['category'])}</td>
  <td class="cust"><b>{esc(r['customer_text']) or '<i>(no customer speech)</i>'}</b></td>
  <td class="rs">{esc(r['v2_reasons'])}</td>
  <td class="full" title="{esc(r['full_text'])}">{esc(r['full_text'][:90])}…</td>
  <td>{r['duration']}s</td>
  <td><audio controls preload="none" src="{r['file']}"></audio></td>
  <td class="rl">
    <button onclick="lbl(this,'fair')">F</button><button onclick="lbl(this,'simple_vm')">Si</button>
    <button onclick="lbl(this,'smart_vm')">Sm</button><button onclick="lbl(this,'no_contact')">N</button>
  </td>
</tr>"""

    counts = {}
    for r in rows:
        counts[r["v2_label"]] = counts.get(r["v2_label"], 0) + 1
    chips = " ".join(f'<button onclick="filt(\'{k}\')" style="border-color:{COLOR.get(k,"#bbb")}">{k} ({v})</button>'
                     for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    body = "\n".join(row(r) for r in rows)

    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>Review v2 — {date}</title>
<style>
 body{{font:13px system-ui,Arial;margin:0;background:#f4f5f7}}
 header{{position:sticky;top:0;background:#0f172a;color:#fff;padding:10px 16px;z-index:5}}
 .bar{{margin:8px 16px}} .bar button{{margin:2px;padding:5px 9px;border:2px solid #ccc;border-radius:14px;background:#fff;cursor:pointer}}
 #q{{margin:6px 16px;padding:7px 9px;width:260px;border:1px solid #ccc;border-radius:6px}}
 table{{border-collapse:collapse;width:calc(100% - 32px);margin:8px 16px 60px;background:#fff}}
 th,td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}}
 th{{position:sticky;top:46px;background:#fafbfc;font-size:11px;text-transform:uppercase;color:#555}}
 .pill{{color:#fff;padding:2px 8px;border-radius:10px;font-size:11px}} .cf{{font-size:10px;color:#888;margin-top:3px}}
 .cust{{max-width:280px;color:#0f172a}} .rs{{max-width:220px;font-size:11px;color:#555}} .full{{max-width:200px;font-size:11px;color:#999}}
 audio{{height:30px}} .rl button{{margin:1px;padding:3px 6px;cursor:pointer;border:1px solid #bbb;border-radius:4px;background:#fff}}
 tr.lbl-fair{{box-shadow:inset 4px 0 #388E3C}} tr.lbl-simple_vm{{box-shadow:inset 4px 0 #F57C00}}
 tr.lbl-smart_vm{{box-shadow:inset 4px 0 #C2185B}} tr.lbl-no_contact{{box-shadow:inset 4px 0 #90A4AE}}
</style></head><body>
<header><b>🧠 Call Review v2 — {date}</b> &nbsp; {len(rows)} calls &nbsp;|&nbsp; class from CUSTOMER words (bot script stripped)
 &nbsp;<button onclick="exp()" style="padding:5px 10px">⬇ Export my labels</button></header>
<div class="bar"><button onclick="filt('all')" style="border-color:#333"><b>all ({len(rows)})</b></button> {chips}</div>
<input id="q" placeholder="search customer text / transcript…" oninput="qf(this.value)">
<table id="t"><thead><tr><th>v2 class</th><th>platform</th><th>customer words</th><th>why</th><th>full transcript</th><th>dur</th><th>listen</th><th>relabel</th></tr></thead>
<tbody>{body}</tbody></table>
<script>
 const my={{}};
 function lbl(b,c){{const tr=b.closest('tr');tr.className='lbl-'+c;my[tr.querySelector('audio').getAttribute('src')]=c;}}
 function filt(c){{for(const tr of document.querySelectorAll('#t tbody tr'))tr.style.display=(c==='all'||tr.dataset.class===c)?'':'none';}}
 function qf(v){{v=v.toLowerCase();for(const tr of document.querySelectorAll('#t tbody tr')){{const t=(tr.children[2].textContent+' '+tr.children[4].textContent).toLowerCase();tr.style.display=t.includes(v)?'':'none';}}}}
 function exp(){{const b=new Blob([JSON.stringify(my,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='my_labels_v2_{date}.json';a.click();}}
</script></body></html>"""
    path = os.path.join(DATA, "audio", date, "review_v2.html")
    open(path, "w").write(out)
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

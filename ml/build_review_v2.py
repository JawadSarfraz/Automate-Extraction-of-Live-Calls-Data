"""Build the v2 review/labeling page: shows our content-based class, a stable ID
(+ phone & time), the isolated CUSTOMER words, the full transcript, reasons, an
inline player, and one-click relabel buttons that export rich corrections to JSON.
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
    """'20260514-200010-3176310317-….wav' -> (time 'HH:MM:SS', phone)."""
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
    rows = json.load(open(os.path.join(DATA, "analysis", f"classified_v2_{date}.json")))

    # filename -> public URL (for shareable mode)
    url_map = {}
    if args.audio == "remote":
        for r in json.load(open(os.path.join(DATA, f"recordings_{date}.json")))["recordings"]:
            url_map[os.path.basename(r["file_url"].split("?")[0])] = r["file_url"]

    def audio_src(fn):
        return url_map.get(fn, fn) if args.audio == "remote" else fn

    # Assign a STABLE id in chronological (filename) order, before display sorting.
    rows.sort(key=lambda r: r["file"])
    for i, r in enumerate(rows, 1):
        r["_id"] = i
        r["_time"], r["_phone"] = parse_file(r["file"])
    # Display order: grouped by class, highest confidence first.
    rows.sort(key=lambda r: (r["v2_label"], -float(r.get("v2_conf", 0) or 0)))

    def esc(s):
        return html.escape(str(s or ""))

    def row(r):
        c = COLOR.get(r["v2_label"], "#bbb")
        return f"""<tr data-class="{r['v2_label']}" data-id="{r['_id']}" data-phone="{r['_phone']}" data-time="{r['_time']}" data-platform="{esc(r['category'])}" data-v2="{r['v2_label']}" data-file="{r['file']}">
  <td class="id">#{r['_id']}</td>
  <td class="ph">{r['_phone']}<div class="tm">{r['_time']}</div></td>
  <td><span class="pill" style="background:{c}">{r['v2_label']}</span><div class="cf">{r.get('v2_conf','')}</div></td>
  <td>{esc(r['category'])}</td>
  <td class="cust"><b>{esc(r['customer_text']) or '<i>(no customer speech)</i>'}</b></td>
  <td class="rs">{esc(r['v2_reasons'])}</td>
  <td class="full" title="{esc(r['full_text'])}">{esc(r['full_text'][:90])}…</td>
  <td>{r['duration']}s</td>
  <td><audio controls preload="none" src="{audio_src(r['file'])}"></audio></td>
  <td class="rl">
    <button type="button" onclick="lbl(this,'fair')" title="Fair">F</button><button type="button" onclick="lbl(this,'simple_vm')" title="Simple VM">Si</button>
    <button type="button" onclick="lbl(this,'smart_vm')" title="Smart VM">Sm</button><button type="button" onclick="lbl(this,'no_contact')" title="No-contact">N</button>
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
 .legend{{font-size:11px;opacity:.85;margin-top:4px}} .legend b{{color:#fff}}
 .bar{{margin:8px 16px}} .bar button{{margin:2px;padding:5px 9px;border:2px solid #ccc;border-radius:14px;background:#fff;cursor:pointer}}
 #q{{margin:6px 16px;padding:7px 9px;width:260px;border:1px solid #ccc;border-radius:6px}}
 table{{border-collapse:collapse;width:calc(100% - 32px);margin:8px 16px 60px;background:#fff}}
 th,td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}}
 th{{position:sticky;top:62px;background:#fafbfc;font-size:11px;text-transform:uppercase;color:#555}}
 .id{{font-weight:700;color:#0f172a;white-space:nowrap}} .ph{{font-family:monospace;color:#334155;white-space:nowrap}} .tm{{font-size:10px;color:#94a3b8}}
 .pill{{color:#fff;padding:2px 8px;border-radius:10px;font-size:11px}} .cf{{font-size:10px;color:#888;margin-top:3px}}
 .cust{{max-width:280px;color:#0f172a}} .rs{{max-width:200px;font-size:11px;color:#555}} .full{{max-width:180px;font-size:11px;color:#999}}
 audio{{height:30px}}
 .rl button{{margin:2px;padding:6px 10px;font-size:13px;font-weight:600;cursor:pointer;border:1px solid #bbb;border-radius:5px;background:#fff}}
 .rl button:hover{{background:#e2e8f0}} .rl button.on{{background:#0f172a;color:#fff;border-color:#0f172a}}
 tr.lbl-fair{{background:#e8f5e9;box-shadow:inset 5px 0 #388E3C}} tr.lbl-simple_vm{{background:#fff3e0;box-shadow:inset 5px 0 #F57C00}}
 tr.lbl-smart_vm{{background:#fce4ec;box-shadow:inset 5px 0 #C2185B}} tr.lbl-no_contact{{background:#eceff1;box-shadow:inset 5px 0 #90A4AE}}
</style></head><body>
<header><b>🧠 Call Review v2 — {date}</b> &nbsp; {len(rows)} calls &nbsp;|&nbsp; class from CUSTOMER words (bot script stripped)
 &nbsp;<button onclick="exp()" style="padding:5px 10px">⬇ Export my labels (<b id="cnt">0</b>)</button>
 <div class="legend">Click a relabel button to mark a call: <b>F</b>=Fair &nbsp; <b>Si</b>=Simple VM &nbsp; <b>Sm</b>=Smart VM &nbsp; <b>N</b>=No-contact &nbsp;— the row tints + button highlights when set. Each row has a stable <b>#ID</b> + phone + time you can quote.</div></header>
<div class="bar"><button onclick="filt('all')" style="border-color:#333"><b>all ({len(rows)})</b></button> {chips}</div>
<input id="q" placeholder="search ID / phone / customer text / transcript…" oninput="qf(this.value)">
<table id="t"><thead><tr><th>ID</th><th>phone / time</th><th>v2 class</th><th>platform</th><th>customer words</th><th>why</th><th>full transcript</th><th>dur</th><th>listen</th><th>relabel</th></tr></thead>
<tbody>{body}</tbody></table>
<script>
 const my={{}};
 function lbl(b,c){{const tr=b.closest('tr');
   tr.classList.remove('lbl-fair','lbl-simple_vm','lbl-smart_vm','lbl-no_contact');tr.classList.add('lbl-'+c);
   tr.querySelectorAll('.rl button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
   const d=tr.dataset;
   my[d.id]={{id:+d.id,phone:d.phone,time:d.time,file:d.file,platform:d.platform,v2_label:d.v2,my_label:c}};
   document.getElementById('cnt').textContent=Object.keys(my).length;}}
 function filt(c){{for(const tr of document.querySelectorAll('#t tbody tr'))tr.style.display=(c==='all'||tr.dataset.class===c)?'':'none';}}
 function qf(v){{v=v.toLowerCase();for(const tr of document.querySelectorAll('#t tbody tr')){{const t=(tr.dataset.id+' '+tr.dataset.phone+' '+tr.children[4].textContent+' '+tr.children[6].textContent).toLowerCase();tr.style.display=t.includes(v)?'':'none';}}}}
 function exp(){{const arr=Object.values(my).sort((a,b)=>a.id-b.id);
   const b=new Blob([JSON.stringify(arr,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='my_labels_v2_{date}.json';a.click();}}
</script></body></html>"""
    name = "review_v2_shareable.html" if args.audio == "remote" else "review_v2.html"
    path = os.path.join(DATA, "audio", date, name)
    open(path, "w").write(out)
    note = " (audio streams from public URLs — send this single file)" if args.audio == "remote" else ""
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows){note}")


if __name__ == "__main__":
    main()

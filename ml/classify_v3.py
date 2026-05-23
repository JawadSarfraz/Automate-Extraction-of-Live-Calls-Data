"""Classifier v3 — supervised model over customer-text embeddings + acoustics.

The first learned classifier (v1=acoustic rules, v2=text rules were both hand-
tuned). v3 fuses three transparent feature blocks per call and lets a model learn
the combination:

  A. MiniLM sentence embedding of the CUSTOMER residue   (384-d, from embed.py)
  B. acoustic DSP features                                (11,   from classify_v1)
  C. v2 lexical cue counts on the residue                 (7: vm/abuse/stall/engage
                                                              hits, max-repeat, word
                                                              count, age-stated)

Supervision = the platform's weak category label (no human labels yet). This is a
BOOTSTRAP: weak labels follow the platform's taxonomy, which differs from ours
(e.g. abuse=DNC there, simple_vm here), so treat the metrics as "how well we
reproduce the platform" — not ground truth. The pipeline is built so human labels
(my_labels_v2_*.json) can later override/augment the targets and lift it past v2.

Honest evaluation: 5-fold stratified out-of-fold (OOF) predictions, so no call is
scored by a model that trained on it. We compare v3 (OOF) head-to-head with the
rule-based v2 against the same weak labels, focusing on macro-F1 and Fair recall
(the cost-sensitive priority: never hang up on a real customer).

Usage:
  .venv/bin/python ml/classify_v3.py --date 2026-05-14
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from classify_v2 import VOICEMAIL, ABUSE, STALL, ENGAGE, hits, max_repeat, AGE_RE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

ACOUSTIC_KEYS = [
    "duration", "speech_ratio", "n_speech_segments", "longest_monologue",
    "longest_silence", "beep_score", "spectral_flatness", "bg_ratio",
    "centroid_hz", "rms_db",
]
CUE_NAMES = ["cue_vm", "cue_abuse", "cue_stall", "cue_engage", "cue_maxrep", "cue_wc", "cue_age"]
CLASSES = ["fair", "no_contact", "simple_vm", "smart_vm"]   # sorted-ish; set explicitly


def lexical_cues(cust: str) -> list[float]:
    c = cust.strip()
    rep_c, _ = max_repeat(c)
    return [
        float(len(hits(c, VOICEMAIL))),
        float(len(hits(c, ABUSE))),
        float(len(hits(c, STALL))),
        float(len(hits(c, ENGAGE))),
        float(rep_c),
        float(len(c.split())),
        1.0 if AGE_RE.search(c) else 0.0,
    ]


def build_features(date: str):
    """Return files[], X[n, d], feat_names[], and aligned weak_label / v2_label / category / cust."""
    v1 = {r["file"]: r for r in json.load(open(os.path.join(DATA, "analysis", f"classified_{date}.json")))}
    v2 = {r["file"]: r for r in json.load(open(os.path.join(DATA, "analysis", f"classified_v2_{date}.json")))}
    npz = np.load(os.path.join(DATA, "analysis", f"cust_emb_{date}.npz"), allow_pickle=True)
    emb_files = list(npz["files"])
    emb = npz["emb"]
    cust_arr = list(npz["cust"])
    emb_idx = {f: i for i, f in enumerate(emb_files)}

    files, rows_emb, rows_aco, rows_cue = [], [], [], []
    weak, v2lab, cats, custs = [], [], [], []
    for f in emb_files:
        r1 = v1.get(f)
        if r1 is None:
            continue
        files.append(f)
        rows_emb.append(emb[emb_idx[f]])
        rows_aco.append([float(r1.get(k, 0.0) or 0.0) for k in ACOUSTIC_KEYS] + [1.0 if r1.get("beep") else 0.0])
        cust = cust_arr[emb_idx[f]]
        rows_cue.append(lexical_cues(cust))
        weak.append(r1.get("weak_label", ""))
        v2lab.append(v2.get(f, {}).get("v2_label", ""))
        cats.append(r1.get("category", ""))
        custs.append(cust)

    A = np.array(rows_emb, dtype=np.float32)               # (n, 384) normalized
    B = np.array(rows_aco, dtype=np.float32)               # (n, 11) acoustic (+beep flag)
    C = np.array(rows_cue, dtype=np.float32)               # (n, 7) lexical cues
    feat_names = ([f"emb_{i}" for i in range(A.shape[1])]
                  + ACOUSTIC_KEYS + ["beep"] + CUE_NAMES)
    X = np.hstack([A, B, C])
    return files, X, feat_names, np.array(weak), np.array(v2lab), np.array(cats), custs, A.shape[1]


def per_class_f1(y_true, y_pred, labels):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {lab: (p[i], r[i], f[i], int(s[i])) for i, lab in enumerate(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    date = args.date

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    import joblib

    files, X, feat_names, weak, v2lab, cats, custs, n_emb = build_features(date)
    print(f"[v3] {len(files)} calls | features: {n_emb} emb + {X.shape[1]-n_emb} (acoustic+cues) = {X.shape[1]}")

    usable = np.array([w in CLASSES for w in weak])
    Xl, yl = X[usable], weak[usable]
    print(f"[v3] usable weak labels: {usable.sum()} / {len(files)}")
    from collections import Counter
    print("[v3] label counts:", dict(Counter(yl)))

    # Scale acoustic+cue block only; embeddings are already L2-normalized.
    # (StandardScaler on all is fine for linear models; we scale the tail block.)
    def make_model(kind):
        if kind == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
            )
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_depth=None,
            l2_regularization=1.0, class_weight="balanced", random_state=args.seed,
        )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    labels_sorted = sorted(set(yl))

    results = {}
    for kind in ("logreg", "hgb"):
        oof = cross_val_predict(make_model(kind), Xl, yl, cv=skf, n_jobs=-1)
        macro = f1_score(yl, oof, average="macro")
        results[kind] = (macro, oof)
        print(f"[v3] {kind}: OOF macro-F1 = {macro:.3f}")

    best = max(results, key=lambda k: results[k][0])
    macro, oof = results[best]
    print(f"\n[v3] BEST model = {best}  (OOF macro-F1 {macro:.3f})\n")

    # ---- v3 (OOF) report ----
    print("=" * 64)
    print(f"v3 ({best}) — out-of-fold report vs platform weak labels")
    print("=" * 64)
    print(classification_report(yl, oof, labels=labels_sorted, digits=3, zero_division=0))
    print("Confusion (rows=weak truth, cols=v3 pred):  " + "  ".join(labels_sorted))
    cm = confusion_matrix(yl, oof, labels=labels_sorted)
    for i, lab in enumerate(labels_sorted):
        print(f"  {lab:11} " + "  ".join(f"{v:5d}" for v in cm[i]))

    # ---- head-to-head vs v2 on the same labeled subset ----
    v2_sub = v2lab[usable]
    # v2 emits 'uncertain' too; map it out of label set for fair comparison (counts as wrong class).
    v3_f1 = per_class_f1(yl, oof, labels_sorted)
    v2_f1 = per_class_f1(yl, v2_sub, labels_sorted)
    v2_macro = f1_score(yl, v2_sub, average="macro", labels=labels_sorted)
    print("\n" + "=" * 64)
    print("HEAD-TO-HEAD vs weak labels   (F1 per class | recall in parens)")
    print("=" * 64)
    print(f"  {'class':11} {'v2_F1':>8} {'v3_F1':>8}   {'v2_rec':>7} {'v3_rec':>7}")
    for lab in labels_sorted:
        print(f"  {lab:11} {v2_f1[lab][2]:8.3f} {v3_f1[lab][2]:8.3f}   "
              f"{v2_f1[lab][1]:7.3f} {v3_f1[lab][1]:7.3f}")
    print(f"  {'MACRO-F1':11} {v2_macro:8.3f} {macro:8.3f}")
    print(f"\n  Fair recall (protect real customers):  v2={v2_f1['fair'][1]:.3f}  v3={v3_f1['fair'][1]:.3f}")

    # ---- fit final model on all labeled, predict ALL calls ----
    final = make_model(best)
    final.fit(Xl, yl)
    classes_ = list(final.classes_)
    proba_all = final.predict_proba(X)
    pred_all = np.array([classes_[i] for i in proba_all.argmax(1)])
    conf_all = proba_all.max(1)

    # For labeled rows, prefer the honest OOF label; keep final-model label for the rest.
    oof_full = pred_all.copy()
    oof_full[usable] = oof
    is_oof = usable.copy()

    out_dir = os.path.join(DATA, "analysis")
    rows = []
    for i, f in enumerate(files):
        row = {
            "file": f,
            "category": cats[i],
            "weak_label": weak[i],
            "v2_label": v2lab[i],
            "v3_label": oof_full[i],
            "v3_conf": round(float(conf_all[i]), 3),
            "v3_is_oof": bool(is_oof[i]),
            "agree_v2": bool(oof_full[i] == v2lab[i]),
            "customer_text": custs[i][:200],
        }
        for c in classes_:
            row[f"p_{c}"] = round(float(proba_all[i][classes_.index(c)]), 3)
        rows.append(row)

    with open(os.path.join(out_dir, f"classified_v3_{date}.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(out_dir, f"classified_v3_{date}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    model_path = os.path.join(out_dir, f"v3_model_{date}.joblib")
    joblib.dump({"model": final, "classes": classes_, "feat_names": feat_names,
                 "n_emb": n_emb, "best_kind": best}, model_path)

    print(f"\n[v3] full-day label distribution ({len(rows)} calls):")
    for k, v in Counter(r["v3_label"] for r in rows).most_common():
        print(f"  {k:12} {v:5} ({100*v/len(rows):.1f}%)")
    print(f"\n[v3] agreement with v2 (all calls): "
          f"{100*np.mean([r['agree_v2'] for r in rows]):.1f}%")
    print(f"[v3] wrote classified_v3_{date}.csv/.json and {os.path.basename(model_path)}")


if __name__ == "__main__":
    main()

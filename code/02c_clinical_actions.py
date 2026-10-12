"""Reclassification expressed as the clinical action implied by the FAST category (v14, clinical framing).

Same-day FAST is the reference. Rule-out (<=0.35) implies routine follow-up without further fibrosis work-up;
indeterminate or rule-in implies further assessment; rule-in (>=0.67) additionally implies hepatology referral
for at-risk steatohepatitis. Counts and patient-cluster bootstrap intervals use the same paired comparisons as 01/02b.
"""
from pathlib import Path
import importlib.util
import io
import sys

import pandas as pd

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r", H/"02b_rcv_threshold.py")
_stdout, sys.stdout = sys.stdout, io.StringIO()
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
sys.stdout = _stdout
m, a = r.m, r.a.copy()
up, dn = r.rcv(r.CVA_PRIMARY)
a["flag"] = r.flag(a, up, dn)

ref, new = a.reference_category, a.category
a["false_reassurance"] = ref.ne("rule_out") & new.eq("rule_out")        # same-day needs assessment, dated AST says rule-out
a["added_workup"] = ref.eq("rule_out") & new.ne("rule_out")              # same-day rule-out, dated AST says assess
a["lost_rule_in"] = ref.eq("rule_in") & new.ne("rule_in")
a["gained_rule_in"] = ref.ne("rule_in") & new.eq("rule_in")
a["crossed_035"] = ref.eq("rule_out") != new.eq("rule_out")
a["crossed_067"] = ref.eq("rule_in") != new.eq("rule_in")

rows = []
for direction, d in [("all", a), ("earlier", a[a.direction.eq("earlier")]), ("later", a[a.direction.eq("later")])]:
    for label, sub, col in [
        ("false_reassurance_among_sameday_assess", d[d.reference_category.ne("rule_out")], "false_reassurance"),
        ("added_workup_among_sameday_ruleout", d[d.reference_category.eq("rule_out")], "added_workup"),
        ("lost_rule_in_among_sameday_rule_in", d[d.reference_category.eq("rule_in")], "lost_rule_in"),
        ("gained_rule_in_among_sameday_not_rule_in", d[d.reference_category.ne("rule_in")], "gained_rule_in"),
        ("crossed_035_among_reclassified", d[d.reclassified], "crossed_035"),
        ("crossed_067_among_reclassified", d[d.reclassified], "crossed_067"),
        ("flagged_among_false_reassurance", d[d.false_reassurance], "flag"),
        ("flagged_among_lost_rule_in", d[d.lost_rule_in], "flag"),
        ("flagged_among_added_workup", d[d.added_workup], "flag"),
        ("flagged_among_gained_rule_in", d[d.gained_rule_in], "flag"),
    ]:
        rows.append(r.prop(sub, col, label, direction=direction))
res = pd.DataFrame(rows)
res.to_csv(m.TABLES/"25_clinical_action_changes.csv", index=False)
print(res[["label", "direction", "n", "patients", "events", "proportion", "ci_low", "ci_high"]].round(3).to_string())

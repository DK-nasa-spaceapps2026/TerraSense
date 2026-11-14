"""Quick sanity check: python smoke_test.py"""
import json
from app import app

c = app.test_client()
r = c.post("/api/field/analyze", json={"field": {"bbox": [-93.6045, 41.5955, -93.5955, 41.6045]},
                                       "preferences": {"priority": "water_saving"}})
assert r.status_code == 200, r.data
d = r.get_json()
print("field", d["field_id"], "zones", d["grid"]["n_zones"], "recommended", d["recommended_scenario"])
for s in d["scenarios"]:
    print(s["id"], s["score"], json.dumps(s["metrics"]))
print(json.dumps(d["guidance"], indent=1)[:900])
t = c.get(f"/api/field/timeline?field_id={d['field_id']}&scenario=status_quo").get_json()
from collections import Counter
for m in t["months"]:
    print(m["label"], m["phase"][:1], m["climate"]["tmax_c"], m["climate"]["precip_mm"], m["field"],
          dict(Counter(x["status"] for x in m["cells"])))

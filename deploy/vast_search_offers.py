#!/usr/bin/env python3
"""Read-only Vast offer search with explicit columns (the installed CLI sends select_cols=['*'], which the API
rejects). Usage: vastai-python deploy/vast_search_offers.py '<query>'"""

import contextlib, io, json, sys
from vastai.api.client import VastClient
from vastai.cli.main import main

COLUMNS = [
    "id",
    "machine_id",
    "gpu_name",
    "num_gpus",
    "cpu_name",
    "cpu_cores_effective",
    "cpu_ram",
    "dph_total",
    "reliability2",
    "inet_down",
    "inet_up",
    "geolocation",
    "disk_space",
    "cuda_max_good",
    "gpu_ram",
]
orig = VastClient.put


def put(self, subpath, query_args=None, json_data=None, **kw):
    if subpath == "/search/asks/" and json_data and json_data.get("select_cols") == ["*"]:
        json_data = dict(json_data, select_cols=COLUMNS)
    return orig(self, subpath, query_args=query_args, json_data=json_data, **kw)


VastClient.put = put
sys.argv = [
    "vastai",
    "search",
    "offers",
    sys.argv[1],
    "--storage",
    "250",
    "--limit",
    "30",
    "--order",
    "dph_total",
    "--raw",
]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
for r in json.loads(buf.getvalue()):
    print(
        f"{r['id']:>10} {r.get('num_gpus')}x{str(r.get('gpu_name')):13s} {r.get('gpu_ram', 0) / 1024:4.0f}GB cores {r.get('cpu_cores_effective'):6.1f} "
        f"ram {r.get('cpu_ram', 0) / 1024:5.0f}GB ${r.get('dph_total'):.3f}/h rel {r.get('reliability2'):.3f} cuda {r.get('cuda_max_good')} "
        f"net {r.get('inet_down'):6.0f} {str(r.get('cpu_name'))[:24]:24s} {r.get('geolocation')}"
    )

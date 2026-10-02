#!/usr/bin/env python3
"""Read-only fresh offer search using installed Vast CLI auth and explicit columns.

The installed CLI sends select_cols=['*'] to /search/asks/, which the current API
rejects. This process-local adapter changes that one request field only and never
prints authentication material or modifies the installed CLI.
"""

import contextlib
import io
import json
import sys
from vastai.api.client import VastClient
from vastai.cli.main import main

COLUMNS = [
    "id",
    "machine_id",
    "gpu_name",
    "num_gpus",
    "dph_total",
    "dph_base",
    "storage_cost",
    "inet_down_cost",
    "inet_up_cost",
    "reliability2",
    "geolocation",
    "rentable",
    "rented",
    "is_bid",
]
original_put = VastClient.put


def fresh_put(self, subpath, query_args=None, json_data=None, **kwargs):
    if subpath == "/search/asks/" and json_data and json_data.get("select_cols") == ["*"]:
        json_data = dict(json_data, select_cols=COLUMNS)
    return original_put(self, subpath, query_args=query_args, json_data=json_data, **kwargs)


VastClient.put = fresh_put
sys.argv = [
    "vastai",
    "search",
    "offers",
    "reliability>0.99 num_gpus=1 gpu_name=RTX_4090 inet_down>500 disk_space>40 inet_up_cost<0.03 inet_down_cost<0.03",
    "--new",
    "--storage",
    "40",
    "--limit",
    "40",
    "--order",
    "dph_total",
    "--raw",
]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
rows = json.loads(buffer.getvalue())
if not isinstance(rows, list):
    raise ValueError("Fresh offer endpoint did not return a list")
print(json.dumps([{key: row.get(key) for key in COLUMNS} for row in rows]))

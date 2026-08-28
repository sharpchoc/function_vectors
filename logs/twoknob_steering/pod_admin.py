#!/usr/bin/env python3
"""RunPod admin helper for this study: stock / create / status / list / terminate.

Usage: pod_admin.py stock|create|status <id>|list|terminate <id>
Reads the write key from ~/.runpod_write_key for create/terminate (export line),
falls back to env RUNPOD_API_KEY for read ops. Never prints keys.
"""
import json
import os
import sys
from pathlib import Path

import requests

URL = "https://api.runpod.io/graphql"
MATS = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDaOhA7MKQBpF3bexz2O5pBfZxtCNyCzuPbTNqJmYY4z MATS"


def key(write=False):
    if write:
        for line in Path("~/.runpod_write_key").expanduser().read_text().splitlines():
            if "RUNPOD_API_KEY" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise SystemExit("no key in ~/.runpod_write_key")
    return os.environ["RUNPOD_API_KEY"]


def gql(query, write=False):
    r = requests.post(URL, headers={"Authorization": f"Bearer {key(write)}"},
                      json={"query": query}, timeout=60)
    r.raise_for_status()
    return r.json()


cmd = sys.argv[1]
if cmd == "stock":
    d = gql("query { dataCenters { id gpuAvailability(input:{}) "
            "{ available stockStatus gpuTypeId } } }")
    for dc in d["data"]["dataCenters"]:
        if dc["id"] == "EU-RO-1":
            for g in dc["gpuAvailability"]:
                if g["available"]:
                    print(g["gpuTypeId"], g["stockStatus"])
elif cmd == "create":
    cpu = Path("~/.ssh/runpod_gpu.pub").expanduser().read_text().strip()
    val = json.dumps(MATS + "\n" + cpu)
    m = ('mutation { podFindAndDeployOnDemand(input:{ cloudType: SECURE, gpuCount: 1, '
         'gpuTypeId: "NVIDIA RTX PRO 4500 Blackwell", name: "fv-twoknob-steering", '
         'imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", '
         'ports: "22/tcp", '
         'dataCenterId: "EU-RO-1", networkVolumeId: "v5gswxsj8s", '
         'volumeMountPath: "/workspace", containerDiskInGb: 40, '
         'env: [{key: "PUBLIC_KEY", value: ' + val + '}] '
         '}) { id name desiredStatus imageName costPerHr networkVolumeId } }')
    print(json.dumps(gql(m, write=True), indent=2))
elif cmd == "status":
    pid = sys.argv[2]
    d = gql('query { pod(input:{podId:"%s"}) { desiredStatus runtime { uptimeInSeconds '
            'ports { ip isIpPublic publicPort privatePort type } gpus { id } } } }' % pid)
    print(json.dumps(d, indent=2))
elif cmd == "list":
    d = gql("query { myself { pods { id name desiredStatus costPerHr } } }")
    print(json.dumps(d, indent=2))
elif cmd == "terminate":
    pid = sys.argv[2]
    own = Path(__file__).parent / "own_pod_id"
    assert own.exists() and own.read_text().strip() == pid, \
        "refusing: not the pod id recorded by this session"
    print(json.dumps(gql('mutation { podTerminate(input:{podId:"%s"}) }' % pid, write=True)))
else:
    raise SystemExit("unknown cmd")

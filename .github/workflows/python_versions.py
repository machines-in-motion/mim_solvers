# Python versions to test: the stable ones actions/setup-python has for this
# runner's Ubuntu, that crocoddyl and proxsuite have Linux wheels for.
import json
import os
import platform
import re
import urllib.request


def get(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


release = platform.freedesktop_os_release()["VERSION_ID"]
manifest = get(
    "https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json"
)
versions = {
    ".".join(r["version"].split(".")[:2])
    for r in manifest
    if r["stable"]
    and any(
        f["platform"] == "linux"
        and f["arch"] == "x64"
        and f.get("platform_version") == release
        for f in r["files"]
    )
}
for package in ("crocoddyl", "proxsuite"):
    wheels = set()
    for f in get(f"https://pypi.org/pypi/{package}/json")["urls"]:
        m = re.search(r"-cp3(\d+)-cp3\d+-manylinux\w*x86_64", f["filename"])
        if m:
            wheels.add(f"3.{m.group(1)}")
    versions &= wheels
matrix = json.dumps(sorted(versions, key=lambda v: int(v.split(".")[1])))
print(f"Ubuntu {release}: {matrix}")
with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as out:
    out.write(f"python={matrix}\n")

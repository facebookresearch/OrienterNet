# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
from pathlib import Path
from typing import Dict

import urllib3
from libfb.py.certpathpicker.cert_path_picker import get_client_credential_paths

from .. import logger
from ..utils.geo import BoundaryBox

FWDPROXY_PORT = 8082
FWDPROXY_HOSTNAME = "https://fwdproxy"
FB_CA_BUNDLE = "/var/facebook/rootcanal/ca.pem"


def get_osm(
    boundary_box: BoundaryBox, cache_file_path: Path, overwrite: bool = False
) -> str:
    if not overwrite and cache_file_path.is_file():
        with cache_file_path.open() as fp:
            return json.load(fp)

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    content: bytes = get_web_data(
        "https://api.openstreetmap.org/api/0.6/map.json",
        {"bbox": f"{left},{bottom},{right},{top}"},
    )

    content_str = content.decode("utf-8")
    if content_str.startswith("You requested too many nodes"):
        raise ValueError(content_str)

    with cache_file_path.open("bw+") as fp:
        fp.write(content)
    return json.loads(content_str)


def get_web_data(address: str, parameters: Dict[str, str]) -> bytes:
    logger.info("Getting %s...", address)

    thrift_cert, thrift_key = get_client_credential_paths()
    fwdproxy_url = f"{FWDPROXY_HOSTNAME}:{FWDPROXY_PORT}"
    with urllib3.proxy_from_url(
        fwdproxy_url,
        ca_certs=FB_CA_BUNDLE,
        cert_file=thrift_cert,
        key_file=thrift_key,
        use_forwarding_for_https=True,
    ) as proxy:
        result = proxy.request("GET", address, parameters, timeout=10)

    return result.data

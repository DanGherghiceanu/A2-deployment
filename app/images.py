"""Turning user input (a URL or uploaded bytes) into a PIL image, safely.

The /predict endpoint fetches URLs on the caller's behalf. Once the service is
publicly reachable that makes it a request proxy, so anything it fetches needs
checking first: scheme, resolved address, response size, and time spent.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from . import config

log = logging.getLogger("xray.images")

# Pillow refuses absurdly large images by default; keep that guard on and set a
# ceiling well above a real chest film but below a decompression bomb.
Image.MAX_IMAGE_PIXELS = 80_000_000


class ImageIntakeError(ValueError):
    """User-facing problem with the supplied image. Maps to HTTP 400."""


def _is_public_address(host: str) -> bool:
    """Resolve `host` and confirm every address it maps to is publicly routable."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ImageIntakeError(f"Could not resolve host '{host}'.") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return True


def load_image_from_url(url: str) -> Image.Image:
    """Download an image over http(s) and decode it."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ImageIntakeError("image_url must start with http:// or https://")
    if not parsed.hostname:
        raise ImageIntakeError("image_url is missing a hostname.")

    if config.BLOCK_PRIVATE_ADDRESSES and not _is_public_address(parsed.hostname):
        raise ImageIntakeError(
            "image_url resolves to a private or internal address, which this "
            "service will not fetch. Upload the file instead."
        )

    try:
        response = requests.get(
            url,
            timeout=config.IMAGE_FETCH_TIMEOUT,
            stream=True,
            headers={"User-Agent": "chest-xray-classifier/1.0"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise ImageIntakeError(
            f"Fetching image_url timed out after {config.IMAGE_FETCH_TIMEOUT:.0f}s."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise ImageIntakeError(
            f"image_url returned HTTP {response.status_code}."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ImageIntakeError(f"Could not fetch image_url: {exc}") from exc

    declared = response.headers.get("Content-Length")
    if declared and declared.isdigit() and int(declared) > config.MAX_IMAGE_BYTES:
        raise ImageIntakeError(
            f"Image is larger than the {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )

    # Read with a hard cap rather than trusting Content-Length, which a server
    # can understate or omit entirely.
    buffer = BytesIO()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        buffer.write(chunk)
        if buffer.tell() > config.MAX_IMAGE_BYTES:
            response.close()
            raise ImageIntakeError(
                f"Image is larger than the {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
            )

    return decode_image(buffer.getvalue())


def decode_image(raw: bytes) -> Image.Image:
    """Decode raw bytes into a PIL image, with a clear error if they aren't one."""
    if not raw:
        raise ImageIntakeError("The uploaded file is empty.")
    if len(raw) > config.MAX_IMAGE_BYTES:
        raise ImageIntakeError(
            f"Image is larger than the {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )

    try:
        image = Image.open(BytesIO(raw))
        image.load()  # force decode now, so failures surface here and not mid-predict
    except UnidentifiedImageError as exc:
        raise ImageIntakeError(
            "That file isn't a readable image. Supported formats: JPEG, PNG."
        ) from exc
    except Exception as exc:  # truncated or malformed file
        raise ImageIntakeError(f"Could not decode the image: {exc}") from exc

    return image

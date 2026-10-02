"""Turn a real estate listing link into reel inputs.

The importer fetches a listing page, extracts property details from its
structured data (JSON-LD, Open Graph) and visible text, lets the configured
LLM fill gaps and write a short voiceover script, and stores the listing
photos as local materials so the normal "local" video pipeline can use them.
"""

import hashlib
import io
import ipaddress
import json
import os
import re
import socket
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Callable, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.services import llm, material_upload

MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = material_upload.MAX_IMAGE_MATERIAL_UPLOAD_BYTES
MAX_REDIRECTS = 5
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_MAX_IMAGES = 10
MIN_IMAGE_DIMENSION = 480
MAX_PAGE_TEXT_FOR_LLM = 6000
MAX_IMAGE_CANDIDATES = 60

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

_LISTING_TYPES = {
    "realestatelisting",
    "residence",
    "house",
    "singlefamilyresidence",
    "apartment",
    "apartmentcomplex",
    "accommodation",
    "room",
    "suite",
    "place",
    "product",
    "offer",
    "landform",
    "lodgingbusiness",
}

_IMAGE_URL_PATTERN = re.compile(
    r"https?:\\?/\\?/[^\s\"'<>()]+?\.(?:jpe?g|png|webp)(?:\?[^\s\"'<>()]*)?",
    re.IGNORECASE,
)
_SKIP_IMAGE_HINTS = (
    "logo",
    "icon",
    "sprite",
    "avatar",
    "favicon",
    "placeholder",
    "badge",
    "banner-ad",
    "profile",
    "agent-photo",
    "flag",
    "emoji",
    "tracking",
    "pixel",
)
_SKIP_TEXT_TAGS = {"script", "style", "noscript", "svg", "template", "head"}


class PropertyListingError(ValueError):
    """A listing link could not be turned into reel inputs."""


@dataclass
class PropertyListing:
    url: str
    title: str = ""
    price: str = ""
    location: str = ""
    property_type: str = ""
    bedrooms: str = ""
    bathrooms: str = ""
    area: str = ""
    features: List[str] = field(default_factory=list)
    description: str = ""
    image_urls: List[str] = field(default_factory=list)
    page_text: str = ""

    def details(self) -> dict:
        """Return the user-facing fields, without page text or image URLs."""
        data = asdict(self)
        data.pop("page_text", None)
        data.pop("image_urls", None)
        return {key: value for key, value in data.items() if value}

    def subject(self) -> str:
        parts = [self.title or self.property_type or "Property for sale"]
        if self.location and self.location not in parts[0]:
            parts.append(self.location)
        return " - ".join(parts)[:200]


# -----------------------------------------------------------------------------
# Fetching
# -----------------------------------------------------------------------------


def _ensure_public_http_url(url: str) -> str:
    """Reject non-HTTP URLs and hosts that resolve to private networks.

    The WebUI can be exposed on a LAN, so a listing link must not become a way
    to make the server request internal services.
    """
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PropertyListingError("Please enter a valid http(s) listing link")

    try:
        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(
                parsed.hostname, parsed.port or None, proto=socket.IPPROTO_TCP
            )
        }
    except socket.gaierror as exc:
        raise PropertyListingError(
            f"Could not resolve host: {parsed.hostname}"
        ) from exc

    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not ip.is_global:
            raise PropertyListingError(
                f"Links to private or local addresses are not allowed: {parsed.hostname}"
            )
    return url


@dataclass
class _FetchResult:
    url: str
    status_code: int
    content_type: str
    content: bytes
    encoding: Optional[str]

    def text(self) -> str:
        encoding = self.encoding
        if not encoding:
            match = re.search(
                rb"<meta[^>]+charset=[\"']?([\w-]+)", self.content[:4096], re.I
            )
            encoding = match.group(1).decode("ascii") if match else "utf-8"
        return self.content.decode(encoding, errors="replace")


def _get(url: str, max_bytes: int, accept: Optional[str] = None) -> _FetchResult:
    """GET a public URL, validating every redirect hop and capping the size."""
    headers = dict(_REQUEST_HEADERS)
    if accept:
        headers["Accept"] = accept

    current_url = url
    with requests.Session() as session:
        for _ in range(MAX_REDIRECTS + 1):
            _ensure_public_http_url(current_url)
            with session.get(
                current_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location", "")
                    if not location:
                        raise PropertyListingError("Redirect without a target")
                    current_url = urljoin(current_url, location)
                    continue

                content = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise PropertyListingError("The response is too large")
                content_type = response.headers.get("Content-Type", "")
                charset = re.search(r"charset=([\w-]+)", content_type, re.I)
                return _FetchResult(
                    url=current_url,
                    status_code=response.status_code,
                    content_type=content_type.lower(),
                    content=bytes(content),
                    encoding=charset.group(1) if charset else None,
                )

    raise PropertyListingError("Too many redirects")


def fetch_listing_html(url: str) -> tuple[str, str]:
    """Return (html, final_url) for a listing page."""
    url = _ensure_public_http_url(url)
    try:
        response = _get(url, MAX_PAGE_BYTES)
    except requests.RequestException as exc:
        raise PropertyListingError(f"Could not open the listing link: {exc}") from exc

    if response.status_code in {401, 403, 429}:
        raise PropertyListingError(
            "The listing site blocked the request "
            f"(HTTP {response.status_code}). Try another listing link, or "
            "upload the photos manually."
        )
    if response.status_code >= 400:
        raise PropertyListingError(
            f"The listing page returned HTTP {response.status_code}"
        )
    if response.content_type and not any(
        kind in response.content_type for kind in ("html", "xml")
    ):
        raise PropertyListingError("The link does not point to a web page")

    try:
        return response.text(), response.url
    except LookupError:
        return response.content.decode("utf-8", errors="replace"), response.url


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------


class _ListingHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.json_ld: List[str] = []
        self.image_urls: List[str] = []
        self.text_parts: List[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_json_ld = False
        self._json_ld_buffer: List[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if tag == "meta":
            name = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).lower()
            content = attributes.get("content", "").strip()
            if name and content:
                if name in {"og:image", "og:image:url", "twitter:image"}:
                    self.image_urls.append(content)
                self.meta.setdefault(name, content)
        elif tag == "title":
            self._in_title = True
        elif tag == "script":
            if "ld+json" in attributes.get("type", "").lower():
                self._in_json_ld = True
                self._json_ld_buffer = []
        elif tag in {"img", "source"}:
            for key in ("data-src", "data-lazy-src", "data-original", "src"):
                value = attributes.get(key, "").strip()
                if value and not value.startswith("data:"):
                    self.image_urls.append(value)
                    break
            srcset = attributes.get("srcset") or attributes.get("data-srcset")
            if srcset:
                best = _largest_srcset_candidate(srcset)
                if best:
                    self.image_urls.append(best)

        if tag in _SKIP_TEXT_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._in_json_ld:
            self.json_ld.append("".join(self._json_ld_buffer))
            self._in_json_ld = False
        if tag in _SKIP_TEXT_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._in_json_ld:
            self._json_ld_buffer.append(data)
            return
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.text_parts.append(text)


def _largest_srcset_candidate(srcset: str) -> str:
    best_url, best_width = "", -1
    for candidate in srcset.split(","):
        pieces = candidate.strip().split()
        if not pieces:
            continue
        width = 0
        if len(pieces) > 1:
            match = re.match(r"(\d+)", pieces[1])
            width = int(match.group(1)) if match else 0
        if width > best_width:
            best_url, best_width = pieces[0], width
    return best_url


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value) -> str:
    if isinstance(value, dict):
        for key in ("name", "value", "@value", "text"):
            if value.get(key):
                return _text(value[key])
        return ""
    if isinstance(value, list):
        return ", ".join(filter(None, (_text(item) for item in value)))
    if value is None:
        return ""
    return " ".join(str(value).split())


def _iter_json_ld_nodes(raw_blocks: Iterable[str]):
    for raw in raw_blocks:
        try:
            data = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                yield node
                for key in ("@graph", "mainEntity", "itemOffered", "offers", "about"):
                    if key in node:
                        stack.extend(_as_list(node[key]))


def _node_types(node: dict) -> set[str]:
    return {str(item).lower() for item in _as_list(node.get("@type"))}


def _format_address(value) -> str:
    if isinstance(value, dict):
        parts = [
            _text(value.get(key))
            for key in (
                "streetAddress",
                "addressLocality",
                "addressRegion",
                "addressCountry",
            )
        ]
        return ", ".join(part for part in parts if part)
    return _text(value)


def _format_price(offer: dict) -> str:
    price = _text(offer.get("price") or offer.get("lowPrice"))
    if not price:
        spec = offer.get("priceSpecification")
        if isinstance(spec, dict):
            price = _text(spec.get("price"))
            offer = {**offer, "priceCurrency": spec.get("priceCurrency")}
    if not price:
        return ""
    currency = _text(offer.get("priceCurrency"))
    if re.fullmatch(r"\d+(\.\d+)?", price):
        number = float(price)
        price = f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"
    return f"{currency} {price}".strip()


def _apply_json_ld(listing: PropertyListing, nodes: Iterable[dict]) -> None:
    for node in nodes:
        types = _node_types(node)
        has_listing_fields = any(
            key in node
            for key in (
                "numberOfRooms",
                "numberOfBedrooms",
                "numberOfBathroomsTotal",
                "floorSize",
                "address",
            )
        )
        if not (types & _LISTING_TYPES or has_listing_fields):
            continue

        if not listing.title and node.get("name") and "offer" not in types:
            listing.title = _text(node.get("name"))
        if not listing.description and node.get("description"):
            listing.description = _text(node.get("description"))
        if not listing.location:
            address = node.get("address")
            if address is None and isinstance(node.get("location"), dict):
                address = node["location"].get("address")
            listing.location = _format_address(address)
        if not listing.bedrooms:
            listing.bedrooms = _text(
                node.get("numberOfBedrooms") or node.get("numberOfRooms")
            )
        if not listing.bathrooms:
            listing.bathrooms = _text(
                node.get("numberOfBathroomsTotal")
                or node.get("numberOfFullBathrooms")
                or node.get("numberOfBathrooms")
            )
        if not listing.area and isinstance(node.get("floorSize"), dict):
            size = node["floorSize"]
            listing.area = " ".join(
                part
                for part in (
                    _text(size.get("value")),
                    _text(size.get("unitText") or size.get("unitCode")),
                )
                if part
            )
        if not listing.property_type:
            specific = [
                item
                for item in _as_list(node.get("@type"))
                if str(item).lower()
                not in {"product", "offer", "place", "realestatelisting"}
            ]
            if specific:
                listing.property_type = _text(specific[0])
        if not listing.price:
            for offer in _as_list(node.get("offers")) + (
                [node] if "offer" in types else []
            ):
                if isinstance(offer, dict):
                    listing.price = _format_price(offer)
                    if listing.price:
                        break
        for feature in _as_list(node.get("amenityFeature")):
            name = _text(feature)
            if name and name not in listing.features:
                listing.features.append(name)
        for image in _as_list(node.get("image")) + _as_list(node.get("photo")):
            url = (
                (image.get("url") or image.get("contentUrl"))
                if isinstance(image, dict)
                else image
            )
            if isinstance(url, str) and url:
                listing.image_urls.append(url)


def _is_probable_photo(url: str) -> bool:
    lowered = url.lower()
    if lowered.endswith((".svg", ".gif", ".ico")):
        return False
    path = urlparse(lowered).path
    return not any(hint in path for hint in _SKIP_IMAGE_HINTS)


def _normalize_image_urls(urls: Iterable[str], base_url: str) -> List[str]:
    seen = set()
    result = []
    for raw in urls:
        url = (raw or "").strip().replace("\\/", "/").replace("&amp;", "&")
        if not url or url.startswith("data:"):
            continue
        url = urljoin(base_url, url)
        if urlparse(url).scheme not in {"http", "https"} or not _is_probable_photo(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        result.append(url)
        if len(result) >= MAX_IMAGE_CANDIDATES:
            break
    return result


def parse_listing_html(html: str, url: str) -> PropertyListing:
    """Extract whatever the page itself declares about the property."""
    parser = _ListingHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # malformed markup should not abort the import
        logger.warning(f"listing HTML parse was incomplete: {exc}")

    listing = PropertyListing(url=url)
    _apply_json_ld(listing, _iter_json_ld_nodes(parser.json_ld))

    meta = parser.meta
    listing.title = (
        listing.title or meta.get("og:title") or " ".join(parser.title.split())
    )
    listing.description = (
        listing.description
        or meta.get("og:description")
        or meta.get("description")
        or ""
    )
    if not listing.price:
        amount = meta.get("product:price:amount") or meta.get("og:price:amount")
        if amount:
            currency = meta.get("product:price:currency") or meta.get(
                "og:price:currency", ""
            )
            listing.price = f"{currency} {amount}".strip()

    # Structured images come first; many sites only expose the gallery inside
    # inline JSON (Next.js/Nuxt state), so fall back to URLs found anywhere.
    raw_images = listing.image_urls + parser.image_urls
    raw_images += [match.group(0) for match in _IMAGE_URL_PATTERN.finditer(html)]
    listing.image_urls = _normalize_image_urls(raw_images, url)

    listing.page_text = " ".join(parser.text_parts)[: MAX_PAGE_TEXT_FOR_LLM * 2]
    return listing


# -----------------------------------------------------------------------------
# LLM enrichment and script
# -----------------------------------------------------------------------------

_DETAIL_FIELDS = (
    "title",
    "price",
    "location",
    "property_type",
    "bedrooms",
    "bathrooms",
    "area",
    "features",
    "description",
)


def _call_llm(prompt: str, app_config=None) -> str:
    if app_config is None:
        response = llm._generate_response(prompt=prompt)
    else:
        response = llm._generate_response(prompt=prompt, app_config=app_config)
    if not isinstance(response, str) or response.startswith("Error: "):
        raise PropertyListingError(str(response) or "The AI model returned no answer")
    return response


def _parse_json_object(text: str) -> dict:
    text = llm._strip_code_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def enrich_listing_with_llm(
    listing: PropertyListing, app_config=None
) -> PropertyListing:
    """Ask the LLM to read the page text and fill in missing details."""
    known = {key: getattr(listing, key) for key in _DETAIL_FIELDS}
    prompt = f"""You extract real estate listing details from a web page.

Return ONLY a JSON object with these keys (strings unless noted):
title, price (with currency), location, property_type, bedrooms, bathrooms,
area (with unit), features (array of up to 8 short strings), description
(2-3 sentences, factual).

Rules:
- Use only facts present in the page. Use "" (or [] for features) when unknown.
- Keep values short. Do not invent prices, sizes or room counts.
- Prefer the values already detected below unless the page clearly contradicts them.

Already detected:
{json.dumps(known, ensure_ascii=False)}

Page URL: {listing.url}
Page text:
{listing.page_text[:MAX_PAGE_TEXT_FOR_LLM]}
"""
    try:
        data = _parse_json_object(_call_llm(prompt, app_config))
    except PropertyListingError as exc:
        logger.warning(f"listing detail extraction failed, using page data only: {exc}")
        return listing

    for key in _DETAIL_FIELDS:
        value = data.get(key)
        if key == "features":
            if isinstance(value, list) and not listing.features:
                listing.features = [_text(item) for item in value if _text(item)][:8]
            continue
        value = _text(value)
        if value and not getattr(listing, key):
            setattr(listing, key, value[:500])
    return listing


def build_reel_script_prompt(
    listing: PropertyListing, language: str = "", extra_requirements: str = ""
) -> str:
    details = json.dumps(listing.details(), ensure_ascii=False, indent=2)
    prompt = f"""You are a real estate marketing copywriter writing the voiceover
for a 30-45 second vertical property reel (Instagram Reels / TikTok / Shorts).

# Rules
- Start with a one-line hook that makes viewers stop scrolling.
- Mention the key facts: property type, location, size, bedrooms, bathrooms,
  standout features and price when available.
- End with a short call to action (for example: book a viewing, send a message).
- Use only the facts below. Never invent numbers, prices or features.
- 80-120 words, written to be spoken aloud, short sentences.
- Plain text only: no markdown, no emojis, no hashtags, no headings, no
  stage directions, no quotes around the script.
- Separate paragraphs with a blank line.

# Property details
{details}
"""
    if language:
        prompt += f"\n# Language\nWrite the script in: {language}\n"
    if extra_requirements:
        prompt += f"\n# Additional requirements\n{extra_requirements.strip()[:2000]}\n"
    return prompt


def _clean_script(text: str) -> str:
    text = llm._strip_code_fence(text)
    text = text.replace("*", "").replace("#", "")
    text = re.sub(r"\[.*?\]", "", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip().strip('"').strip()


def generate_reel_script(
    listing: PropertyListing,
    language: str = "",
    extra_requirements: str = "",
    app_config=None,
) -> str:
    prompt = build_reel_script_prompt(listing, language, extra_requirements)
    last_error = ""
    for attempt in range(llm._max_retries):
        try:
            script = _clean_script(_call_llm(prompt, app_config))
            if script:
                return script
        except PropertyListingError as exc:
            last_error = str(exc)
            logger.warning(
                f"failed to generate property reel script (attempt {attempt + 1}): {exc}"
            )
    raise PropertyListingError(
        last_error or "The AI model could not write a script for this listing"
    )


# -----------------------------------------------------------------------------
# Photos
# -----------------------------------------------------------------------------


def _download_image(url: str) -> Optional[Image.Image]:
    try:
        response = _get(
            url, MAX_IMAGE_BYTES, accept="image/avif,image/webp,image/*,*/*;q=0.8"
        )
    except (PropertyListingError, requests.RequestException) as exc:
        logger.debug(f"skip listing image {url}: {exc}")
        return None
    if response.status_code >= 400:
        return None
    try:
        image = Image.open(io.BytesIO(response.content))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return None
    return image


def download_listing_images(
    image_urls: Iterable[str], max_images: int = DEFAULT_MAX_IMAGES
) -> List[str]:
    """Download listing photos and store them as validated local materials.

    Returns stored file names inside the local materials directory. Small
    images (icons, thumbnails) and duplicates are skipped.
    """
    stored = []
    seen_hashes = set()
    for url in image_urls:
        if len(stored) >= max_images:
            break
        image = _download_image(url)
        if image is None:
            continue
        width, height = image.size
        if min(width, height) < MIN_IMAGE_DIMENSION:
            continue

        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=92)
        digest = hashlib.sha256(output.getvalue()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        output.seek(0)

        try:
            stored.append(
                material_upload.save_material_upload("listing-photo.jpg", output)
            )
        except (
            material_upload.MaterialUploadError,
            material_upload.MaterialServiceError,
        ) as exc:
            logger.warning(f"skip listing image {url}: {exc}")
    return stored


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


@dataclass
class PropertyReelDraft:
    listing: PropertyListing
    video_subject: str
    video_script: str
    material_files: List[str]


def create_reel_draft(
    url: str,
    language: str = "",
    extra_requirements: str = "",
    max_images: int = DEFAULT_MAX_IMAGES,
    app_config=None,
    run_llm: Optional[Callable[[str, Callable], object]] = None,
) -> PropertyReelDraft:
    """Fetch a listing and return the subject, script and stored photos.

    ``run_llm(operation_name, operation)`` lets callers control how LLM calls
    obtain their config (the WebUI wraps them in its config snapshot lock);
    ``operation`` receives the app config to use. Network downloads run
    outside of it so they never hold that lock.
    """
    if run_llm is None:

        def run_llm(_operation_name, operation):
            return operation(app_config)

    html, final_url = fetch_listing_html(url)
    listing = parse_listing_html(html, final_url)
    listing = run_llm(
        "property_listing_details",
        lambda config_snapshot: enrich_listing_with_llm(
            listing, app_config=config_snapshot
        ),
    )
    if not (listing.title or listing.description or listing.page_text):
        raise PropertyListingError("No property details were found on that page")

    material_files = download_listing_images(listing.image_urls, max_images=max_images)
    if not material_files:
        raise PropertyListingError(
            "No usable property photos were found on that page "
            f"(photos must be at least {MIN_IMAGE_DIMENSION}px). "
            "You can upload photos manually with the Local file source."
        )

    try:
        script = run_llm(
            "property_reel_script",
            lambda config_snapshot: generate_reel_script(
                listing,
                language=language,
                extra_requirements=extra_requirements,
                app_config=config_snapshot,
            ),
        )
    except Exception:
        remove_material_files(material_files)
        raise
    logger.info(
        f"property reel draft created: url={final_url}, "
        f"photos={len(material_files)}, details={listing.details()}"
    )
    return PropertyReelDraft(
        listing=listing,
        video_subject=listing.subject(),
        video_script=script,
        material_files=material_files,
    )


def remove_material_files(material_files: Iterable[str]) -> None:
    material_dir = material_upload.uploaded_material_dir(create=False)
    for name in material_files:
        try:
            os.remove(os.path.join(material_dir, os.path.basename(name)))
        except OSError as exc:
            logger.warning(f"failed to remove listing photo {name}: {exc}")

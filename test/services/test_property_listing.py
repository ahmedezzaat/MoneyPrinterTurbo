import io
import json
import os
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from app.services import material_upload, property_listing


def _public_dns(host, port, proto=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, proto, "", ("93.184.216.34", 443))]


def _image_bytes(size=(800, 600), color="red", image_format="JPEG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color=color).save(output, format=image_format)
    return output.getvalue()


def _fake_response(status=200, content=b"", headers=None, redirect_to=None):
    response = MagicMock()
    response.status_code = status
    response.headers = dict(headers or {})
    if redirect_to:
        response.headers["Location"] = redirect_to
    response.is_redirect = bool(redirect_to)
    response.is_permanent_redirect = False
    response.iter_content.return_value = [content]
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


LISTING_HTML = """
<html>
<head>
  <title>Ignored page title</title>
  <meta property="og:image" content="https://cdn.example.com/photos/front.jpg">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@graph": [
      {"@type": "Organization", "name": "Acme Realty",
       "logo": "https://cdn.example.com/logo.png"},
      {
        "@type": ["Product", "SingleFamilyResidence"],
        "name": "Sunny 3BR Villa with Pool",
        "description": "Bright villa close to the beach.",
        "numberOfBedrooms": 3,
        "numberOfBathroomsTotal": 2,
        "floorSize": {"value": 210, "unitText": "sqm"},
        "address": {"streetAddress": "12 Palm St",
                    "addressLocality": "New Cairo", "addressCountry": "EG"},
        "amenityFeature": [{"name": "Private pool"}, {"name": "Garden"}],
        "image": ["https://cdn.example.com/photos/pool.jpg",
                  {"url": "/photos/kitchen.jpg"}],
        "offers": {"@type": "Offer", "price": "4500000", "priceCurrency": "EGP"}
      }
    ]
  }
  </script>
</head>
<body>
  <img src="https://cdn.example.com/static/logo.svg">
  <img src="/assets/icons/bed-icon.png">
  <img data-src="https://cdn.example.com/photos/garden.webp" src="data:image/gif;base64,AAA">
  <img srcset="https://cdn.example.com/photos/living-400.jpg 400w,
               https://cdn.example.com/photos/living-1600.jpg 1600w">
  <script>window.__STATE__={"gallery":["https:\\/\\/cdn.example.com\\/photos\\/bedroom.jpg"]}</script>
  <h1>Sunny 3BR Villa</h1>
  <p>Walking distance to schools.</p>
</body>
</html>
"""


class TestParseListingHtml(unittest.TestCase):
    def test_reads_json_ld_details_and_photos(self):
        listing = property_listing.parse_listing_html(
            LISTING_HTML, "https://homes.example.com/listing/1"
        )

        self.assertEqual(listing.title, "Sunny 3BR Villa with Pool")
        self.assertEqual(listing.description, "Bright villa close to the beach.")
        self.assertEqual(listing.price, "EGP 4,500,000")
        self.assertEqual(listing.location, "12 Palm St, New Cairo, EG")
        self.assertEqual(listing.bedrooms, "3")
        self.assertEqual(listing.bathrooms, "2")
        self.assertEqual(listing.area, "210 sqm")
        self.assertEqual(listing.property_type, "SingleFamilyResidence")
        self.assertEqual(listing.features, ["Private pool", "Garden"])
        self.assertIn("Walking distance to schools.", listing.page_text)
        self.assertNotIn("window.__STATE__", listing.page_text)

        self.assertEqual(
            listing.image_urls,
            [
                "https://cdn.example.com/photos/pool.jpg",
                "https://homes.example.com/photos/kitchen.jpg",
                "https://cdn.example.com/photos/front.jpg",
                "https://cdn.example.com/photos/garden.webp",
                "https://cdn.example.com/photos/living-1600.jpg",
                "https://cdn.example.com/photos/living-400.jpg",
                "https://cdn.example.com/photos/bedroom.jpg",
            ],
        )

    def test_falls_back_to_open_graph_and_title(self):
        html = """
        <html><head><title> Flat in   Dubai Marina </title>
        <meta name="description" content="Two bedroom flat with sea view.">
        <meta property="product:price:amount" content="1200000">
        <meta property="product:price:currency" content="AED">
        </head><body><p>Sea view</p></body></html>
        """
        listing = property_listing.parse_listing_html(html, "https://x.example/1")

        self.assertEqual(listing.title, "Flat in Dubai Marina")
        self.assertEqual(listing.description, "Two bedroom flat with sea view.")
        self.assertEqual(listing.price, "AED 1200000")
        self.assertEqual(listing.image_urls, [])

    def test_invalid_json_ld_is_ignored(self):
        html = '<script type="application/ld+json">{not json</script><title>T</title>'
        listing = property_listing.parse_listing_html(html, "https://x.example/1")
        self.assertEqual(listing.title, "T")

    def test_subject_combines_title_and_location(self):
        listing = property_listing.PropertyListing(
            url="https://x.example", title="Modern Loft", location="Zamalek, Cairo"
        )
        self.assertEqual(listing.subject(), "Modern Loft - Zamalek, Cairo")
        self.assertNotIn("page_text", listing.details())


class TestUrlSafety(unittest.TestCase):
    def test_rejects_non_http_urls(self):
        for url in ("", "ftp://example.com/x", "file:///etc/passwd", "example.com"):
            with self.subTest(url=url):
                with self.assertRaises(property_listing.PropertyListingError):
                    property_listing._ensure_public_http_url(url)

    def test_rejects_private_addresses(self):
        for address in (
            "127.0.0.1",
            "10.0.0.5",
            "192.168.1.10",
            "169.254.169.254",
            "::1",
        ):
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            with self.subTest(address=address):
                with patch.object(
                    property_listing.socket,
                    "getaddrinfo",
                    return_value=[(family, socket.SOCK_STREAM, 6, "", (address, 80))],
                ):
                    with self.assertRaises(property_listing.PropertyListingError):
                        property_listing._ensure_public_http_url(
                            "http://internal.test/"
                        )

    def test_accepts_public_address(self):
        with patch.object(property_listing.socket, "getaddrinfo", _public_dns):
            self.assertEqual(
                property_listing._ensure_public_http_url(
                    " https://homes.example.com/1 "
                ),
                "https://homes.example.com/1",
            )

    def test_redirect_to_private_address_is_blocked(self):
        def fake_dns(host, port, proto=0):
            address = "127.0.0.1" if host == "localhost" else "93.184.216.34"
            return [(socket.AF_INET, socket.SOCK_STREAM, proto, "", (address, 80))]

        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = _fake_response(
            status=302, redirect_to="http://localhost/admin"
        )
        with (
            patch.object(property_listing.socket, "getaddrinfo", fake_dns),
            patch.object(property_listing.requests, "Session", return_value=session),
        ):
            with self.assertRaises(property_listing.PropertyListingError):
                property_listing.fetch_listing_html("https://homes.example.com/1")
        session.get.assert_called_once()


class TestFetchListingHtml(unittest.TestCase):
    def _fetch(self, response):
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = response
        with (
            patch.object(property_listing.socket, "getaddrinfo", _public_dns),
            patch.object(property_listing.requests, "Session", return_value=session),
        ):
            return property_listing.fetch_listing_html("https://homes.example.com/1")

    def test_returns_decoded_html(self):
        html, final_url = self._fetch(
            _fake_response(
                content="<title>Villa – شقة</title>".encode("utf-8"),
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        )
        self.assertEqual(html, "<title>Villa – شقة</title>")
        self.assertEqual(final_url, "https://homes.example.com/1")

    def test_blocked_site_gives_helpful_error(self):
        with self.assertRaisesRegex(property_listing.PropertyListingError, "blocked"):
            self._fetch(
                _fake_response(status=403, headers={"Content-Type": "text/html"})
            )

    def test_rejects_non_html_content(self):
        with self.assertRaisesRegex(property_listing.PropertyListingError, "web page"):
            self._fetch(_fake_response(headers={"Content-Type": "application/pdf"}))

    def test_rejects_oversized_page(self):
        response = _fake_response(headers={"Content-Type": "text/html"})
        response.iter_content.return_value = [
            b"x" * (property_listing.MAX_PAGE_BYTES + 1)
        ]
        with self.assertRaisesRegex(property_listing.PropertyListingError, "too large"):
            self._fetch(response)


class TestLlmSteps(unittest.TestCase):
    def test_enrich_fills_only_missing_fields(self):
        listing = property_listing.PropertyListing(
            url="https://x.example", title="Detected title", page_text="Some text"
        )
        response = (
            "```json\n"
            + json.dumps(
                {
                    "title": "LLM title",
                    "price": "USD 300,000",
                    "bedrooms": 2,
                    "features": ["Balcony", "", "Parking"],
                }
            )
            + "\n```"
        )
        with patch.object(
            property_listing.llm, "_generate_response", return_value=response
        ) as generate:
            property_listing.enrich_listing_with_llm(listing, app_config={"k": "v"})

        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["app_config"], {"k": "v"})
        self.assertEqual(listing.title, "Detected title")
        self.assertEqual(listing.price, "USD 300,000")
        self.assertEqual(listing.bedrooms, "2")
        self.assertEqual(listing.features, ["Balcony", "Parking"])

    def test_enrich_keeps_page_data_when_llm_fails(self):
        listing = property_listing.PropertyListing(url="https://x.example", title="T")
        with patch.object(
            property_listing.llm, "_generate_response", return_value="Error: no key"
        ):
            result = property_listing.enrich_listing_with_llm(listing)
        self.assertIs(result, listing)
        self.assertEqual(listing.title, "T")

    def test_script_is_cleaned_and_uses_language(self):
        listing = property_listing.PropertyListing(
            url="https://x.example", title="Villa", price="EGP 4,500,000"
        )
        prompts = []

        def fake_generate(prompt):
            prompts.append(prompt)
            return '```\n"## Your dream **villa** awaits [music]\n\n\n\nBook a viewing."\n```'

        with patch.object(property_listing.llm, "_generate_response", fake_generate):
            script = property_listing.generate_reel_script(
                listing, language="ar-EG", extra_requirements="Mention the sea view"
            )

        self.assertEqual(script, "Your dream villa awaits\n\nBook a viewing.")
        self.assertIn("EGP 4,500,000", prompts[0])
        self.assertIn("ar-EG", prompts[0])
        self.assertIn("Mention the sea view", prompts[0])

    def test_script_failure_raises_after_retries(self):
        listing = property_listing.PropertyListing(url="https://x.example", title="V")
        with (
            patch.object(
                property_listing.llm, "_generate_response", return_value="Error: down"
            ) as generate,
            patch.object(property_listing.llm, "_max_retries", 2),
        ):
            with self.assertRaisesRegex(property_listing.PropertyListingError, "down"):
                property_listing.generate_reel_script(listing)
        self.assertEqual(generate.call_count, 2)


class TestDownloadListingImages(unittest.TestCase):
    def test_saves_large_unique_photos_as_local_materials(self):
        images = {
            "https://cdn.example.com/a.jpg": _image_bytes(color="red"),
            "https://cdn.example.com/a-copy.jpg": _image_bytes(color="red"),
            "https://cdn.example.com/thumb.jpg": _image_bytes(size=(200, 150)),
            "https://cdn.example.com/broken.jpg": b"not an image",
            "https://cdn.example.com/b.png": _image_bytes(
                color="blue", image_format="PNG"
            ),
            "https://cdn.example.com/c.jpg": _image_bytes(color="green"),
        }

        def fake_get(url, max_bytes, accept=None):
            return property_listing._FetchResult(
                url=url,
                status_code=200,
                content_type="image/jpeg",
                content=images[url],
                encoding=None,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(property_listing, "_get", side_effect=fake_get),
                patch.object(
                    material_upload, "uploaded_material_dir", return_value=temp_dir
                ),
            ):
                stored = property_listing.download_listing_images(
                    list(images), max_images=2
                )

            self.assertEqual(len(stored), 2)
            for name in stored:
                self.assertTrue(name.endswith(".jpg"))
                with Image.open(os.path.join(temp_dir, name)) as image:
                    self.assertEqual(image.size, (800, 600))
            self.assertEqual(sorted(os.listdir(temp_dir)), sorted(stored))


class TestCreateReelDraft(unittest.TestCase):
    def test_runs_llm_steps_through_runner_and_cleans_up_on_failure(self):
        listing = property_listing.PropertyListing(
            url="https://x.example", title="Villa", image_urls=["https://x/a.jpg"]
        )
        calls = []

        def runner(name, operation):
            calls.append(name)
            return operation({"snapshot": True})

        with (
            patch.object(
                property_listing,
                "fetch_listing_html",
                return_value=("<html></html>", "https://x.example"),
            ),
            patch.object(property_listing, "parse_listing_html", return_value=listing),
            patch.object(
                property_listing,
                "enrich_listing_with_llm",
                side_effect=lambda item, app_config: item,
            ),
            patch.object(
                property_listing, "download_listing_images", return_value=["p1.jpg"]
            ),
            patch.object(
                property_listing, "generate_reel_script", return_value="Script"
            ) as generate_script,
        ):
            draft = property_listing.create_reel_draft(
                "https://x.example", language="en-US", run_llm=runner
            )

        self.assertEqual(calls, ["property_listing_details", "property_reel_script"])
        self.assertEqual(
            generate_script.call_args.kwargs["app_config"], {"snapshot": True}
        )
        self.assertEqual(draft.video_subject, "Villa")
        self.assertEqual(draft.video_script, "Script")
        self.assertEqual(draft.material_files, ["p1.jpg"])

        with (
            patch.object(
                property_listing,
                "fetch_listing_html",
                return_value=("<html></html>", "https://x.example"),
            ),
            patch.object(property_listing, "parse_listing_html", return_value=listing),
            patch.object(
                property_listing,
                "enrich_listing_with_llm",
                side_effect=lambda item, app_config: item,
            ),
            patch.object(
                property_listing, "download_listing_images", return_value=["p1.jpg"]
            ),
            patch.object(
                property_listing,
                "generate_reel_script",
                side_effect=property_listing.PropertyListingError("no llm"),
            ),
            patch.object(property_listing, "remove_material_files") as remove_files,
        ):
            with self.assertRaises(property_listing.PropertyListingError):
                property_listing.create_reel_draft("https://x.example")
        remove_files.assert_called_once_with(["p1.jpg"])

    def test_requires_photos(self):
        listing = property_listing.PropertyListing(
            url="https://x.example", title="Villa"
        )
        with (
            patch.object(
                property_listing,
                "fetch_listing_html",
                return_value=("<html></html>", "https://x.example"),
            ),
            patch.object(property_listing, "parse_listing_html", return_value=listing),
            patch.object(
                property_listing,
                "enrich_listing_with_llm",
                side_effect=lambda item, app_config: item,
            ),
            patch.object(property_listing, "download_listing_images", return_value=[]),
        ):
            with self.assertRaisesRegex(
                property_listing.PropertyListingError, "photos"
            ):
                property_listing.create_reel_draft("https://x.example")


if __name__ == "__main__":
    unittest.main()

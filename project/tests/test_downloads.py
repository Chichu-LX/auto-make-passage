import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import httpx

from research_agents.data_materialization import materialize_dataset_payload
from research_agents.data_profiling import profile_dataset_payload
from research_agents.downloads import (
    DownloadPolicy,
    DownloadPolicyError,
    SafeHttpDownloader,
    validate_dataset_payload,
)
from research_agents.schemas import AgentName
from research_agents.storage import ArtifactStore


PUBLIC_RESOLVER = lambda hostname: ["93.184.216.34"]


class SafeDownloadTests(unittest.TestCase):
    def test_dta_profile_detects_representation_aliases_and_label_conflicts(self):
        payload = (
            b"compound_iso_smiles,target_key,target_sequence,affinity\n"
            b"CCO,target-a,AAAA,7.0\n"
            b"CCO,target-b,AAAA,8.0\n"
            b"CCC,target-c,BBBB,6.0\n"
        )
        profile = profile_dataset_payload(payload, "csv")

        target_profile = profile.entity_representation_profiles[0]
        self.assertEqual(target_profile.unique_identifier_count, 3)
        self.assertEqual(target_profile.unique_representation_count, 2)
        self.assertEqual(target_profile.shared_representation_group_count, 1)
        self.assertEqual(profile.repeated_exact_input_group_count, 1)
        self.assertEqual(profile.conflicting_exact_input_label_group_count, 1)
        self.assertEqual(profile.conflicting_exact_input_label_row_count, 2)
        self.assertTrue(profile.representation_audit_complete)
        self.assertTrue(
            any("identifier-disjoint" in note for note in profile.notes),
            profile.notes,
        )

    def test_generic_profile_detects_cross_domain_exact_input_conflicts(self):
        payload = (
            b"patient_id,site_id,image_embedding,outcome\n"
            b"p1,hospital-a,0.1|0.2,0\n"
            b"p2,hospital-a,0.1|0.2,1\n"
            b"p3,hospital-b,0.3|0.4,1\n"
        )
        profile = profile_dataset_payload(payload, "csv")

        subject_profile = profile.entity_representation_profiles[0]
        self.assertEqual(subject_profile.entity_role, "subject")
        self.assertEqual(subject_profile.shared_representation_group_count, 1)
        self.assertEqual(profile.repeated_exact_input_group_count, 1)
        self.assertEqual(profile.conflicting_exact_input_label_group_count, 1)
        self.assertEqual(profile.conflicting_exact_input_label_row_count, 2)
        self.assertTrue(profile.representation_audit_complete)

    def test_resolves_standard_citation_pdf_url_then_revalidates_pdf(self):
        calls = []

        def handler(request):
            calls.append(str(request.url))
            if request.url.path == "/article":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html; charset=utf-8"},
                    content=(
                        b'<html><head><meta name="citation_pdf_url" '
                        b'content="/article.pdf"></head></html>'
                    ),
                    request=request,
                )
            return httpx.Response(
                200,
                headers={"content-type": "application/pdf"},
                content=b"%PDF-1.7\nresolved pdf",
                request=request,
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        downloader = SafeHttpDownloader(client=client, resolver=PUBLIC_RESOLVER)
        resolved = downloader.resolve_citation_pdf_url("https://example.com/article")
        self.assertEqual(resolved, "https://example.com/article.pdf")
        payload = downloader.fetch(
            resolved,
            DownloadPolicy(
                max_bytes=100,
                allowed_content_types=frozenset({"application/pdf"}),
                require_pdf_signature=True,
            ),
        )
        self.assertTrue(payload.content.startswith(b"%PDF-"))
        self.assertEqual(
            calls,
            ["https://example.com/article", "https://example.com/article.pdf"],
        )
        client.close()

    def test_pmc_landing_page_maps_to_europe_pmc_pdf_render(self):
        downloader = SafeHttpDownloader(resolver=PUBLIC_RESOLVER)
        self.assertEqual(
            downloader.resolve_citation_pdf_url(
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC6129291/"
            ),
            "https://europepmc.org/articles/PMC6129291?pdf=render",
        )

    def test_downloads_pdf_after_mime_size_and_signature_checks(self):
        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "application/pdf", "content-length": "18"},
                content=b"%PDF-1.7\ntest pdf",
                request=request,
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        downloader = SafeHttpDownloader(client=client, resolver=PUBLIC_RESOLVER)
        payload = downloader.fetch(
            "https://example.com/paper.pdf",
            DownloadPolicy(
                max_bytes=100,
                allowed_content_types=frozenset({"application/pdf"}),
                require_pdf_signature=True,
            ),
        )
        self.assertTrue(payload.content.startswith(b"%PDF-"))
        self.assertEqual(payload.content_type, "application/pdf")
        client.close()

    def test_rejects_private_target_before_http_request(self):
        called = []

        def handler(request):
            called.append(request)
            return httpx.Response(200, content=b"unexpected", request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        downloader = SafeHttpDownloader(client=client, resolver=PUBLIC_RESOLVER)
        with self.assertRaisesRegex(DownloadPolicyError, "non-global"):
            downloader.fetch(
                "http://127.0.0.1/file.pdf",
                DownloadPolicy(
                    max_bytes=100,
                    allowed_content_types=frozenset({"application/pdf"}),
                ),
            )
        self.assertEqual(called, [])
        client.close()

    def test_revalidates_redirect_and_rejects_private_hop(self):
        calls = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(
                302,
                headers={"location": "http://169.254.169.254/latest/meta-data"},
                request=request,
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        downloader = SafeHttpDownloader(client=client, resolver=PUBLIC_RESOLVER)
        with self.assertRaisesRegex(DownloadPolicyError, "non-global"):
            downloader.fetch(
                "https://example.com/start",
                DownloadPolicy(
                    max_bytes=100,
                    allowed_content_types=frozenset({"application/pdf"}),
                ),
            )
        self.assertEqual(calls, ["https://example.com/start"])
        client.close()

    def test_rejects_stream_larger_than_policy(self):
        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                stream=httpx.ByteStream(b"x" * 101),
                request=request,
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        downloader = SafeHttpDownloader(client=client, resolver=PUBLIC_RESOLVER)
        with self.assertRaisesRegex(DownloadPolicyError, "streamed file size"):
            downloader.fetch(
                "https://example.com/data.txt",
                DownloadPolicy(
                    max_bytes=100,
                    allowed_content_types=frozenset({"text/plain"}),
                ),
            )
        client.close()

    def test_zip_path_traversal_is_rejected(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("../escape.csv", "a,b\n1,2\n")
        policy = DownloadPolicy(
            max_bytes=1024,
            allowed_content_types=frozenset({"application/zip"}),
        )
        with self.assertRaisesRegex(DownloadPolicyError, "path traversal"):
            validate_dataset_payload(output.getvalue(), "zip", policy)

    def test_valid_data_only_zip_passes(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("dataset/train.csv", "a,b\n1,2\n")
        policy = DownloadPolicy(
            max_bytes=1024,
            allowed_content_types=frozenset({"application/zip"}),
        )
        validate_dataset_payload(output.getvalue(), "zip", policy)

    def test_valid_zip_is_materialized_as_local_data_body(self):
        output = io.BytesIO()
        content = b"a,b\n1,2\n"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("dataset/train.csv", content)
        with tempfile.TemporaryDirectory() as directory:
            records = materialize_dataset_payload(
                output.getvalue(),
                "zip",
                artifacts=ArtifactStore(directory),
                agent=AgentName.DOWNLOAD_DATA,
                relative_root="data/test-project/datasets/example/task-token",
                max_bytes=1024,
            )
            target = (
                Path(directory)
                / "data/test-project/datasets/example/task-token/dataset/train.csv"
            )
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].size_bytes, len(content))


if __name__ == "__main__":
    unittest.main()

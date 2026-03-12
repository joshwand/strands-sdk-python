"""Tests for location source resolution in model validation utilities."""

from unittest.mock import MagicMock, patch

import pytest

from strands.models._validation import (
    _fetch_s3_bytes,
    _get_s3_client,
    _resolve_location_source,
)


class TestGetS3Client:
    """Tests for _get_s3_client lazy initialization."""

    def test_returns_boto3_s3_client(self):
        """Test that _get_s3_client returns a boto3 S3 client."""
        import strands.models._validation as mod

        original_client = mod._s3_client
        try:
            mod._s3_client = None
            with patch("boto3.client") as mock_boto3_client:
                mock_client = MagicMock()
                mock_boto3_client.return_value = mock_client
                result = _get_s3_client()
                mock_boto3_client.assert_called_once_with("s3")
                assert result is mock_client
        finally:
            mod._s3_client = original_client


class TestFetchS3Bytes:
    """Tests for _fetch_s3_bytes download function."""

    def setup_method(self):
        """Clear LRU cache before each test."""
        _fetch_s3_bytes.cache_clear()

    def test_valid_s3_uri(self):
        """Test downloading from a valid S3 URI."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"image-data"
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch("strands.models._validation._get_s3_client", return_value=mock_client):
            result = _fetch_s3_bytes("s3://my-bucket/path/to/image.png")

        assert result == b"image-data"
        mock_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="path/to/image.png")

    def test_s3_uri_with_bucket_owner(self):
        """Test downloading with bucket owner specified."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"data"
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch("strands.models._validation._get_s3_client", return_value=mock_client):
            result = _fetch_s3_bytes("s3://bucket/key.pdf", bucket_owner="123456789012")

        mock_client.get_object.assert_called_once_with(
            Bucket="bucket", Key="key.pdf", ExpectedBucketOwner="123456789012"
        )
        assert result == b"data"

    def test_invalid_uri_not_s3(self):
        """Test that non-S3 URIs raise ValueError."""
        with pytest.raises(ValueError, match="invalid S3 URI, expected"):
            _fetch_s3_bytes("https://example.com/image.png")

    def test_invalid_uri_missing_key(self):
        """Test that S3 URIs without a key raise ValueError."""
        with pytest.raises(ValueError, match="missing key"):
            _fetch_s3_bytes("s3://bucket-only")

    def test_caching_avoids_redundant_downloads(self):
        """Test that repeated calls with the same URI use the cache."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"cached-data"
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch("strands.models._validation._get_s3_client", return_value=mock_client):
            result1 = _fetch_s3_bytes("s3://bucket/cached-key.png")
            result2 = _fetch_s3_bytes("s3://bucket/cached-key.png")

        assert result1 == b"cached-data"
        assert result2 == b"cached-data"
        # Should only call S3 once due to caching
        mock_client.get_object.assert_called_once()

    def test_different_uris_not_cached_together(self):
        """Test that different URIs are fetched independently."""
        mock_body1 = MagicMock()
        mock_body1.read.return_value = b"data1"
        mock_body2 = MagicMock()
        mock_body2.read.return_value = b"data2"
        mock_client = MagicMock()
        mock_client.get_object.side_effect = [{"Body": mock_body1}, {"Body": mock_body2}]

        with patch("strands.models._validation._get_s3_client", return_value=mock_client):
            result1 = _fetch_s3_bytes("s3://bucket/key1.png")
            result2 = _fetch_s3_bytes("s3://bucket/key2.png")

        assert result1 == b"data1"
        assert result2 == b"data2"
        assert mock_client.get_object.call_count == 2


class TestResolveLocationSource:
    """Tests for _resolve_location_source function."""

    def setup_method(self):
        """Clear LRU cache before each test."""
        _fetch_s3_bytes.cache_clear()

    def test_resolve_image_s3_location(self):
        """Test resolving an image with S3 location source."""
        content = {
            "image": {
                "format": "png",
                "source": {"location": {"type": "s3", "uri": "s3://bucket/image.png"}},
            }
        }

        with patch("strands.models._validation._fetch_s3_bytes", return_value=b"image-bytes"):
            result = _resolve_location_source(content)

        assert "image" in result
        assert result["image"]["format"] == "png"
        assert result["image"]["source"]["bytes"] == b"image-bytes"
        assert "location" not in result["image"]["source"]

    def test_resolve_document_s3_location(self):
        """Test resolving a document with S3 location source."""
        content = {
            "document": {
                "format": "pdf",
                "name": "doc.pdf",
                "source": {"location": {"type": "s3", "uri": "s3://bucket/doc.pdf"}},
            }
        }

        with patch("strands.models._validation._fetch_s3_bytes", return_value=b"pdf-bytes"):
            result = _resolve_location_source(content)

        assert "document" in result
        assert result["document"]["format"] == "pdf"
        assert result["document"]["name"] == "doc.pdf"
        assert result["document"]["source"]["bytes"] == b"pdf-bytes"
        assert "location" not in result["document"]["source"]

    def test_resolve_video_s3_location(self):
        """Test resolving a video with S3 location source."""
        content = {
            "video": {
                "format": "mp4",
                "source": {"location": {"type": "s3", "uri": "s3://bucket/video.mp4"}},
            }
        }

        with patch("strands.models._validation._fetch_s3_bytes", return_value=b"video-bytes"):
            result = _resolve_location_source(content)

        assert "video" in result
        assert result["video"]["format"] == "mp4"
        assert result["video"]["source"]["bytes"] == b"video-bytes"
        assert "location" not in result["video"]["source"]

    def test_resolve_with_bucket_owner(self):
        """Test that bucketOwner is passed to _fetch_s3_bytes."""
        content = {
            "image": {
                "format": "jpeg",
                "source": {
                    "location": {"type": "s3", "uri": "s3://bucket/img.jpg", "bucketOwner": "123456789012"}
                },
            }
        }

        with patch("strands.models._validation._fetch_s3_bytes", return_value=b"data") as mock_fetch:
            _resolve_location_source(content)
            mock_fetch.assert_called_once_with("s3://bucket/img.jpg", "123456789012")

    def test_unsupported_location_type_raises(self):
        """Test that non-S3 location types raise ValueError."""
        content = {
            "image": {
                "format": "png",
                "source": {"location": {"type": "gcs", "uri": "gs://bucket/image.png"}},
            }
        }

        with pytest.raises(ValueError, match="unsupported location source type"):
            _resolve_location_source(content)

    def test_no_location_source_raises(self):
        """Test that content without a location source raises ValueError."""
        content = {"text": "hello"}

        with pytest.raises(ValueError, match="no resolvable location source"):
            _resolve_location_source(content)

    def test_does_not_mutate_original_content(self):
        """Test that the original content block is not modified."""
        original_location = {"type": "s3", "uri": "s3://bucket/image.png"}
        original_source = {"location": original_location}
        original_image = {"format": "png", "source": original_source}
        content = {"image": original_image}

        with patch("strands.models._validation._fetch_s3_bytes", return_value=b"bytes"):
            result = _resolve_location_source(content)

        # Original should be unchanged
        assert "location" in content["image"]["source"]
        assert "bytes" not in content["image"]["source"]
        # Result should have bytes, not location
        assert "bytes" in result["image"]["source"]
        assert "location" not in result["image"]["source"]
        # Should be different objects
        assert result is not content
        assert result["image"] is not content["image"]
        assert result["image"]["source"] is not content["image"]["source"]

    def test_preserves_other_source_fields(self):
        """Test that fields other than 'location' in source are preserved."""
        content = {
            "document": {
                "format": "pdf",
                "name": "test.pdf",
                "source": {
                    "location": {"type": "s3", "uri": "s3://bucket/test.pdf"},
                },
                "citations": {"enabled": True},
            }
        }

        with patch("strands.models._validation._fetch_s3_bytes", return_value=b"data"):
            result = _resolve_location_source(content)

        assert result["document"]["citations"] == {"enabled": True}
        assert result["document"]["name"] == "test.pdf"
        assert result["document"]["format"] == "pdf"

"""Configuration validation utilities for model providers."""

import functools
import logging
import threading
import warnings
from collections.abc import Mapping
from typing import Any, cast

from typing_extensions import get_type_hints

from ..types.content import ContentBlock
from ..types.media import S3Location
from ..types.tools import ToolChoice

logger = logging.getLogger(__name__)


def validate_config_keys(config_dict: Mapping[str, Any], config_class: type) -> None:
    """Validate that config keys match the TypedDict fields.

    Args:
        config_dict: Dictionary of configuration parameters
        config_class: TypedDict class to validate against
    """
    valid_keys = set(get_type_hints(config_class).keys())
    provided_keys = set(config_dict.keys())
    invalid_keys = provided_keys - valid_keys

    if invalid_keys:
        warnings.warn(
            f"Invalid configuration parameters: {sorted(invalid_keys)}."
            f"\nValid parameters are: {sorted(valid_keys)}."
            f"\n"
            f"\nSee https://github.com/strands-agents/sdk-python/issues/815",
            stacklevel=4,
        )


def warn_on_tool_choice_not_supported(tool_choice: ToolChoice | None) -> None:
    """Emits a warning if a tool choice is provided but not supported by the provider.

    Args:
        tool_choice: the tool_choice provided to the provider
    """
    if tool_choice:
        warnings.warn(
            "A ToolChoice was provided to this provider but is not supported and will be ignored",
            stacklevel=4,
        )


def _has_location_source(content: ContentBlock) -> bool:
    """Check if a content block contains a location source.

    Providers need to explicitly define an implementation to support content locations.

    Args:
        content: Content block to check.

    Returns:
        True if the content block contains an location source, False otherwise.
    """
    if "image" in content:
        return "location" in content["image"].get("source", {})
    if "document" in content:
        return "location" in content["document"].get("source", {})
    if "video" in content:
        return "location" in content["video"].get("source", {})
    return False


_s3_client_lock = threading.Lock()
_s3_client: Any = None


def _get_s3_client() -> Any:
    """Get a lazily-initialized S3 client for resolving location sources.

    Uses double-checked locking for thread safety.

    Returns:
        A boto3 S3 client instance.
    """
    global _s3_client
    if _s3_client is None:
        with _s3_client_lock:
            if _s3_client is None:
                import boto3

                _s3_client = boto3.client("s3")
    return _s3_client


@functools.lru_cache(maxsize=256)
def _fetch_s3_bytes(uri: str, bucket_owner: str | None = None) -> bytes:
    """Download content from an S3 URI with LRU caching.

    Results are cached by URI to avoid redundant S3 downloads when the same
    content is referenced multiple times across conversation turns.

    Args:
        uri: S3 URI in the format ``s3://bucket/key``.
        bucket_owner: Optional expected bucket owner account ID.

    Returns:
        The raw bytes of the S3 object.

    Raises:
        ValueError: If the URI is not a valid S3 URI.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"uri=<{uri}> | invalid S3 URI, expected s3://bucket/key format")

    path = uri[5:]
    slash_idx = path.find("/")
    if slash_idx <= 0:
        raise ValueError(f"uri=<{uri}> | invalid S3 URI, missing key")

    bucket = path[:slash_idx]
    key = path[slash_idx + 1:]

    client = _get_s3_client()
    kwargs: dict[str, str] = {"Bucket": bucket, "Key": key}
    if bucket_owner:
        kwargs["ExpectedBucketOwner"] = bucket_owner

    logger.debug("uri=<%s> | fetching S3 object", uri)
    response = client.get_object(**kwargs)
    return response["Body"].read()  # type: ignore[no-any-return]


def _resolve_location_source(content: ContentBlock) -> ContentBlock:
    """Resolve a location source in a content block to bytes.

    Creates a **new** content block with the location replaced by downloaded bytes,
    leaving the original content block unmodified. This ensures session managers do not
    rewrite resolved bytes back to storage on every save.

    Downloads are cached via :func:`_fetch_s3_bytes` so repeated references to the
    same S3 URI across conversation turns do not trigger redundant fetches.

    Only S3 locations (``type: "s3"``) are currently supported.

    Args:
        content: Content block containing a location source.

    Returns:
        A new content block with bytes instead of the location source.

    Raises:
        ValueError: If the location source type is unsupported or cannot be resolved.
    """
    for media_type in ("image", "document", "video"):
        if media_type not in content:
            continue

        media = content[media_type]
        source = media.get("source", {})
        if "location" not in source:
            continue

        location = source["location"]
        location_type = location.get("type")
        if location_type != "s3":
            raise ValueError(f"location_type=<{location_type}> | unsupported location source type")

        s3_location = cast(S3Location, location)
        data = _fetch_s3_bytes(s3_location["uri"], s3_location.get("bucketOwner"))

        # Build new content block without mutating the original
        new_source = {k: v for k, v in source.items() if k != "location"}
        new_source["bytes"] = data
        new_media = dict(media)
        new_media["source"] = new_source
        new_content: dict[str, Any] = dict(content)
        new_content[media_type] = new_media
        return cast(ContentBlock, new_content)

    raise ValueError("no resolvable location source found in content block")

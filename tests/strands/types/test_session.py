import json
import unittest.mock
from uuid import uuid4

from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.state import AgentState
from strands.interrupt import _InterruptState
from strands.types.content import ContentBlock
from strands.types.media import DocumentContent, DocumentSource, ImageContent, ImageSource, VideoContent, VideoSource
from strands.types.session import (
    Session,
    SessionAgent,
    SessionMessage,
    SessionType,
    decode_bytes_values,
    encode_bytes_values,
)


def test_session_json_serializable():
    session = Session(session_id=str(uuid4()), session_type=SessionType.AGENT)
    # json dumps will fail if its not json serializable
    session_json_string = json.dumps(session.to_dict())
    loaded_session = Session.from_dict(json.loads(session_json_string))
    assert loaded_session is not None


def test_agent_json_serializable():
    agent = SessionAgent(
        agent_id=str(uuid4()), state={"foo": "bar"}, conversation_manager_state=NullConversationManager().get_state()
    )
    # json dumps will fail if its not json serializable
    agent_json_string = json.dumps(agent.to_dict())
    loaded_agent = SessionAgent.from_dict(json.loads(agent_json_string))
    assert loaded_agent is not None


def test_message_json_serializable():
    message = SessionMessage(message={"role": "user", "content": [{"text": "Hello!"}]}, message_id=0)
    # json dumps will fail if its not json serializable
    message_json_string = json.dumps(message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))
    assert loaded_message is not None


def test_bytes_encoding_decoding():
    # Test simple bytes
    test_bytes = b"Hello, world!"
    encoded = encode_bytes_values(test_bytes)
    assert isinstance(encoded, dict)
    assert encoded["__bytes_encoded__"] is True
    decoded = decode_bytes_values(encoded)
    assert decoded == test_bytes

    # Test nested structure with bytes
    test_data = {
        "text": "Hello",
        "binary": b"Binary data",
        "nested": {"more_binary": b"More binary data", "list_with_binary": [b"Item 1", "Text item", b"Item 3"]},
    }

    encoded = encode_bytes_values(test_data)
    # Verify it's JSON serializable
    json_str = json.dumps(encoded)
    # Deserialize and decode
    decoded = decode_bytes_values(json.loads(json_str))

    # Verify the decoded data matches the original
    assert decoded["text"] == test_data["text"]
    assert decoded["binary"] == test_data["binary"]
    assert decoded["nested"]["more_binary"] == test_data["nested"]["more_binary"]
    assert decoded["nested"]["list_with_binary"][0] == test_data["nested"]["list_with_binary"][0]
    assert decoded["nested"]["list_with_binary"][1] == test_data["nested"]["list_with_binary"][1]
    assert decoded["nested"]["list_with_binary"][2] == test_data["nested"]["list_with_binary"][2]


def test_session_message_with_bytes():
    # Create a message with bytes content
    message = {
        "role": "user",
        "content": [{"text": "Here is some binary data"}, {"binary_data": b"This is binary data"}],
    }

    # Create a SessionMessage
    session_message = SessionMessage.from_message(message, 0)

    # Verify it's JSON serializable
    message_json_string = json.dumps(session_message.to_dict())

    # Load it back
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    # Convert back to original message and verify
    original_message = loaded_message.to_message()

    assert original_message["role"] == message["role"]
    assert original_message["content"][0]["text"] == message["content"][0]["text"]
    assert original_message["content"][1]["binary_data"] == message["content"][1]["binary_data"]


def test_session_agent_from_agent():
    agent = unittest.mock.Mock()
    agent.agent_id = "a1"
    agent.conversation_manager = unittest.mock.Mock(get_state=lambda: {"test": "conversation"})
    agent.state = AgentState({"test": "state"})
    agent._interrupt_state = _InterruptState(interrupts={}, context={}, activated=False)

    tru_session_agent = SessionAgent.from_agent(agent)
    exp_session_agent = SessionAgent(
        agent_id="a1",
        conversation_manager_state={"test": "conversation"},
        state={"test": "state"},
        _internal_state={"interrupt_state": {"interrupts": {}, "context": {}, "activated": False}},
        created_at=unittest.mock.ANY,
        updated_at=unittest.mock.ANY,
    )
    assert tru_session_agent == exp_session_agent


def test_session_agent_initialize_internal_state():
    agent = unittest.mock.Mock()
    session_agent = SessionAgent(
        agent_id="a1",
        conversation_manager_state={},
        state={},
        _internal_state={"interrupt_state": {"interrupts": {}, "context": {"test": "init"}, "activated": False}},
    )

    session_agent.initialize_internal_state(agent)

    tru_interrupt_state = agent._interrupt_state
    exp_interrupt_state = _InterruptState(interrupts={}, context={"test": "init"}, activated=False)
    assert tru_interrupt_state == exp_interrupt_state


def test_session_message_with_image_bytes():
    """Test SessionMessage round-trip with ImageContent containing bytes."""
    image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    message = {
        "role": "user",
        "content": [
            ContentBlock(
                image=ImageContent(
                    format="png",
                    source=ImageSource(bytes=image_bytes),
                )
            )
        ],
    }

    session_message = SessionMessage.from_message(message, 0)
    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    original_message = loaded_message.to_message()
    assert original_message["content"][0]["image"]["source"]["bytes"] == image_bytes
    assert original_message["content"][0]["image"]["format"] == "png"


def test_session_message_with_document_bytes():
    """Test SessionMessage round-trip with DocumentContent containing bytes."""
    doc_bytes = b"%PDF-1.4 fake pdf content"
    message = {
        "role": "user",
        "content": [
            ContentBlock(
                document=DocumentContent(
                    format="pdf",
                    name="test.pdf",
                    source=DocumentSource(bytes=doc_bytes),
                )
            )
        ],
    }

    session_message = SessionMessage.from_message(message, 0)
    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    original_message = loaded_message.to_message()
    assert original_message["content"][0]["document"]["source"]["bytes"] == doc_bytes
    assert original_message["content"][0]["document"]["format"] == "pdf"
    assert original_message["content"][0]["document"]["name"] == "test.pdf"


def test_session_message_with_video_bytes():
    """Test SessionMessage round-trip with VideoContent containing bytes."""
    video_bytes = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00"
    message = {
        "role": "user",
        "content": [
            ContentBlock(
                video=VideoContent(
                    format="mp4",
                    source=VideoSource(bytes=video_bytes),
                )
            )
        ],
    }

    session_message = SessionMessage.from_message(message, 0)
    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    original_message = loaded_message.to_message()
    assert original_message["content"][0]["video"]["source"]["bytes"] == video_bytes
    assert original_message["content"][0]["video"]["format"] == "mp4"


def test_session_message_with_reasoning_redacted_content():
    """Test SessionMessage round-trip with ReasoningContentBlock containing redactedContent bytes."""
    redacted_bytes = b"\x01\x02\x03\x04encrypted-reasoning"
    message = {
        "role": "assistant",
        "content": [
            ContentBlock(
                reasoningContent={
                    "redactedContent": redacted_bytes,
                }
            )
        ],
    }

    session_message = SessionMessage.from_message(message, 0)
    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    original_message = loaded_message.to_message()
    assert original_message["content"][0]["reasoningContent"]["redactedContent"] == redacted_bytes


def test_session_message_with_mixed_media_content():
    """Test SessionMessage round-trip with text, image, document, and video in one message."""
    image_bytes = b"\x89PNG-image-data"
    doc_bytes = b"%PDF-document-data"
    video_bytes = b"\x00\x00-video-data"

    message = {
        "role": "user",
        "content": [
            ContentBlock(text="Here are some files"),
            ContentBlock(
                image=ImageContent(
                    format="jpeg",
                    source=ImageSource(bytes=image_bytes),
                )
            ),
            ContentBlock(
                document=DocumentContent(
                    format="txt",
                    name="notes.txt",
                    source=DocumentSource(bytes=doc_bytes),
                )
            ),
            ContentBlock(
                video=VideoContent(
                    format="webm",
                    source=VideoSource(bytes=video_bytes),
                )
            ),
        ],
    }

    session_message = SessionMessage.from_message(message, 0)
    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    original_message = loaded_message.to_message()
    assert original_message["content"][0]["text"] == "Here are some files"
    assert original_message["content"][1]["image"]["source"]["bytes"] == image_bytes
    assert original_message["content"][2]["document"]["source"]["bytes"] == doc_bytes
    assert original_message["content"][3]["video"]["source"]["bytes"] == video_bytes


def test_session_message_with_empty_bytes():
    """Test SessionMessage round-trip with empty bytes."""
    message = {
        "role": "user",
        "content": [
            ContentBlock(
                image=ImageContent(
                    format="png",
                    source=ImageSource(bytes=b""),
                )
            )
        ],
    }

    session_message = SessionMessage.from_message(message, 0)
    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    original_message = loaded_message.to_message()
    assert original_message["content"][0]["image"]["source"]["bytes"] == b""


def test_session_message_with_large_binary_content():
    """Test SessionMessage round-trip with large binary content."""
    large_bytes = bytes(range(256)) * 1000  # 256KB of data
    message = {
        "role": "user",
        "content": [
            ContentBlock(
                image=ImageContent(
                    format="png",
                    source=ImageSource(bytes=large_bytes),
                )
            )
        ],
    }

    session_message = SessionMessage.from_message(message, 0)
    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    original_message = loaded_message.to_message()
    assert original_message["content"][0]["image"]["source"]["bytes"] == large_bytes


def test_encode_bytes_values_preserves_non_bytes():
    """Test that encode_bytes_values does not modify non-bytes values."""
    data = {"text": "hello", "number": 42, "list": [1, 2, 3], "nested": {"key": "value"}}
    encoded = encode_bytes_values(data)
    assert encoded == data


def test_decode_bytes_values_preserves_non_encoded():
    """Test that decode_bytes_values does not modify non-encoded values."""
    data = {"text": "hello", "number": 42, "list": [1, 2, 3], "nested": {"key": "value"}}
    decoded = decode_bytes_values(data)
    assert decoded == data


def test_decode_bytes_values_ignores_partial_marker():
    """Test that decode_bytes_values does not decode dicts that partially match the marker."""
    # Has __bytes_encoded__ but no data key
    data = {"__bytes_encoded__": True, "other": "value"}
    decoded = decode_bytes_values(data)
    assert decoded == {"__bytes_encoded__": True, "other": "value"}

    # Has data key but __bytes_encoded__ is not True
    data = {"__bytes_encoded__": False, "data": "aGVsbG8="}
    decoded = decode_bytes_values(data)
    assert decoded == {"__bytes_encoded__": False, "data": "aGVsbG8="}


def test_session_message_with_redact_message_containing_bytes():
    """Test SessionMessage round-trip when redact_message also contains bytes."""
    image_bytes = b"\x89PNG-original"
    redact_image_bytes = b"\x89PNG-redacted"

    session_message = SessionMessage(
        message={
            "role": "user",
            "content": [
                ContentBlock(
                    image=ImageContent(
                        format="png",
                        source=ImageSource(bytes=image_bytes),
                    )
                )
            ],
        },
        message_id=0,
        redact_message={
            "role": "user",
            "content": [
                ContentBlock(
                    image=ImageContent(
                        format="png",
                        source=ImageSource(bytes=redact_image_bytes),
                    )
                )
            ],
        },
    )

    message_json_string = json.dumps(session_message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    # When redact_message is set, to_message() returns the redact content
    result = loaded_message.to_message()
    assert result["content"][0]["image"]["source"]["bytes"] == redact_image_bytes

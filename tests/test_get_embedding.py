"""Tests for get_embedding text extraction logic (issue #37)."""
import numpy as np
from unittest.mock import patch, MagicMock
import os

# Set a fake API key so module-level OpenAI() doesn't crash
os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-testing"

from hyperdb.hyperdb import get_embedding


class TestGetEmbeddingTextExtraction:
    """Test that get_embedding correctly extracts text from documents."""

    @patch("hyperdb.hyperdb.client")
    def test_nested_key_extraction(self, mock_client):
        """Nested dot-separated key should extract the correct value from all docs."""
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3]),
                  MagicMock(embedding=[0.4, 0.5, 0.6])]
        )
        docs = [
            {"name": "A", "info": {"description": "Hello world"}},
            {"name": "B", "info": {"description": "Goodbye world"}},
        ]
        # Before the fix: the second doc would get doc["description"] instead of
        # doc["info"]["description"] because `for key in key_chain` shadowed the
        # function parameter `key`, corrupting it after the first iteration.
        get_embedding(docs, key="info.description", model_type="openai")

        call_args = mock_client.embeddings.create.call_args
        input_texts = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
        assert input_texts == ["Hello world", "Goodbye world"], (
            f"Expected ['Hello world', 'Goodbye world'], got {input_texts}"
        )

    @patch("hyperdb.hyperdb.client")
    def test_simple_key_extraction(self, mock_client):
        """Simple key without dots should work."""
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3]),
                  MagicMock(embedding=[0.4, 0.5, 0.6])]
        )
        docs = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        get_embedding(docs, key="name", model_type="openai")

        call_args = mock_client.embeddings.create.call_args
        input_texts = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
        assert input_texts == ["Alice", "Bob"], (
            f"Expected ['Alice', 'Bob'], got {input_texts}"
        )

    @patch("hyperdb.hyperdb.client")
    def test_list_key_extraction(self, mock_client):
        """When key is a list, use those specific keys for concatenation."""
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3]),
                  MagicMock(embedding=[0.4, 0.5, 0.6])]
        )
        docs = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]
        get_embedding(docs, key=["name", "city"], model_type="openai")

        call_args = mock_client.embeddings.create.call_args
        input_texts = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
        assert input_texts == ["name: Alice, city: NYC", "name: Bob, city: LA"], (
            f"Expected list key extraction, got {input_texts}"
        )

    @patch("hyperdb.hyperdb.client")
    def test_none_key_extraction(self, mock_client):
        """When key is None, concatenate all key-value pairs."""
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )
        docs = [{"name": "Alice", "age": 30}]
        get_embedding(docs, key=None, model_type="openai")

        call_args = mock_client.embeddings.create.call_args
        input_texts = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
        assert input_texts == ["name: Alice, age: 30"], (
            f"Expected ['name: Alice, age: 30'], got {input_texts}"
        )

    @patch("hyperdb.hyperdb.client")
    def test_newlines_stripped_from_all_paths(self, mock_client):
        """Newlines should be stripped in all key paths (str, list, None)."""
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )
        # Test with key=None (was missing newline stripping before fix)
        docs = [{"text": "line1\nline2"}]
        get_embedding(docs, key=None, model_type="openai")

        call_args = mock_client.embeddings.create.call_args
        input_texts = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
        assert "\n" not in input_texts[0], (
            f"Newlines not stripped in None-key path: {input_texts}"
        )

    @patch("hyperdb.hyperdb.client")
    def test_deeply_nested_key(self, mock_client):
        """Deep nested keys (a.b.c) should traverse correctly."""
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )
        docs = [{"a": {"b": {"c": "deep value"}}}]
        get_embedding(docs, key="a.b.c", model_type="openai")

        call_args = mock_client.embeddings.create.call_args
        input_texts = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
        assert input_texts == ["deep value"], (
            f"Deep nesting failed: {input_texts}"
        )

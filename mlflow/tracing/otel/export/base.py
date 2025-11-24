"""
Base class for OTLP export schema converters.

This module provides a base class for converting MLflow span attributes
to various OTLP semantic conventions when exporting.
"""

import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)


class OtelExportSchemaConverter:
    """
    Base class for OTLP export schema converters.

    Each semantic convention (GenAI, OpenInference, etc.) should extend this class
    and implement the conversion methods.
    """

    def convert_span_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """
        Convert MLflow span attributes to target schema attributes.

        Args:
            attributes: Dictionary of MLflow span attributes

        Returns:
            Dictionary with converted attributes (may include both original and new attributes)
        """
        return attributes

    def convert_metric_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """
        Convert MLflow metric attributes to target schema attributes.

        Args:
            attributes: Dictionary of MLflow metric attributes

        Returns:
            Dictionary with converted attributes
        """
        return attributes

    def get_metric_name(self, mlflow_metric_name: str) -> str:
        """
        Convert MLflow metric name to target schema metric name.

        Args:
            mlflow_metric_name: MLflow metric name

        Returns:
            Converted metric name
        """
        return mlflow_metric_name

    @staticmethod
    def _parse_json_attribute(value: Any) -> Any:
        """
        Parse JSON-encoded attribute value.

        Args:
            value: Attribute value (may be JSON string or already parsed)

        Returns:
            Parsed value
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

from unittest.mock import MagicMock

import pytest

from graphiti_core.tracer import NoOpSpan, OpenTelemetryTracer


def test_opentelemetry_tracer_preserves_application_exception():

    tracer       = MagicMock()
    span_context = MagicMock()
    span         = MagicMock()

    tracer.start_as_current_span.return_value = span_context
    span_context.__enter__.return_value       = span
    span_context.__exit__.return_value        = False

    wrapped_tracer = OpenTelemetryTracer(tracer)

    with pytest.raises(ValueError, match="application failed"):

        with wrapped_tracer.start_span("test"):
            raise ValueError("application failed")


def test_opentelemetry_tracer_falls_back_when_span_start_fails():

    tracer = MagicMock()
    tracer.start_as_current_span.side_effect = RuntimeError("tracing unavailable")

    wrapped_tracer = OpenTelemetryTracer(tracer)

    with wrapped_tracer.start_span("test") as span:
        assert isinstance(span, NoOpSpan)


def test_opentelemetry_tracer_falls_back_when_span_enter_fails():

    tracer       = MagicMock()
    span_context = MagicMock()

    tracer.start_as_current_span.return_value = span_context
    span_context.__enter__.side_effect        = RuntimeError("span enter failed")

    wrapped_tracer = OpenTelemetryTracer(tracer)

    with wrapped_tracer.start_span("test") as span:
        assert isinstance(span, NoOpSpan)
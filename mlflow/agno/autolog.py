from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, Tuple

import mlflow
from mlflow.entities.span import SpanStatus, SpanType, SpanStatusCode
from mlflow.utils.autologging_utils.config import AutoLoggingConfig


# Define constants for this integration
FLAVOR_NAME = "agno"

def _construct_inputs(func: Any, args: Tuple[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    try:
        signature = inspect.signature(func)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for name, val in bound.arguments.items():
            if name in ("self", "cls"):
                continue
            inputs[name] = val
    except Exception:
        inputs["args"] = [a for a in args]
        inputs["kwargs"] = {str(k): v for k, v in kwargs.items()}
    return inputs

def _parse_tools(tools: Iterable[Any]) -> Iterable[str]:
    names = []
    try:
        for tool in tools:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if name:
                names.append(str(name))
            else:
                names.append(str(tool))
    except Exception:
        try:
            names = [str(t) for t in tools]
        except Exception:
            names = []
    return names

def _get_agent_attributes(agent: Any) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    try:
        attributes["agent_class"] = agent.__class__.__name__
        if hasattr(agent, "name"):
            name_value = getattr(agent, "name")
            if name_value:
                attributes["agent_name"] = str(name_value)
        if hasattr(agent, "instructions"):
            attributes["agent_instructions"] = agent.instructions
        if hasattr(agent, "knowledge"):
            attributes["agent_knowledge"] = agent.knowledge
        # Tools
        if hasattr(agent, "tools"):
            tools_obj = getattr(agent, "tools")
            if isinstance(tools_obj, dict):
                tool_names = list(map(str, tools_obj.keys()))
            else:
                tool_names = _parse_tools(tools_obj)
            attributes["tools"] = tool_names
        if hasattr(agent, "session_id"):
            attributes["session_id"] = str(getattr(agent, "session_id"))
        if hasattr(agent, "user_id"):
            attributes["user_id"] = str(getattr(agent, "user_id"))
    except Exception:
        pass
    return attributes

def _extract_token_usage(metrics: Dict[str, Any]) -> Dict[str, Any]:
    token_usage: Dict[str, Any] = {}
    try:
        if not isinstance(metrics, dict):
            return token_usage
        for key in ["input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"]:
            if key in metrics:
                token_usage[key] = metrics.get(key)
        for parent_key in ["session_metrics", "usage"]:
            if parent_key in metrics and isinstance(metrics[parent_key], dict):
                for key in ["input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"]:
                    if key in metrics[parent_key]:
                        token_usage[key] = metrics[parent_key].get(key)
    except Exception:
        pass
    return token_usage

def _set_token_usage_on_span(span: Any, token_usage: Dict[str, Any]) -> None:
    for key, value in token_usage.items():
        try:
            usage_key = {
                "input_tokens": "input",
                "prompt_tokens": "prompt",
                "completion_tokens": "completion",
                "output_tokens": "output",
                "total_tokens": "total",
            }.get(key, key)
            span.set_token_usage(key=usage_key, value=value)
        except Exception:
            continue

def _patched_agent_run(original, agent, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return original(agent, *args, **kwargs)
    
    inputs = _construct_inputs(original, args, kwargs)
    attributes = _get_agent_attributes(agent)
    span_name = attributes.get("agent_name") or attributes.get("agent_class") or "AgnoAgentRun"
    with mlflow.start_span(span_type=SpanType.AGENT, name=span_name) as span:
        span.set_inputs(inputs)
        span.set_attributes(attributes)
        try:
            result = original(agent, *args, **kwargs)
            output = result
            span.set_outputs(output)
            metrics = None
            try:
                if hasattr(result, "metrics"):
                    metrics = getattr(result, "metrics")
                elif hasattr(result, "session_metrics"):
                    metrics = getattr(result, "session_metrics")
            except Exception:
                metrics = None
            if metrics:
                token_usage = _extract_token_usage(metrics)
                _set_token_usage_on_span(span, token_usage)
            span.set_status(SpanStatus(SpanStatusCode.OK))
            return result
        except Exception as e:
            span.set_status(SpanStatus(SpanStatusCode.ERROR))
            raise

async def _patched_agent_arun(original, agent, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(agent, *args, **kwargs)

    inputs = _construct_inputs(original, args, kwargs)
    attributes = _get_agent_attributes(agent)
    span_name = attributes.get("agent_name") or attributes.get("agent_class") or "AgnoAgentARun"
    async with mlflow.start_span(span_type=SpanType.AGENT, name=span_name) as span:
        span.set_inputs(inputs)
        span.set_attributes(attributes)
        try:
            result = await original(agent, *args, **kwargs)
            output = result
            span.set_outputs(output)
            metrics = None
            try:
                if hasattr(result, "metrics"):
                    metrics = getattr(result, "metrics")
                elif hasattr(result, "session_metrics"):
                    metrics = getattr(result, "session_metrics")
            except Exception:
                metrics = None
            if metrics:
                token_usage = _extract_token_usage(metrics)
                _set_token_usage_on_span(span, token_usage)
            span.set_status(SpanStatus(SpanStatusCode.OK))
            return result
        except Exception as e:
            span.set_status(SpanStatus(SpanStatusCode.ERROR))
            raise


def _patched_tool_execute(original, func_call, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return original(func_call, *args, **kwargs)
    tool_name = None
    description = None
    for attr in ["function_name", "name", "tool_name"]:
        val = getattr(func_call, attr, None)
        if val:
            tool_name = str(val)
            break
    if not tool_name and hasattr(func_call, "function"):
        underlying_fn = getattr(func_call, "function")
        for attr in ["name", "__name__", "function_name"]:
            val = getattr(underlying_fn, attr, None)
            if val:
                tool_name = str(val)
                break
        if hasattr(underlying_fn, "description"):
            description = getattr(underlying_fn, "description")
    if not tool_name:
        tool_name = "AgnoToolCall"
    if description is None:
        for attr in ["description", "function_description"]:
            val = getattr(func_call, attr, None)
            if val:
                description = str(val)
                break
    span_name = tool_name
    with mlflow.start_span(span_type=SpanType.TOOL, name=span_name) as span:
        inputs = _construct_inputs(original, args, kwargs)
        if hasattr(func_call, "args"):
            try:
                inputs["tool_args"] = func_call.args
            except Exception:
                pass
        span.set_inputs(inputs)
        attrs = {"tool_name": tool_name}
        if description:
            attrs["tool_description"] = str(description)
        span.set_attributes(attrs)
        try:
            result = original(func_call, *args, **kwargs)
            span.set_outputs(result)
            span.set_status(SpanStatus(SpanStatusCode.OK))
            return result
        except Exception as e:
            span.set_status(SpanStatus(SpanStatusCode.ERROR))
            raise

async def _patched_tool_aexecute(original, func_call, *args, **kwargs):
    cfg = AutoLoggingConfig.get_config(FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(func_call, *args, **kwargs)
    tool_name = None
    description = None
    for attr in ["function_name", "name", "tool_name"]:
        val = getattr(func_call, attr, None)
        if val:
            tool_name = str(val)
            break
    if not tool_name and hasattr(func_call, "function"):
        underlying_fn = getattr(func_call, "function")
        for attr in ["name", "__name__", "function_name"]:
            val = getattr(underlying_fn, attr, None)
            if val:
                tool_name = str(val)
                break
        if hasattr(underlying_fn, "description"):
            description = getattr(underlying_fn, "description")
    if not tool_name:
        tool_name = "AgnoToolCall"
    if description is None:
        for attr in ["description", "function_description"]:
            val = getattr(func_call, attr, None)
            if val:
                description = str(val)
                break
    span_name = tool_name
    async with mlflow.start_span(span_type=SpanType.TOOL, name=span_name) as span:
        inputs = _construct_inputs(original, args, kwargs)
        if hasattr(func_call, "args"):
            try:
                inputs["tool_args"] = func_call.args
            except Exception:
                pass
        span.set_inputs(inputs)
        attrs = {"tool_name": tool_name}
        if description:
            attrs["tool_description"] = str(description)
        span.set_attributes(attrs)
        try:
            result = await original(func_call, *args, **kwargs)
            span.set_outputs(result)
            span.set_status(SpanStatus(SpanStatusCode.OK))
            return result
        except Exception as e:
            span.set_status(SpanStatus(SpanStatusCode.ERROR), message=str(e))
            raise

def _patched_agent_print_response(original, agent, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return original(agent, *args, **kwargs)

    inputs = _construct_inputs(original, args, kwargs)
    attributes = _get_agent_attributes(agent)
    span_name = attributes.get("agent_name") or "print_response"
    with mlflow.start_span(span_type=SpanType.AGENT, name=span_name) as span:
        span.set_inputs(inputs)
        span.set_attributes(attributes)
        try:
            result = original(agent, *args, **kwargs)
            span.set_outputs(result)
            span.set_status(SpanStatus(SpanStatusCode.OK))
            return result
        except Exception as e:
            span.set_status(SpanStatus(SpanStatusCode.ERROR))
            raise

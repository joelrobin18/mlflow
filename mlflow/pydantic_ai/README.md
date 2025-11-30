# Pydantic AI Autolog Implementation Notes

## Why `run_stream_sync` Requires Special Handling

This document explains why `run_stream_sync` requires a wrapper class (`_StreamedRunResultSyncWrapper`) 
and `start_span_no_context`, while other methods (`run`, `run_sync`, `run_stream`) do not.

---

## Evidence from Pydantic AI Source Code

### 1. `run_stream_sync` Returns Immediately (NOT a Context Manager)

From `pydantic_ai/agent/abstract.py` (lines 704-723):

```python
def run_stream_sync(self, user_prompt, ...):
    async def _consume_stream():
        async with self.run_stream(...) as stream_result:
            yield stream_result

    async_result = _utils.get_event_loop().run_until_complete(anext(_consume_stream()))
    return result.StreamedRunResultSync(async_result)  # Returns immediately!
```

**Key insight:** `run_stream_sync` returns a `StreamedRunResultSync` object immediately. 
It does NOT block until streaming completes.

### 2. `StreamedRunResultSync` - Outputs Available Only After Consumption

From `pydantic_ai/result.py` (lines 703-709):

```python
def usage(self) -> RunUsage:
    """Return the usage of the whole run.

    !!! note
        This won't return the full usage until the stream is finished.
    """
    return self._streamed_run_result.usage()
```

From `pydantic_ai/result.py` (lines 727-736):

```python
@property
def is_complete(self) -> bool:
    """Whether the stream has all been received.

    This is set to `True` when one of
    [`stream_output`], [`stream_text`], [`stream_responses`] or
    [`get_output`] completes.
    """
    return self._streamed_run_result.is_complete
```

**Key insight:** The `usage()` and other outputs are NOT available until streaming completes.
`is_complete` becomes `True` only after one of the streaming methods finishes.

### 3. Contrast with `run_stream` (Async Context Manager)

From `pydantic_ai/agent/abstract.py` (lines 405-406):

```python
@asynccontextmanager  # <-- Context manager!
async def run_stream(self, user_prompt, ...):
    ...
```

**Key insight:** `run_stream` is a context manager, so we can capture outputs in the `finally` 
block when the `async with` exits. No wrapper needed.

---

## The Problem Visualized

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  run_stream_sync Timeline                                                    │
│                                                                              │
│  T1: Function called                                                         │
│      └── Span should START here                                              │
│      └── Inputs available ✅                                                 │
│      └── Outputs NOT available ❌ (usage, messages empty)                    │
│                                                                              │
│  T2: Function returns StreamedRunResultSync                                  │
│      └── If using `with start_span():`, span would END here ❌               │
│      └── User hasn't iterated yet!                                           │
│                                                                              │
│  T3: User iterates (stream_text, stream_output, etc.)                        │
│      └── Streaming happens here                                              │
│                                                                              │
│  T4: Iteration completes                                                     │
│      └── Outputs NOW available ✅ (usage, messages populated)                │
│      └── Span should END here ✅                                             │
│      └── is_complete becomes True                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why `start_span_no_context` is Required

### Option 1: `with mlflow.start_span()` - WRONG ❌

```python
def patched_sync_stream_call(original, self, *args, **kwargs):
    with mlflow.start_span(...) as span:  # Span starts at T1
        span.set_inputs(...)
        result = original(...)
        return result
    # Span ends at T2 - BEFORE user iterates! ❌
```

### Option 2: `mlflow.start_span_no_context()` - CORRECT ✅

```python
def patched_sync_stream_call(original, self, *args, **kwargs):
    span = mlflow.start_span_no_context(...)  # Span starts at T1, no auto-end
    span.set_inputs(...)
    result = original(...)
    return _StreamedRunResultSyncWrapper(result, span)  # Wrapper will end span at T4
```

---

## Methods That Need Wrapping in `StreamedRunResultSync`

Based on pydantic_ai source (result.py lines 649-696), these methods consume the stream:

| Method | Lines | Description |
|--------|-------|-------------|
| `stream_text()` | 666-679 | Streams text incrementally |
| `stream_output()` | 649-664 | Streams validated output |
| `stream_responses()` | 681-692 | Streams raw model responses |
| `get_output()` | 694-696 | Blocks until complete, returns output |

Our wrapper intercepts ALL of these to detect completion and end the span.

---

## Comparison with Other Methods

| Method | Type | Blocking | Outputs Ready | Span Management |
|--------|------|----------|---------------|-----------------|
| `run` | async | Yes (awaits) | After return | `with start_span()` ✅ |
| `run_sync` | sync | Yes (blocks) | After return | `with start_span()` ✅ |
| `run_stream` | async context mgr | No | On `__aexit__` | `with start_span()` in finally ✅ |
| `run_stream_sync` | sync (returns obj) | No | After iteration | `start_span_no_context` + wrapper ✅ |

---

## Edge Cases Considered

### What if user never consumes the stream?

The span will remain open indefinitely. This is acceptable behavior - similar to:
- Never exiting a `with` block
- Never closing a file handle

The user is responsible for consuming the stream they requested.

### What if user partially consumes the stream?

If the user stops iterating early (e.g., breaks from the loop), our generator's `finally` 
block still runs due to Python's generator cleanup, ensuring the span ends.

```python
def _wrap_iterator(self, iterator):
    try:
        yield from iterator
    finally:
        self._finalize()  # Always called, even on break/exception
```

---

## Summary

The wrapper class exists because:

1. **pydantic_ai design choice**: `run_stream_sync` is NOT a context manager (for convenience)
2. **Outputs delayed**: `usage()`, `new_messages()` only available after streaming
3. **No completion hook**: `StreamedRunResultSync` has no callback for "streaming done"
4. **Manual span control**: We must use `start_span_no_context` to end span at the right time

This is the minimal code required to properly trace `run_stream_sync` with accurate outputs.


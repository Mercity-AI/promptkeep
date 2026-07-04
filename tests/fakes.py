"""Fake OpenAI-shaped clients for wrapper tests. No network, no openai dep.

Mirrors the SDK surface the wrapper touches: `client.chat.completions.create`
returning either a response object, a stream of chunks, or raising — in both
sync and async flavors.
"""

from types import SimpleNamespace


def make_response(
    content="hello!", model="gpt-test", response_id="resp_1", prompt_tokens=10, completion_tokens=5
):
    """Build a chat-completion response shaped like the real SDK's object."""
    return SimpleNamespace(
        id=response_id,
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def make_chunk(content=None, model="gpt-test", response_id="resp_s", usage=None):
    """Build one streaming chunk (delta content, optionally final usage)."""
    return SimpleNamespace(
        id=response_id,
        model=model,
        usage=usage,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
    )


class FakeStream:
    """Iterable + context-manager stand-in for the SDK's sync Stream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.entered = False
        self.exited = False

    def __iter__(self):
        return iter(self._chunks)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class FakeAsyncStream:
    """Async-iterable stand-in for the SDK's AsyncStream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeCompletions:
    """Records every create() call and replays a canned response/stream/error."""

    def __init__(self, response=None, error=None, stream_chunks=None):
        self.response = response if response is not None else make_response()
        self.error = error
        self.stream_chunks = stream_chunks
        self.calls = []

    def create(self, **kwargs):
        """Mimic chat.completions.create: record kwargs, then respond as configured."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return FakeStream(self.stream_chunks or [])
        return self.response


class FakeAsyncCompletions(FakeCompletions):
    """Async variant — the wrapper must detect the coroutine and await it."""

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return FakeAsyncStream(self.stream_chunks or [])
        return self.response


class FakeClient:
    """Mimics openai.OpenAI: client.chat.completions.create(**kwargs)."""

    completions_cls = FakeCompletions

    def __init__(self, api_key=None, response=None, error=None, stream_chunks=None):
        self.api_key = api_key
        self.chat = SimpleNamespace(
            completions=self.completions_cls(
                response=response, error=error, stream_chunks=stream_chunks
            )
        )

    @property
    def calls(self):
        """Shortcut to the recorded create() calls."""
        return self.chat.completions.calls


class FakeAsyncClient(FakeClient):
    """Mimics openai.AsyncOpenAI by swapping in the async completions."""

    completions_cls = FakeAsyncCompletions

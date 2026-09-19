"""Fake OpenAI-shaped clients for wrapper tests. No network, no openai dep.

Mirrors the SDK surface the wrapper touches: `client.chat.completions.create`
returning either a response object, a stream of chunks, or raising — in both
sync and async flavors.
"""

from types import SimpleNamespace


def make_usage(prompt_tokens=10, completion_tokens=5, cost=None):
    """A usage block. ``cost`` is the field OpenRouter adds to OpenAI's shape;
    left off entirely when None, as it is against OpenAI's own API."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    if cost is not None:
        usage.cost = cost
    return usage


def make_response(
    content="hello!",
    model="gpt-test",
    response_id="resp_1",
    prompt_tokens=10,
    completion_tokens=5,
    cost=None,
):
    """Build a chat-completion response shaped like the real SDK's object."""
    return SimpleNamespace(
        id=response_id,
        model=model,
        usage=make_usage(prompt_tokens, completion_tokens, cost),
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


class FakeDecoratedAsyncCompletions(FakeAsyncCompletions):
    """The real AsyncOpenAI's shape: ``create`` is an ``async def`` behind a
    plain-``def`` decorator (the SDK's ``@required_args``), so it does not
    *look* like a coroutine function — it just returns a coroutine.

    ``leave_trail`` is whether the decorator used functools.wraps: the SDK's
    does (``__wrapped__`` leads back to the async def); a careless one doesn't,
    and then the only evidence is the awaitable that comes back.
    """

    def __init__(self, leave_trail=True, **kwargs):
        super().__init__(**kwargs)
        inner = super().create

        def create(**call_kwargs):
            return inner(**call_kwargs)

        if leave_trail:
            create.__wrapped__ = inner
        self.create = create


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


# --- the Responses API surface ---------------------------------------------------


def make_responses_usage(input_tokens=10, output_tokens=5, cost=None):
    """A Responses usage block (input/output, not prompt/completion tokens)."""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    if cost is not None:
        usage.cost = cost
    return usage


def make_responses_response(
    text="hello!", model="gpt-test", response_id="resp_1", usage=None, previous_response_id=None
):
    """Build a Response shaped like the SDK's: the reply is an ``output_text``
    block inside a ``message`` item, after a reasoning item that carries none."""
    return SimpleNamespace(
        id=response_id,
        model=model,
        previous_response_id=previous_response_id,
        usage=usage if usage is not None else make_responses_usage(),
        output=[
            SimpleNamespace(type="reasoning", summary=[]),
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text=text)],
            ),
        ],
    )


def make_responses_events(parts, model="gpt-test", response_id="resp_s", usage=None):
    """A Responses event stream: created, one text delta per part, completed
    (the only event whose Response carries usage)."""
    started = SimpleNamespace(id=response_id, model=model, usage=None, output=[])
    finished = make_responses_response(
        "".join(parts),
        model=model,
        response_id=response_id,
        usage=usage if usage is not None else make_responses_usage(7, 3),
    )
    return (
        [SimpleNamespace(type="response.created", response=started)]
        + [SimpleNamespace(type="response.output_text.delta", delta=part) for part in parts]
        + [SimpleNamespace(type="response.completed", response=finished)]
    )


class FakeResponses:
    """Records every create() call; replays canned responses (one per call,
    the last one repeating), a stream of events, or an error."""

    def __init__(self, responses=None, error=None, stream_events=None):
        self.responses = list(responses) if responses else [make_responses_response()]
        self.error = error
        self.stream_events = stream_events
        self.calls = []

    def _next(self, kwargs):
        """Record the call and pick what to answer it with."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return None
        return self.responses[min(len(self.calls), len(self.responses)) - 1]

    def create(self, **kwargs):
        """Mimic responses.create."""
        response = self._next(kwargs)
        return FakeStream(self.stream_events or []) if response is None else response


class FakeAsyncResponses(FakeResponses):
    """Async variant of the Responses surface."""

    async def create(self, **kwargs):
        response = self._next(kwargs)
        return FakeAsyncStream(self.stream_events or []) if response is None else response


class FakeResponsesClient:
    """Mimics the part of openai.OpenAI the Responses adapter touches:
    client.responses.create(**kwargs)."""

    responses_cls = FakeResponses

    def __init__(self, api_key=None, responses=None, error=None, stream_events=None):
        self.api_key = api_key
        self.responses = self.responses_cls(
            responses=responses, error=error, stream_events=stream_events
        )

    @property
    def calls(self):
        """Shortcut to the recorded create() calls."""
        return self.responses.calls


class FakeAsyncResponsesClient(FakeResponsesClient):
    """Mimics openai.AsyncOpenAI's Responses surface."""

    responses_cls = FakeAsyncResponses


class FakeFullClient(FakeClient):
    """Both surfaces on one client, like the real SDK's."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.responses = FakeResponses()

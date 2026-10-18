"""llm.chat: retries rate limits with backoff, strips <think> blocks, gives up after the last retry."""
import sys
import types

import pytest

import llm


class RateLimitError(Exception):
    pass


class APIConnectionError(Exception):
    pass


def install_fake_groq(monkeypatch, outcomes):
    """outcomes: list of exceptions to raise or strings to return, one per API call."""
    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            out = outcomes[len(calls) - 1]
            if isinstance(out, Exception):
                raise out
            msg = types.SimpleNamespace(content=out)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class Groq:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=Completions())

    fake = types.SimpleNamespace(Groq=Groq, RateLimitError=RateLimitError, APIConnectionError=APIConnectionError)
    monkeypatch.setitem(sys.modules, "groq", fake)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return calls, sleeps


def test_retries_a_rate_limit_then_succeeds(monkeypatch):
    calls, sleeps = install_fake_groq(monkeypatch, [RateLimitError(), RateLimitError(), "उत्तर [3]"])
    assert llm.chat("sys", "user") == "उत्तर [3]"
    assert len(calls) == 3 and sleeps == [2, 4]                    # exponential backoff
    assert calls[0]["temperature"] == 0 and calls[0]["messages"][0]["content"] == "sys"


def test_gives_up_after_the_last_retry(monkeypatch):
    calls, _ = install_fake_groq(monkeypatch, [APIConnectionError()] * 4)
    with pytest.raises(APIConnectionError):
        llm.chat("s", "u", retries=3)
    assert len(calls) == 4


def test_other_errors_are_not_retried(monkeypatch):
    calls, sleeps = install_fake_groq(monkeypatch, [ValueError("bad request")])
    with pytest.raises(ValueError):
        llm.chat("s", "u")
    assert len(calls) == 1 and sleeps == []


def test_think_blocks_are_removed(monkeypatch):
    install_fake_groq(monkeypatch, ["<think>hidden reasoning</think>अंतिम उत्तर"])
    assert llm.chat("s", "u") == "अंतिम उत्तर"

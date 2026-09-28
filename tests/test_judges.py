import asyncio

import pytest

from interptemp.config import JudgeConfig
from interptemp.judges import JudgeInput
from interptemp.judges.llm import LLMJudge
from interptemp.judges.metrics import agreement_report, cohen_kappa
from interptemp.judges.substring import SubstringJudge


@pytest.fixture
def judge(tmp_path):
    return LLMJudge(
        JudgeConfig(cache_path=str(tmp_path / "c.sqlite")),
        rubric="P: {prompt}\nR: {response}\nW: {word}",
    )


def test_parse(judge):
    assert judge.parse('{"reasoning": "ok", "label": "YES"}') == ("yes", "ok")
    assert judge.parse('```json\n{"label": "no", "reasoning": "r"}\n```') == ("no", "r")
    assert judge.parse('{"label": "maybe"}')[0] is None  # not an allowed label
    assert judge.parse("no json here")[0] is None
    assert judge.parse("{broken")[0] is None


def test_build_messages_with_braces_in_response(judge):
    msgs = judge.build_messages(JudgeInput("p", "code {x: 1}", extra={"word": "cat"}))
    body = msgs[-1]["content"]
    assert "R: code {x: 1}" in body and "W: cat" in body and '"label"' in body


class _FakeCompletions:
    def __init__(self, reply, reasoning_tokens=None):
        self.calls, self.reply, self.reasoning_tokens = 0, reply, reasoning_tokens

    async def create(self, **req):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        msg = type("M", (), {"content": self.reply})
        details = type("D", (), {"reasoning_tokens": self.reasoning_tokens})
        usage = type("U", (), {"completion_tokens_details": details})
        return type("R", (), {"choices": [type("C", (), {"message": msg})], "usage": usage})


def _fake_client(judge, reply, reasoning_tokens=None):
    comp = _FakeCompletions(reply, reasoning_tokens)
    judge._client = type("Cl", (), {"chat": type("Ch", (), {"completions": comp})})
    return comp


def test_cache_prevents_repeat_calls(judge):
    comp = _fake_client(judge, '{"reasoning": "r", "label": "yes"}')
    items = [JudgeInput("p", "r", extra={"word": "w"})]
    first = asyncio.run(judge.ajudge(items))
    second = asyncio.run(judge.ajudge(items))
    assert first[0].label == second[0].label == "yes"
    assert comp.calls == 1


def test_api_error_yields_none_and_is_not_cached(judge):
    comp = _fake_client(judge, RuntimeError("boom"))
    items = [JudgeInput("p", "r", extra={"word": "w"})]
    res = asyncio.run(judge.ajudge(items))
    assert res[0].label is None and "boom" in res[0].meta["error"]
    asyncio.run(judge.ajudge(items))
    assert comp.calls == 2


def test_substring_judge():
    j = SubstringJudge()
    res = j.judge([JudgeInput("", "I'm sorry, I can't help."), JudgeInput("", "Sure, here's how")])
    assert [r.label for r in res] == ["yes", "no"]


def test_kappa_known_values():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # p_o = 0.5, p_e = 0.5 -> 0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    # p_o = 0.8, p_e = (3*2 + 2*3)/25 = 0.48 -> 0.32/0.52
    assert cohen_kappa(["y", "y", "n", "n", "y"], ["y", "n", "n", "n", "y"]) == pytest.approx(
        0.6153846, rel=1e-6
    )


def test_agreement_report_handles_missing():
    rep = agreement_report(["y", None, "n"], ["y", "y", "n"])
    assert rep["n_scored"] == 2 and rep["n_judge_missing"] == 1 and rep["accuracy"] == 1.0


@pytest.mark.parametrize(("tokens", "warns"), [(0, True), (57, False), (None, False)])
def test_warns_when_reasoning_silently_dropped(judge, caplog, tokens, warns):
    _fake_client(judge, '{"reasoning": "r", "label": "yes"}', reasoning_tokens=tokens)
    res = asyncio.run(judge.ajudge([JudgeInput("p", "r", extra={"word": "w"})]))
    assert res[0].meta["reasoning_tokens"] == tokens
    assert ("0 reasoning tokens" in caplog.text) is warns


def test_no_reasoning_warning_when_reasoning_off(tmp_path, caplog):
    cfg = JudgeConfig(cache_path=str(tmp_path / "c.sqlite"), reasoning_effort=None)
    j = LLMJudge(cfg, rubric="{prompt} {response}")
    _fake_client(j, '{"label": "no"}', reasoning_tokens=0)
    asyncio.run(j.ajudge([JudgeInput("p", "r")]))
    assert "reasoning tokens" not in caplog.text

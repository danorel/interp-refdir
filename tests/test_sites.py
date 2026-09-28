import pytest

from interptemp.sites import Site, sort_sites


def test_parse_roundtrip():
    for s in ["embed", "resid_pre.0", "attn_out.3", "mlp_out.3", "resid_post.27"]:
        assert str(Site.parse(s)) == s


@pytest.mark.parametrize("bad", ["nope.1", "resid_post", "embed.1", "resid_post.-1"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        Site.parse(bad)


def test_execution_order():
    got = sort_sites(
        [
            Site.parse(s)
            for s in [
                "resid_post.1",
                "mlp_out.1",
                "resid_pre.2",
                "embed",
                "attn_out.1",
                "resid_pre.1",
            ]
        ]
    )
    assert [str(s) for s in got] == [
        "embed",
        "resid_pre.1",
        "attn_out.1",
        "mlp_out.1",
        "resid_post.1",
        "resid_pre.2",
    ]


def test_sort_dedups():
    assert sort_sites([Site("resid_post", 1), Site("resid_post", 1)]) == [Site("resid_post", 1)]

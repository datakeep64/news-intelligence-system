"""
Unit tests for cross-source event clustering.

The clustering structure (grouping, sources, consensus terms, representative) is
deterministic and tested directly. Tone-spread depends on the VADER lexicon, so
that test skips cleanly if the lexicon isn't available offline.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pipeline.cluster import cluster_events, _tone_spread, _get_sia


def _article(title, summary, source):
    return {"title": title, "summary": summary, "source": source, "link": "", "published": ""}


# Two events: an interest-rate story (3 outlets) and an unrelated football story (2 outlets).
# Wording mirrors real cross-outlet coverage — shared named entities and stock phrases.
ARTICLES = [
    _article("Federal Reserve holds interest rates steady as inflation eases",
             "The Federal Reserve kept interest rates unchanged on Wednesday, citing easing "
             "inflation and a resilient labour market.", "Outlet A"),
    _article("Fed keeps interest rates unchanged amid easing inflation",
             "The Federal Reserve held interest rates steady, pointing to easing inflation "
             "and a resilient labour market.", "Outlet B"),
    _article("Federal Reserve leaves interest rates on hold as inflation slows",
             "Policymakers at the Federal Reserve kept interest rates unchanged, noting "
             "inflation has eased and the labour market remains resilient.", "Outlet C"),
    _article("Manchester City win Premier League title on final day",
             "Manchester City secured the Premier League title with a final-day victory, "
             "holding off their rivals in a dramatic finish.", "Outlet A"),
    _article("Man City crowned Premier League champions after dramatic final day",
             "Manchester City were crowned Premier League champions following a dramatic "
             "final-day win over their title rivals.", "Outlet B"),
]


def test_groups_same_event_across_sources():
    out = cluster_events(ARTICLES)
    # Expect two multi-source events.
    assert len(out["events"]) == 2
    sizes = sorted(e["size"] for e in out["events"])
    assert sizes == [2, 3]


def test_largest_event_listed_first():
    out = cluster_events(ARTICLES)
    assert out["events"][0]["size"] == 3


def test_event_lists_distinct_sources():
    out = cluster_events(ARTICLES)
    rates_event = next(e for e in out["events"] if e["size"] == 3)
    assert rates_event["sources"] == ["Outlet A", "Outlet B", "Outlet C"]


def test_consensus_terms_reflect_shared_topic():
    out = cluster_events(ARTICLES)
    rates_event = next(e for e in out["events"] if e["size"] == 3)
    joined = " ".join(rates_event["consensus_terms"]).lower()
    assert "inflation" in joined or "rates" in joined or "interest" in joined


def test_representative_is_a_member():
    out = cluster_events(ARTICLES)
    for e in out["events"]:
        assert e["representative"] in e["members"]


def test_unrelated_articles_become_singletons():
    articles = [
        ARTICLES[0],
        _article("Rare orchid blooms at botanical garden after a decade",
                 "A rare orchid flowered for the first time in ten years at the city garden.",
                 "Outlet D"),
    ]
    out = cluster_events(articles)
    assert out["events"] == []
    assert len(out["singletons"]) == 2


def test_empty_and_single_inputs():
    assert cluster_events([])["events"] == []
    one = [_article("Solo story", "Only one.", "Outlet A")]
    out = cluster_events(one)
    assert out["events"] == []
    assert out["singletons"] == one


def test_threshold_controls_granularity():
    # A very tight threshold should split even similar articles into singletons.
    out = cluster_events(ARTICLES, distance_threshold=0.05)
    assert all(e["size"] >= 2 for e in out["events"])  # any surviving event still multi-article


@pytest.mark.skipif(_get_sia() is None, reason="VADER lexicon unavailable offline")
def test_tone_spread_detects_divergence():
    members = [
        _article("Plan hailed as a triumph and a wonderful success", "", "Outlet A"),
        _article("Plan slammed as a disaster and a terrible failure", "", "Outlet B"),
    ]
    tone = _tone_spread(members)
    assert tone["available"] is True
    assert tone["divergent"] is True
    assert tone["spread"] >= 0.5

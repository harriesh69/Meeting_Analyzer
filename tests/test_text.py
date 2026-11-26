from analyzer import text
from analyzer.text import UNKNOWN, Sentence, Word


def test_assign_speakers_picks_largest_overlap():
    words = [(0.0, 1.0, "hello"), (1.0, 2.0, "there"), (9.0, 9.5, "gap")]
    turns = [(0.0, 1.2, "A"), (1.2, 3.0, "B")]
    got = text.assign_speakers(words, turns)
    assert [w.speaker for w in got] == ["A", "B", UNKNOWN]


def test_assign_speakers_finds_long_turn_that_started_earlier():
    # The binary search must still see a long turn starting well before the word.
    turns = [(0.0, 100.0, "A"), (50.0, 50.5, "B")]
    got = text.assign_speakers([(60.0, 61.0, "w")], turns)
    assert got[0].speaker == "A"


def test_assign_speakers_matches_brute_force():
    import random
    rng = random.Random(1)
    turns = []
    t = 0.0
    for _ in range(200):
        d = rng.uniform(0.2, 8)
        turns.append((t, t + d, rng.choice("ABC")))
        t += d - rng.uniform(0, 0.5)
    words = [(x, x + rng.uniform(0.1, 0.6), "w") for x in (rng.uniform(0, t) for _ in range(500))]
    got = text.assign_speakers(words, turns)
    for (ws, we, _), w in zip(words, got, strict=True):
        best, spk = 0.0, UNKNOWN
        for s, e, lab in sorted(turns):
            o = min(we, e) - max(ws, s)
            if o > best:
                best, spk = o, lab
        assert w.speaker == spk


def test_fill_unknown_uses_agreeing_neighbours_and_handles_edges():
    words = [Word("A", 0, 1, "a"), Word(UNKNOWN, 1, 2, "b"), Word("A", 2, 3, "c"),
             Word(UNKNOWN, 3, 4, "d")]  # last word unknown used to raise IndexError
    got = text.fill_unknown(words)
    assert [w.speaker for w in got] == ["A", "A", "A", UNKNOWN]


def test_fill_unknown_leaves_disagreement_alone():
    words = [Word("A", 0, 1, "a"), Word(UNKNOWN, 1, 2, "b"), Word("B", 2, 3, "c")]
    assert text.fill_unknown(words)[1].speaker == UNKNOWN


def test_merge_and_split():
    words = [Word("A", 0, 1, "Hello."), Word("A", 1, 2, "How"), Word("A", 2, 4, "are you?"),
             Word("B", 4, 5, "Fine.")]
    blocks = text.merge_turns(words)
    assert [b.speaker for b in blocks] == ["A", "B"]
    sents = text.split_sentences(blocks)
    assert [s.text for s in sents] == ["Hello.", "How are you?", "Fine."]
    assert sents[0].start == 0 and sents[1].end == 4
    assert sents[0].end == sents[1].start  # time split by character share


def test_action_items_whole_word_hedges():
    sentences = [
        "Priya will send the report by Friday.",
        "Maybe we should review it later.",            # hedged proposal
        "Please review the rethinking doc today.",     # "think" inside a word is fine
        "The weather is nice.",
    ]
    assert text.action_items(sentences) == [sentences[0], sentences[2]]


def test_dominance():
    sents = [Sentence("A", 0, 6, "x"), Sentence("B", 6, 9, "y"), Sentence("A", 9, 10, "z")]
    d = text.dominance(sents)
    assert d["A"]["ratio"] == 70.0 and d["A"]["label"] == "Dominant" and d["A"]["turns"] == 2
    assert d["B"]["label"] == "High"


def test_summary_candidate():
    assert text.is_summary_candidate("We agreed to ship the release on Monday.")
    assert not text.is_summary_candidate("Too short.")
    assert not text.is_summary_candidate("This sentence has no ending punctuation at all")
    assert not text.is_summary_candidate("It was sort of a long and winding meeting today.")


def test_format_time():
    assert text.format_time(75.5) == "1:15.50"
    assert text.format_time(3725) == "1:02:05.00"

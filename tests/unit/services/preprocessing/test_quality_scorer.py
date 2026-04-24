from src.services.preprocessing.quality_scorer import noise_score, assess_quality


# ---------------------------------------------------------------------------
# noise_score
# ---------------------------------------------------------------------------


class TestNoiseScore:
    def test_empty_text_returns_all_zeros(self):
        result = noise_score("")
        assert result["score"] == 0.0
        assert result["non_letter_ratio"] == 0.0
        assert result["punct_ratio"] == 0.0
        assert result["upper_ratio"] == 0.0
        assert result["avg_sent_len"] == 0.0
        assert result["url_ratio"] == 0.0

    def test_score_always_between_0_and_1(self):
        texts = [
            "",
            "normal prose text here.",
            "!!!???!!!???!!!",
            "EVERYTHING IS CAPS AND LOUD",
            " ".join(f"https://site{i}.com" for i in range(10)),
        ]
        for t in texts:
            result = noise_score(t)
            assert 0.0 <= result["score"] <= 1.0, f"score out of range for: {t!r}"

    def test_clean_prose_has_low_score(self):
        text = (
            "Alice walked slowly through the ancient forest, admiring the tall trees. "
            "The morning sun filtered through the canopy and cast dappled light on the mossy ground. "
            "She had always loved this place and felt a deep sense of peace whenever she visited."
        )
        result = noise_score(text)
        assert result["score"] < 0.4

    def test_all_caps_raises_upper_ratio(self):
        lower = noise_score("this is normal text with lowercase letters only.")
        upper = noise_score("THIS IS ALL CAPS TEXT WITH UPPERCASE LETTERS ONLY.")
        assert upper["upper_ratio"] > lower["upper_ratio"]

    def test_url_heavy_text_raises_url_ratio(self):
        text_with_urls = " ".join(f"https://example{i}.com" for i in range(5))
        text_clean = "Some regular text without any URLs or links whatsoever."
        assert (
            noise_score(text_with_urls)["url_ratio"]
            > noise_score(text_clean)["url_ratio"]
        )

    def test_punct_heavy_text_raises_punct_ratio(self):
        heavy = "!!!...;;;???!!!...;;;???!!!"
        clean = "This is a normal sentence with minimal punctuation."
        assert noise_score(heavy)["punct_ratio"] > noise_score(clean)["punct_ratio"]

    def test_returns_all_expected_keys(self):
        result = noise_score("some text")
        expected_keys = {
            "score",
            "non_letter_ratio",
            "punct_ratio",
            "upper_ratio",
            "avg_sent_len",
            "url_ratio",
        }
        assert set(result.keys()) == expected_keys

    def test_score_values_are_rounded(self):
        result = noise_score("hello world!")
        for key, val in result.items():
            assert val == round(val, 3), f"key '{key}' not rounded: {val}"

    def test_very_long_sentence_triggers_elif_branch(self):
        # avg_sent_len > 250 déclenche le elif (ligne 45 quality_scorer.py)
        long_text = "word " * 60 + "."  # une seule phrase de ~300 chars
        result = noise_score(long_text)
        assert result["avg_sent_len"] > 250
        assert result["score"] > 0


# ---------------------------------------------------------------------------
# assess_quality
# ---------------------------------------------------------------------------


class TestAssessQuality:
    def test_returns_valid_label(self):
        valid = {"excellent", "good", "fair", "poor"}
        for text in ["", "normal text.", "!!!", "CAPS EVERYWHERE !!!"]:
            assert assess_quality(text) in valid

    def test_clean_prose_is_excellent_or_good(self):
        text = (
            "The protagonist walked through the ancient forest, feeling the weight of her quest. "
            "Each step brought her closer to the truth she had sought for many years. "
            "The trees whispered secrets only she could understand."
        )
        assert assess_quality(text) in ("excellent", "good")

    def test_heavy_noise_is_fair_or_poor(self):
        text = "!!!! #### @@@@ %%%% $$$$ ^^^^ &&&&" * 5
        assert assess_quality(text) in ("fair", "poor")

    def test_thresholds_match_noise_score(self):
        """assess_quality doit être cohérent avec les seuils de noise_score."""
        texts = [
            "Normal well-written prose with multiple sentences. Each one is clear and informative.",
        ]
        for text in texts:
            score = noise_score(text)["score"]
            label = assess_quality(text)
            if score < 0.2:
                assert label == "excellent"
            elif score < 0.4:
                assert label == "good"
            elif score < 0.6:
                assert label == "fair"
            else:
                assert label == "poor"

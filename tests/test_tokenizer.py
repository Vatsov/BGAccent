from bgaccent.tokenizer import Token, detokenize, tokenize

BG_OPEN_QUOTE = "„"
BG_CLOSE_QUOTE = "“"


class TestBasicTokenization:
    def test_two_words(self) -> None:
        tokens = tokenize("здравей свят")
        assert len(tokens) == 3
        assert tokens[0] == Token(kind="word", text="здравей", original="здравей", line=1, col=1)
        assert tokens[1] == Token(kind="space", text=" ", original=" ", line=1, col=8)
        assert tokens[2] == Token(kind="word", text="свят", original="свят", line=1, col=9)

    def test_multiline_line_col_tracking(self) -> None:
        tokens = tokenize("ред 1\nред 2")
        words = [t for t in tokens if t.kind == "word"]
        assert words[0].line == 1
        assert words[0].col == 1
        assert words[1].line == 2
        assert words[1].col == 1

    def test_punctuation_classified(self) -> None:
        tokens = tokenize("Здравей, свят!")
        kinds = [t.kind for t in tokens]
        assert kinds == ["word", "punct", "space", "word", "punct"]

    def test_bulgarian_quotes(self) -> None:
        text = BG_OPEN_QUOTE + "текст" + BG_CLOSE_QUOTE
        tokens = tokenize(text)
        assert tokens[0] == Token(
            kind="punct", text=BG_OPEN_QUOTE, original=BG_OPEN_QUOTE, line=1, col=1
        )
        assert tokens[1] == Token(kind="word", text="текст", original="текст", line=1, col=2)
        assert tokens[2] == Token(
            kind="punct", text=BG_CLOSE_QUOTE, original=BG_CLOSE_QUOTE, line=1, col=7
        )

    def test_whitespace_runs_single_token(self) -> None:
        tokens = tokenize("a  b")
        spaces = [t for t in tokens if t.kind == "space"]
        assert len(spaces) == 1
        assert spaces[0].text == "  "


class TestSpecialTokens:
    def test_abbreviation_dr(self) -> None:
        tokens = tokenize("д-р Иванов")
        assert tokens[0].kind == "abbrev"
        assert tokens[0].text == "д-р"

    def test_abbreviation_prof(self) -> None:
        tokens = tokenize("проф. Петров")
        assert tokens[0].kind == "abbrev"
        assert tokens[0].text == "проф."

    def test_abbreviation_te_tn(self) -> None:
        tokens = tokenize("т.е. или т.н.")
        abbrevs = [t for t in tokens if t.kind == "abbrev"]
        assert len(abbrevs) == 2

    def test_number_classified(self) -> None:
        tokens = tokenize("2024 година")
        assert tokens[0].kind == "number"
        assert tokens[0].text == "2024"

    def test_hyphenated_word(self) -> None:
        tokens = tokenize("по-голям")
        assert len(tokens) == 1
        assert tokens[0].kind == "word"
        assert tokens[0].text == "по-голям"

    def test_number_word_compound(self) -> None:
        tokens = tokenize("100-годишнина")
        assert len(tokens) == 1
        assert tokens[0].kind == "word"
        assert tokens[0].text == "100-годишнина"

    def test_latin_word(self) -> None:
        tokens = tokenize("Hello world")
        assert tokens[0].kind == "word"
        assert tokens[0].text == "Hello"

    def test_em_dash(self) -> None:
        tokens = tokenize("текст — друг")
        dashes = [t for t in tokens if t.kind == "punct"]
        assert any("—" in t.text for t in dashes)


class TestDetokenize:
    def test_simple_roundtrip(self) -> None:
        assert detokenize(tokenize("здравей свят")) == "здравей свят"

    def test_multiline_roundtrip(self) -> None:
        assert detokenize(tokenize("ред 1\nред 2")) == "ред 1\nред 2"

    def test_bulgarian_quotes_roundtrip(self) -> None:
        text = BG_OPEN_QUOTE + "текст" + BG_CLOSE_QUOTE
        assert detokenize(tokenize(text)) == text

    def test_abbreviations_roundtrip(self) -> None:
        text = "д-р Иванов, т.е. проф."
        assert detokenize(tokenize(text)) == text

    def test_mixed_roundtrip(self) -> None:
        text = "2024 по-голям 100-годишнина"
        assert detokenize(tokenize(text)) == text

    def test_latin_roundtrip(self) -> None:
        text = "Hello world! Здравей."
        assert detokenize(tokenize(text)) == text

    def test_em_dash_ellipsis_double_space(self) -> None:
        text = "текст — друг...  интервал"
        assert detokenize(tokenize(text)) == text

"""On-screen text, rewritten as words a text-to-speech voice says correctly.

The captions are written to be read: θ₀, $83k, α = 1.0e-5, F_m−1, RMSE. A voice given those reads symbols
out one by one, or skips them. ``spoken`` rewrites them as words (theta zero, 83 thousand dollars, alpha
equals 10 to the minus 5, F m minus 1, root mean squared error) and leaves plain sentences exactly as they
are.

Jargon is always said by its full name (MLP as "multi-layer perceptron", never "M-L-P"), so a viewer who
does not know an abbreviation still follows. An abbreviation with no full name in ``JARGON`` is spelled out
letter by letter, and ``unexpanded_acronyms`` finds it, so the voice-over can refuse to say it.

The rules run in order: whole phrases first (where a symbol's meaning depends on its sentence; a project
can add its own, such as how its probability notation is said), then jargon, then numbers, then notation,
then single symbols, then spacing. Running ``spoken`` twice changes nothing.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from functools import reduce

# Phrases whose symbols only make sense together, in the order they are tried. A project adds its own
# (its scene's ``spoken_phrases``), which are tried before these.
PHRASES = (
    ("|score| < 1", "a score between minus 1 and plus 1"),
    ("½|weights|²", "half the squared weights"),
    ("σ(F)", "sigma of F"),
    ("0% / 100%", "0% or 100%"),
    ("0/1", "0-or-1"),
    ("blue = ", "blue means "),
    (" (partial_fit)", ""),
    (" (y, log scale)", " on a log scale"),
    (" (x)", ""),
    (" (y)", ""),
    ("ReLU units", "ReLUs"),  # said "rectified linear units", not "rectified linear unit units"
    ("ReLU neurons", "ReLUs"),
)
# Abbreviations, said by their full names. Case matters (kNN, ReLU); a plural "s" stays, and an "a" or "an"
# before one is changed to suit its full name ("an MLP" is "a multi-layer perceptron").
JARGON = {
    "RMSE": "root mean squared error",
    "MSE": "mean squared error",
    "MAE": "mean absolute error",
    "MLP": "multi-layer perceptron",
    "SVM": "support vector machine",
    "SGD": "stochastic gradient descent",
    "kNN": "k-nearest neighbours",
    "KNN": "k-nearest neighbours",
    "ReLU": "rectified linear unit",
    "AUC": "area under the curve",
    "ROC": "receiver operating characteristic",
    "RBF": "radial basis function",
    "PCA": "principal component analysis",
    "OLS": "ordinary least squares",
    "CNN": "convolutional neural network",
    "RNN": "recurrent neural network",
    "LSTM": "long short-term memory",
}
GREEK = {"θ": "theta", "α": "alpha", "σ": "sigma", "β": "beta", "λ": "lambda", "μ": "mu"}
SYMBOLS = {"×": " times ", "−": " minus ", "±": " plus or minus ", "+": " plus ", "≤": " is at most ",
           "≥": " is at least ", "≈": " about ", "=": " equals ", "<": " is less than ", ">": " is more than ",
           "→": " to ", "…": " up to ", "·": ", ", "°": " degrees", "½": "half", "○": "open circles"}
SUBSCRIPT_WORDS = dict(zip("₀₁₂₃₄₅₆₇₈₉", ("zero", "one", "two", "three", "four", "five", "six", "seven",
                                         "eight", "nine")))
SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
SMALLEST_PLAIN_DECIMAL = 1e-3  # below this a number is said as a power of ten
LARGEST_PLAIN_DECIMAL = 1e6  # from this up a number is said whole, with thousands separators
DECIMAL_PLACES = 6
MONEY_SCALE = {"k": ("thousand", 1_000), "M": ("million", 1_000_000)}
SENTENCE_ENDS = ".!?"

SCIENTIFIC = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)e([+-]?\d+)\b")
MONEY = re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)([kM]?)(?!\w|\.\d)")
POWER = re.compile(r"([A-Za-z0-9|)])(⁻?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
AND_A_HALF = re.compile(r"(\d)½")
SUBSCRIPT = re.compile(r"([A-Za-z])([₀₁₂₃₄₅₆₇₈₉]+)")
UNDERSCORE = re.compile(r"\b([A-Za-z])_([A-Za-z0-9]+)")
ABSOLUTE = re.compile(r"\|(\w+)\|")
EITHER = re.compile(r"\b([A-Za-z]+)/([A-Za-z]+)\b")  # yes/no; numbers like 80/20 are left as they are
RANGE_DASH = re.compile(r"(\d)\s*–\s*(\d)")
JARGON_WORD = re.compile(r"\b(?:(an?|An?) )?(" + "|".join(sorted(map(re.escape, JARGON), key=len, reverse=True))
                         + r")(s?)\b")
ACRONYM = re.compile(r"\b([A-Z]{2,5})\b")  # any abbreviation JARGON does not know: spelled out, then refused
SPELLED_OUT = re.compile(r"\b[A-Z](?:-[A-Z])+\b")


def ordinal(n: int) -> str:
    """``n`` with its English ordinal suffix.

    Args:
        n: A whole number, 0 or more.

    Returns:
        For example "1st", "2nd", "3rd", "4th", "11th", "12th", "21st".
    """
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _say_scientific(match: re.Match) -> str:
    """1.6e-2 as 0.016; 1.0e-5 as "10 to the minus 5"; 2.5e-4 as "2.5 times 10 to the minus 4"."""
    mantissa, exponent = float(match.group(1)), int(match.group(2))
    value = mantissa * 10.0 ** exponent
    if value >= SMALLEST_PLAIN_DECIMAL:
        if value >= LARGEST_PLAIN_DECIMAL:
            return f"{value:,.0f}"
        return f"{value:.{DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    power = f"10 to the {'minus ' if exponent < 0 else ''}{abs(exponent)}"
    return power if mantissa == 1 else f"{mantissa:g} times {power}"


def _starts_sentence(text: str, position: int) -> bool:
    """Whether ``position`` is the start of ``text`` or of a sentence in it."""
    before = text[:position].rstrip()
    return not before or before[-1] in SENTENCE_ENDS


def _say_jargon(match: re.Match) -> str:
    """MLP as "multi-layer perceptron", SVMs as "support vector machines", "an MLP" as "a multi-layer perceptron".

    A full name that starts a sentence starts with a capital, unless its abbreviation does not (kNN).
    """
    article, term, plural = match.groups()
    full = JARGON[term] + plural
    if article:
        fitting = "an" if full[0] in "aeiou" else "a"
        return f"{fitting.capitalize() if article[0].isupper() else fitting} {full}"
    if term[0].isupper() and _starts_sentence(match.string, match.start()):
        return full[0].upper() + full[1:]
    return full


def _dollars(amount: str) -> str:
    """An amount with "dollar" or "dollars"."""
    return f"{amount} dollar" if amount == "1" else f"{amount} dollars"


def _say_money(match: re.Match) -> str:
    """$83k as "83 thousand dollars", $1.5M as "1.5 million dollars", $0.08k as "80 dollars", $1 as "1 dollar"."""
    amount, suffix = match.group(1), match.group(2)
    if not suffix:
        return _dollars(amount)
    word, scale = MONEY_SCALE[suffix]
    value = float(amount.replace(",", ""))
    return _dollars(str(round(value * scale))) if value < 1 else f"{amount} {word} dollars"


def _say_power(match: re.Match) -> str:
    """x² as "x squared", x³ as "x cubed", x¹² as "x to the 12th", 10⁻⁵ as "10 to the minus 5"."""
    base, negative, digits = match.group(1), match.group(2), match.group(3)
    exponent = int(digits.translate(SUPERSCRIPT_DIGITS))
    if negative:
        return f"{base} to the minus {exponent}"
    words = {2: "squared", 3: "cubed"}.get(exponent, f"to the {ordinal(exponent)}")
    return f"{base} {words}"


def _say_subscript(match: re.Match) -> str:
    """θ₀ (already "theta₀") as "theta zero"; F₁₂ as "F one two"."""
    return " ".join((match.group(1), *(SUBSCRIPT_WORDS[digit] for digit in match.group(2))))


def _replace_all(text: str, table: Mapping[str, str] | Sequence[tuple[str, str]]) -> str:
    """Each key of ``table`` replaced by its value, in the table's order."""
    pairs = table.items() if isinstance(table, Mapping) else table
    return reduce(lambda result, pair: result.replace(*pair), pairs, text)


def _tidy(text: str) -> str:
    """One space between words, none inside brackets or before punctuation."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    return text.strip()


REWRITES: tuple[Callable[[str], str], ...] = (
    lambda text: JARGON_WORD.sub(_say_jargon, text),
    lambda text: SCIENTIFIC.sub(_say_scientific, text),
    lambda text: MONEY.sub(_say_money, text),
    lambda text: _replace_all(text, GREEK),
    lambda text: POWER.sub(_say_power, text),
    lambda text: AND_A_HALF.sub(r"\1 and a half", text),
    lambda text: SUBSCRIPT.sub(_say_subscript, text),
    lambda text: UNDERSCORE.sub(r"\1 \2", text),
    lambda text: ABSOLUTE.sub(r"absolute \1", text),
    lambda text: RANGE_DASH.sub(r"\1 to \2", text),
    lambda text: EITHER.sub(r"\1-or-\2", text),
    lambda text: _replace_all(text, SYMBOLS),
    lambda text: ACRONYM.sub(lambda match: "-".join(match.group(1)), text),
    _tidy,
)


def spoken(text: str, phrases: Sequence[tuple[str, str]] = ()) -> str:
    """``text`` as a voice should say it: symbols, money and powers as words, jargon by its full name.

    Args:
        text: On-screen text, for example a caption.
        phrases: The project's own (written, spoken) phrases, tried first and in order.

    Returns:
        The same sentence in plain words; text with nothing to rewrite comes back unchanged.

    Example:
        >>> spoken("First guess: θ₀ = θ₁ = 0, train RMSE $83k.")
        'First guess: theta zero equals theta one equals 0, train root mean squared error 83 thousand dollars.'
    """
    return reduce(lambda result, rewrite: rewrite(result), REWRITES, _replace_all(text, (*phrases, *PHRASES)))


def unexpanded_acronyms(said: str) -> tuple[str, ...]:
    """The abbreviations ``spoken`` had no full name for, and so spelled out letter by letter.

    Args:
        said: Text from ``spoken``.

    Returns:
        Each one once, in alphabetical order; empty when every abbreviation was said by its full name.

    Example:
        >>> unexpanded_acronyms(spoken("The XYZ score of an MLP."))
        ('XYZ',)
    """
    return tuple(sorted({letters.replace("-", "") for letters in SPELLED_OUT.findall(said)}))

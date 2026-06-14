
import re
import unicodedata

TEENCODE_NORMALIZE = {
    'ko': 'không', 'k': 'không', 'khong': 'không',
    'dc': 'được', 'đc': 'được',
    'j': 'gì',
    'z': 'vậy',
    'r': 'rồi',
}

MASK_PATTERNS = [
    (r'\bn[\*\._-]*g[\*\._-]*u\b', 'ngu'),
    (r'\bd[\*\._-]*m\b', 'dm'),
    (r'\bđ[\*\._-]*m\b', 'đm'),
    (r'\bv[\*\._-]*c[\*\._-]*l\b', 'vcl'),
    (r'\bc[\*\._-]*c\b', 'cc'),
]

def reduce_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1', str(text))

def strip_special_inside_words(text):
    text = str(text).lower()
    return re.sub(r'(?<=\w)[\.\*_\-]+(?=\w)', '', text)

def normalize_teencode(text):
    words = str(text).split()
    out = []
    for w in words:
        prefix = re.match(r'^\W*', w).group(0)
        suffix = re.search(r'\W*$', w).group(0)
        core = w[len(prefix): len(w) - len(suffix) if suffix else len(w)]
        out.append(prefix + TEENCODE_NORMALIZE.get(core.lower(), core) + suffix)
    return ' '.join(out)

def normalize_masked_common(text):
    s = str(text).lower()
    for pat, repl in MASK_PATTERNS:
        s = re.sub(pat, repl, s)
    return s

def normalize_comment(text):
    s = unicodedata.normalize('NFC', str(text)).lower()
    s = normalize_masked_common(s)
    s = strip_special_inside_words(s)
    s = reduce_repeated_chars(s)
    s = normalize_teencode(s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

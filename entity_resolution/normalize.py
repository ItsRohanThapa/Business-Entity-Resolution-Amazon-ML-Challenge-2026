"""Normalisation of business names and addresses.

Design rules
------------
* Country-agnostic: the same functions run for every record, whatever its
  country label (the test set contains a country that training does not).
* The maps below are small, hand-written canonicalisation rules (variant ->
  canonical SHORT form). They contain no external data. Both sides of every pair
  go through the same function, so a mapping only has to be consistent, not
  linguistically perfect (e.g. "street" and French "saint" both become "st").
* Nothing is thrown away blindly: legal forms, numbers, postal codes and
  landmark markers are extracted into their own fields and become features.
"""
import multiprocessing as mp
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from .progress import bar

# ----------------------------------------------------------------------------- maps
NAME_CANON = {
    # legal forms
    "incorporated": "inc", "incorporation": "inc", "incorp": "inc", "inc": "inc",
    "corporation": "corp", "corporations": "corp", "corpn": "corp", "corp": "corp",
    "company": "co", "companies": "co", "compagnie": "co", "cie": "co", "coy": "co", "cos": "co",
    "limited": "ltd", "ltda": "ltd", "ltee": "ltd", "lim": "ltd", "ltd": "ltd",
    "private": "pvt", "pte": "pvt", "prvt": "pvt", "pvte": "pvt", "priv": "pvt", "pvt": "pvt",
    "societe": "soc", "ste": "soc", "society": "soc", "soc": "soc",
    # frequent descriptors
    "international": "intl", "internationale": "intl", "intl": "intl",
    "manufacturing": "mfg", "manufacturer": "mfr", "manufacturers": "mfr", "mfrs": "mfr",
    "services": "svc", "service": "svc", "svcs": "svc", "srvc": "svc", "serv": "svc",
    "servs": "svc", "svc": "svc",
    "solutions": "soln", "solution": "soln", "sol": "soln", "solns": "soln",
    "technologies": "tech", "technology": "tech", "technol": "tech", "techno": "tech",
    "technologie": "tech", "techs": "tech", "tech": "tech",
    "systems": "sys", "system": "sys", "syst": "sys", "systemes": "sys",
    "enterprises": "ent", "enterprise": "ent", "entp": "ent", "entps": "ent",
    "entreprise": "ent", "ent": "ent",
    "industries": "ind", "industry": "ind", "inds": "ind", "indus": "ind", "industrie": "ind",
    "brothers": "bros", "brother": "bros", "bro": "bros", "bros": "bros", "freres": "bros",
    "associates": "assoc", "associate": "assoc", "assocs": "assoc", "associes": "assoc",
    "association": "assn",
    "national": "natl", "nationale": "natl",
    "management": "mgmt", "mgt": "mgmt",
    "development": "dev", "developments": "dev", "devt": "dev", "developpement": "dev",
    "group": "grp", "groupe": "grp", "grp": "grp",
    "holdings": "hldg", "holding": "hldg", "hldgs": "hldg",
    "engineering": "engg", "eng": "engg", "ingenierie": "engg",
    "engineers": "engr", "engineer": "engr", "engrs": "engr",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharm": "pharma",
    "laboratories": "lab", "laboratory": "lab", "labs": "lab", "laboratoire": "lab",
    "laboratoires": "lab",
    "department": "dept", "government": "govt", "university": "univ",
    "institute": "inst", "institut": "inst",
    "hospital": "hosp", "medical": "med",
    "center": "ctr", "centre": "ctr", "cntr": "ctr",
    "distributors": "dist", "distributor": "dist", "distribution": "dist", "distr": "dist",
    "distrib": "dist",
    "general": "gen", "generale": "gen",
    "financial": "fin", "finance": "fin", "financiere": "fin",
    "investments": "invest", "investment": "invest", "invt": "invest",
    "marketing": "mktg",
    "products": "prod", "product": "prod", "prods": "prod", "produits": "prod",
    "trading": "trdg", "traders": "trdr", "trader": "trdr", "trdrs": "trdr",
    "construction": "constr", "constructions": "constr", "const": "constr",
    "contractors": "contr", "contractor": "contr",
    "electricals": "elec", "electrical": "elec", "electric": "elec", "electrique": "elec",
    "saint": "st", "sainte": "st", "st": "st",
    "mount": "mt", "fort": "ft",
    "and": "and", "et": "and", "und": "and",
    "shri": "sri", "shree": "sri", "sree": "sri", "shrii": "sri", "sri": "sri",
    # romanised Indic spellings of legal forms (after romanize_indic)
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "prayvet": "pvt", "pra": "pvt",
    "limitet": "ltd", "limatid": "ltd", "limitid": "ltd", "limited": "ltd", "li": "ltd",
    "kanpani": "co", "kampani": "co", "kanpni": "co",
}

LEGAL = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc", "pllc", "pc", "lllp",
    "gmbh", "ag", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scs", "sca", "selarl",
    "scop", "eirl", "bv", "nv", "srl", "spa", "pty", "opc", "pllp", "llp",
}
STOP = {"the", "and", "of", "a", "an", "le", "la", "les", "l", "de", "du", "des", "d",
        "ms", "messrs",                                   # "M/s" = Messrs (Indian prefix)
        "sri", "smt", "shrimati", "mr", "mrs", "dr", "esq", "cpa", "md", "dds", "phd", "jr"}
# generic words the data adds / swaps at the ends of names ("X Services", "Center X", "X Partners")
EDGE_NOISE = {"svc", "ctr", "partner", "grp", "hldg", "ent", "com"}

ADDR_CANON = {
    "street": "st", "str": "st", "strt": "st", "st": "st", "saint": "st",
    "suite": "ste", "sainte": "ste", "ste": "ste",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "av": "ave", "avn": "ave", "aven": "ave", "ave": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd", "blvd": "blvd",
    "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct", "crt": "ct",
    "place": "pl", "pce": "pl", "plaza": "plz", "square": "sq",
    "highway": "hwy", "hway": "hwy", "parkway": "pkwy", "pky": "pkwy",
    "freeway": "fwy", "expressway": "expy", "circle": "cir", "trail": "trl",
    "terrace": "ter", "terr": "ter", "crescent": "cres",
    "apartment": "apt", "apartments": "apt", "apts": "apt", "appt": "apt", "appartement": "apt",
    "floor": "fl", "flr": "fl", "etage": "fl", "room": "rm",
    "building": "bldg", "bldng": "bldg", "batiment": "bldg", "bat": "bldg",
    "block": "blk", "tower": "twr",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "nord": "n", "sud": "s", "ouest": "w",
    "near": "nr", "nearby": "nr", "nr": "nr",
    "opposite": "opp", "oppo": "opp", "opp": "opp",
    "behind": "bhd", "bhnd": "bhd", "beside": "bsd", "besides": "bsd",
    "sector": "sec", "sect": "sec", "phase": "ph", "nagar": "ngr", "market": "mkt",
    "chowk": "chk", "marg": "mrg", "extension": "extn", "ext": "extn",
    "colony": "col", "clny": "col", "complex": "cplx", "cmplx": "cplx",
    "industrial": "indl", "ind": "indl", "indl": "indl",
    "estate": "est", "estt": "est",
    "district": "dist", "distt": "dist", "taluka": "tal", "tq": "tal",
    "cross": "crs", "cours": "crs",
    "mount": "mt", "fort": "ft",
    "chemin": "ch", "che": "ch", "chem": "ch", "impasse": "imp", "allee": "all",
    "faubourg": "fbg", "route": "rte", "quai": "qu", "r": "rue",
    "center": "ctr", "centre": "ctr",
    # pure noise markers (numbering formats: "No. 12", "#12", "Plot 12", "Door 12") -> dropped
    "number": "", "no": "", "num": "", "nbr": "", "hno": "", "plot": "", "door": "",
    "flat": "", "unit": "", "shop": "", "hn": "", "null": "", "none": "", "na": "", "nan": "",
    "city": "", "po": "", "box": "",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "tg": "ts", "cove": "cv",
}

# multi-word phrases, applied on the punctuation-free string before tokenising
# multi-word phrases, applied on the punctuation-free string before tokenising
ADDR_PHRASES = {
    **{p: " nr " for p in ["next to", "close to", "adjacent to", "in front of", "pres de",
                           "pres du", "pres des", "a cote de", "a cote du", "a proximite de"]},
    **{p: " opp " for p in ["opposite to", "en face de", "en face du", "en face des",
                            "vis a vis de"]},
    "derriere": " bhd ",
    **{p: " " for p in ["pin code", "zip code", "postal code", "post code", "code postal",
                        "pin", "zip"]},
    # a few Indian city renamings (old -> current)
    "bangalore": " bengaluru ", "bombay": " mumbai ", "madras": " chennai ",
    "calcutta": " kolkata ", "gurgaon": " gurugram ", "poona": " pune ", "baroda": " vadodara ",
    "trivandrum": " thiruvananthapuram ", "cochin": " kochi ", "mysore": " mysuru ",
    "mangalore": " mangaluru ", "pondicherry": " puducherry ", "orissa": " od ",
    "simla": " shimla ", "allahabad": " prayagraj ", "gurgoan": " gurugram ",
    "new delhi": " dl ", "delhi": " dl ", "keralam": " kl ", "tamilnatu": " tn ",
    "tamil natu": " tn ",
    # Indian states -> short codes
    "maharashtra": " mh ", "karnataka": " ka ", "tamil nadu": " tn ", "uttar pradesh": " up ",
    "madhya pradesh": " mp ", "andhra pradesh": " ap ", "himachal pradesh": " hp ",
    "west bengal": " wb ", "gujarat": " gj ", "rajasthan": " rj ", "kerala": " kl ",
    "telangana": " ts ", "haryana": " hr ", "punjab": " pb ", "bihar": " br ",
    "odisha": " od ", "jharkhand": " jh ", "chhattisgarh": " cg ", "uttarakhand": " uk ",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
ADDR_PHRASES.update({k: f" {v} " for k, v in US_STATES.items()})
_PHRASE_LOOKUP = {k.replace(" ", ""): v for k, v in ADDR_PHRASES.items()}
_PHRASE_RE = re.compile(r"\b(?:" + "|".join(
    re.escape(k).replace(" ", " ?") for k in sorted(ADDR_PHRASES, key=len, reverse=True)) + r")\b")


def _phrase_sub(m):
    return _PHRASE_LOOKUP.get(m.group(0).replace(" ", ""), " ")


LANDMARK = {"nr", "opp", "bhd", "bsd"}

# ----------------------------------------------------------------------------- helpers
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ALPHA_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_APOS = re.compile(r"[\u2019'`\u00b4]")
_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d b a|dba|doing business as|trading as|t\s*/\s*a"
                  r"|a\s*/\s*k\s*/\s*a|aka|formerly known as|formerly)\b")
_PVT_PAREN = re.compile(r"\(\s*p\s*\)")
_ZIP4 = re.compile(r"\b(\d{5})\s*-\s*\d{4}\b")
_ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th)\b")
_PIN_SPLIT = re.compile(r"\b(\d{3})\s(\d{3})\b(?=\s*(?:,|$))")
_VOWELS = re.compile(r"[aeiou]")
_REPEAT = re.compile(r"(.)\1+")
_SKEL_RULES = [("ksh", "x"), ("sch", "s"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"),
               ("bh", "b"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("jh", "j"), ("ck", "k"),
               ("q", "k"), ("w", "v"), ("v", "b"), ("z", "j"), ("y", "i")]


# ----------------------------------------------------------------------------- Indic -> Latin
# All Brahmic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu,
# Kannada, Malayalam) share the same internal layout, so one hand-written offset table
# romanises all of them: "सूर्या सॉफ्टवेयर प्राइवेट लिमिटेड" -> "surya sophtaveyar praivet limited".
_I_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
           0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
           0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
           0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
           0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
           0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
           0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_I_VOW = {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
          0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o",
          0x12: "o", 0x13: "o", 0x14: "au", 0x50: "om", 0x60: "ri", 0x61: "li"}
_I_MATRA = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
            0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
            0x4C: "au", 0x62: "li", 0x63: "li"}
_I_NUKTA = {"j": "z", "ph": "f", "d": "r", "dh": "rh", "k": "q"}
_CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}   # Malayalam
_SCHWA_DROP = {0x0900, 0x0980, 0x0A00, 0x0A80}   # Hindi-like scripts drop the final "a"
_INDIC_RE = re.compile(r"[\u0900-\u0DFF]")


def romanize_indic(text):
    if not _INDIC_RE.search(text):
        return text
    out, pending, pend_base, cluster, after_virama = [], False, 0, False, False
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0DFF:
            base = cp - ((cp - 0x0900) % 0x80)
            off = cp - base
            if (base == 0x0D00 and off in _CHILLU) or (base == 0x0980 and off == 0x4E):
                if pending:
                    out.append("a")
                out.append(_CHILLU.get(off, "t"))
                pending = after_virama = False
            elif off in _I_CONS:
                if pending:
                    out.append("a")
                out.append(_I_CONS[off])
                pending, pend_base, cluster, after_virama = True, base, after_virama, False
            elif off in _I_MATRA:
                out.append(_I_MATRA[off])
                pending = after_virama = False
            elif off == 0x4D:                                   # virama: no inherent vowel
                pending, after_virama = False, True
            elif off == 0x3C and out:                           # nukta
                out[-1] = _I_NUKTA.get(out[-1], out[-1])
            elif off in (0x01, 0x02) or (base == 0x0A00 and off == 0x70):
                if pending:
                    out.append("a")
                out.append("n")
                pending = after_virama = False
            elif off == 0x03:
                if pending:
                    out.append("a")
                out.append("h")
                pending = after_virama = False
            elif off in _I_VOW or 0x66 <= off <= 0x6F:
                if pending:
                    out.append("a")
                out.append(_I_VOW[off] if off in _I_VOW else str(off - 0x66))
                pending = after_virama = False
            elif off in (0x64, 0x65):
                out.append(" ")
                pending = after_virama = False
        elif cp in (0x200C, 0x200D):                            # zero-width (non-)joiner
            continue
        else:
            if pending and (cluster or pend_base not in _SCHWA_DROP):
                out.append("a")
            pending = after_virama = False
            out.append(ch)
    if pending and (cluster or pend_base not in _SCHWA_DROP):
        out.append("a")
    return "".join(out)


_OCR_DIGIT = re.compile(r"(?<=[a-z])[01358](?=[a-z])")      # r0dgers, hea1th, comp1ete
_OCR_LEAD = re.compile(r"(?<![a-z0-9])([015])(?=[a-z])(?!(?:st|nd|rd|th)\b)")   # 5ams -> sams
TRANSLIT_MAP = {}   # romanised Indic token -> English token, LEARNED from training pairs
_OCR_MAP = {"0": "o", "1": "l", "3": "e", "5": "s", "8": "b"}
_DOMAIN = re.compile(r"(?:https?\s*:\s*/+\s*)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\s*\.\s*"
                     r"(?:com|net|org|co\.in|in|co|biz|info|us|fr|io|shop|store)\b")
_L_AS_I = re.compile(r"^l(?=[bcdfghjkmnpqrstvwxz])")          # lndia, lnc, lnfra (I read as l)


def ocr_fix_token(t, table):
    """'lndia' -> 'india' (capital I printed as l); known words such as 'ltd' are left alone."""
    return _L_AS_I.sub("i", t) if len(t) > 2 and t not in table else t


def strip_accents(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def base_clean(text):
    if not isinstance(text, str):
        return ""
    text = strip_accents(romanize_indic(text)).lower()
    text = _APOS.sub("", text)
    return text.replace("&", " and ").replace("+", " and ").replace("@", " ")


def tokenize(text):
    text = _ALPHA_DIGIT.sub(" ", text)
    return [t for t in _NON_ALNUM.split(text) if t]


def merge_single_letters(tokens):
    """'l l c' -> 'llc', 's v road' -> 'sv road' (dotted abbreviations)."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def stem(t):
    if t.isdigit() or len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith(("ss", "us", "is", "os")):
        return t[:-1]
    return t


def canon(t, table):
    v = table.get(t)
    if v is not None:
        return v
    s = stem(t)
    return table.get(s, s)


def skeleton_token(t):
    """Crude phonetic key that survives transliteration: lakshmi/laxmi -> lxm."""
    if not t or t.isdigit():
        return t
    for a, b in _SKEL_RULES:
        t = t.replace(a, b)
    t = _REPEAT.sub(r"\1", t)
    return _VOWELS.sub("", t) or t[0]


def split_legal(tokens):
    """Strip legal forms / stop words / generic edge words from both ends.
    Returns (core, legal_tokens, generic_edge_words)."""
    t = list(tokens)
    legal, extra = [], []

    def take(x):
        if x in LEGAL:
            legal.append(x)
        elif x in EDGE_NOISE:
            extra.append(x)

    while t and (t[0] in LEGAL or t[0] in STOP or t[0] in EDGE_NOISE):
        take(t.pop(0))
    while t and (t[-1] in LEGAL or t[-1] in STOP or t[-1] in EDGE_NOISE):
        take(t.pop())
    core = [x for x in t if x not in STOP]
    if not core:
        core = [x for x in tokens if x not in STOP and x not in LEGAL] or list(tokens)
    return core, sorted(set(legal)), sorted(set(extra))


# ----------------------------------------------------------------------------- names
def name_tokens_raw(raw):
    """Tokens after romanisation / cleaning but BEFORE canonical mapping (used to learn the
    transliteration map)."""
    s = base_clean(raw)
    s = _DOMAIN.sub(r" \1 ", s).replace(".", " ")
    return [t for seg in s.split(",") for t in merge_single_letters(tokenize(seg))]


def normalize_name(raw):
    """-> (core, legal, aliases, extra).  core = canonical tokens without legal forms / stop
    words / generic edge words, legal = legal-form tokens, aliases = ' | '-joined DBA / trade
    names ('' if none), extra = generic edge words that were stripped (Group, Enterprises ...)."""
    s = base_clean(raw)
    s = _DOMAIN.sub(r" \1 ", s)                      # keystonedaedalus.com -> keystonedaedalus
    s = _OCR_DIGIT.sub(lambda m: _OCR_MAP[m.group(0)], s)
    s = _OCR_LEAD.sub(lambda m: _OCR_MAP[m.group(1)], s)
    s = _PVT_PAREN.sub(" pvt ", s).replace("(india)", " ").replace(".", " ")
    parts = [p for p in _DBA.split(s) if p and p.strip()] or [s]
    core, legal, extra, aliases = [], [], [], []
    tm = TRANSLIT_MAP
    for i, part in enumerate(parts):
        toks = [canon(ocr_fix_token(tm.get(t, t), NAME_CANON), NAME_CANON)
                for seg in part.split(",") for t in merge_single_letters(tokenize(seg))]
        toks = [t for t in toks if t]
        c, lg, ex = split_legal(toks)
        if i == 0:
            core, legal, extra = c, lg, ex
        if c:
            aliases.append(" ".join(c))
    return (" ".join(core), " ".join(legal), " | ".join(aliases) if len(aliases) > 1 else "",
            " ".join(extra))


def learn_translit_map(records, truth, min_count=3, min_share=0.5, log=print):
    """Data-driven transliteration dictionary: in true pairs where one name is written in an
    Indic script and the other in Latin script with the same number of tokens, align the tokens
    position by position ("sarvisej" <-> "services"). Uses only the training ground truth."""
    from collections import defaultdict
    name = dict(zip(records["entity_id"].values, records["business_name"].values))
    cnt, latin = defaultdict(Counter), Counter()
    for s1, ms in bar(truth.items(), "learn transliteration map", total=len(truth), unit="S1"):
        a = name.get(s1)
        if not isinstance(a, str) or _INDIC_RE.search(a):
            continue
        ta = name_tokens_raw(a)
        latin.update(ta)
        for m in ms:
            b = name.get(m)
            if not isinstance(b, str) or not _INDIC_RE.search(b):
                continue
            tb = name_tokens_raw(b)
            if len(ta) == len(tb):
                for x, y in zip(tb, ta):
                    if x != y and not x.isdigit():
                        cnt[x][y] += 1
    out = {}
    for x, c in cnt.items():
        y, k = c.most_common(1)[0]
        if k >= min_count and k / sum(c.values()) >= min_share and latin[x] < 3:
            out[x] = y
    log(f"    learned transliteration map: {len(out):,} tokens, e.g. "
        + ", ".join(f"{k}->{v}" for k, v in sorted(out.items(), key=lambda kv: -sum(cnt[kv[0]].values()))[:12]))
    return out


# ----------------------------------------------------------------------------- addresses
def normalize_address(raw):
    """-> (addr_norm, postal, landmark_flag)."""
    s = base_clean(raw)
    s = _ZIP4.sub(r"\1", s)
    s = _PIN_SPLIT.sub(r"\1\2", s)
    s = _ORDINAL.sub(r"\1", s)                                 # 12th -> 12, 3rd -> 3
    segs = [" " + _NON_ALNUM.sub(" ", seg.replace(".", " ")) + " " for seg in s.split(",")]
    toks = []
    for seg in segs:
        seg = _PHRASE_RE.sub(_phrase_sub, seg)
        toks += [canon(ocr_fix_token(t, ADDR_CANON), ADDR_CANON)
                 for t in merge_single_letters(tokenize(seg))]
    toks = [t for t in toks if t]
    postal = ""
    pos = [i for i, t in enumerate(toks) if t.isdigit() and len(t) in (5, 6) and t[0] != "0"]
    if pos and (pos[-1] != 0 or len(toks) == 1):
        postal = toks[pos[-1]]
    toks = [t if (not t.isdigit() or t == postal) else (t.lstrip("0") or "0") for t in toks]
    lm = int(any(t in LANDMARK for t in toks))
    return " ".join(toks), postal, lm


# ----------------------------------------------------------------------------- derived views
def name_view(core):
    """tokens, skeleton string, number set, acronym - derived from the stored core name."""
    toks = core.split()
    alpha = [t for t in toks if not t.isdigit()]
    return (tuple(toks), " ".join(skeleton_token(t) for t in toks),
            frozenset(t for t in toks if t.isdigit()),
            "".join(t[0] for t in alpha) if len(alpha) >= 2 else "")


def addr_view(addr, postal):
    """alpha tokens, non-postal numbers, first number, skeleton string."""
    toks = addr.split()
    nums = [t for t in toks if t.isdigit()]
    if postal and postal in nums:
        nums.reverse()
        nums.remove(postal)       # drop the (last) postal occurrence
        nums.reverse()
    alpha = tuple(t for t in toks if not t.isdigit())
    skel = " ".join(skeleton_token(t) for t in alpha if t not in LANDMARK)
    return alpha, frozenset(nums), (nums[0] if nums else ""), skel


# ----------------------------------------------------------------------------- profiles
NORM_VERSION = 4   # bump when normalisation changes (invalidates cached profiles)
PROFILE_COLS = ["name_core", "name_legal", "name_alias", "name_extra", "addr_norm",
                "addr_postal", "addr_landmark"]


def _norm_chunk(args):
    names, addrs, tmap = args
    TRANSLIT_MAP.clear()
    TRANSLIT_MAP.update(tmap)
    return [normalize_name(n) + normalize_address(a) for n, a in zip(names, addrs)]


def build_profiles(records, n_jobs=1, chunk=50_000):
    """records: DataFrame(entity_id, business_name, business_address, country, src)."""
    names = records["business_name"].tolist()
    addrs = records["business_address"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], dict(TRANSLIT_MAP))
            for i in range(0, len(names), chunk)]
    if n_jobs > 1 and len(jobs) > 1:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            parts = list(bar(pool.imap(_norm_chunk, jobs), "normalise records",
                             total=len(jobs), unit="chunk"))
    else:
        parts = [_norm_chunk(j) for j in bar(jobs, "normalise records", unit="chunk")]
    prof = pd.DataFrame([r for p in parts for r in p], columns=PROFILE_COLS)
    prof.insert(0, "entity_id", records["entity_id"].values)
    prof["src"] = records["src"].values.astype(np.int8)
    prof["country"] = records["country"].values
    prof["country_key"] = prof["country"].astype(str).str.strip().str.lower()
    # original text is kept for the cross-encoder: it must see "Group", "Enterprises", the native
    # script, house-number formats ... everything normalisation deliberately throws away
    prof["raw_name"] = records["business_name"].values
    prof["raw_addr"] = records["business_address"].values
    prof["addr_landmark"] = prof["addr_landmark"].astype(np.int8)
    return prof


def token_idf(token_iter, n_docs, desc="token IDF"):
    """Smoothed IDF over a corpus (computed per split, so an unseen country in test gets
    its own statistics)."""
    df = Counter()
    for toks in bar(token_iter, desc, total=n_docs, unit="rec"):
        df.update(set(toks))
    idf = {t: float(np.log((n_docs + 1) / (c + 1)) + 1.0) for t, c in df.items()}
    return idf, float(np.log(n_docs + 1) + 1.0)

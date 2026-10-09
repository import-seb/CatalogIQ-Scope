"""Product-only comparison evidence for the rule refinement.

The strings here are a temporary matching view. Source attributes, cleaning and
the frozen shared evidence used by the TF-IDF baseline are not changed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re

from .split_matching import CORE_NOISE, canonical_brand, matching_text
from .splitting import grouping_view

# Roles/forms and general advertising language cannot establish a family by
# themselves. Unknown product-line words are deliberately retained.
EXTRA_NOISE = frozenset("""per contains containing promotes promote helps help
nutrient nutrients absorption absorb total super flavor flavoured flavored
unflavored unflavoured friendly keto packaging may vary coated mini maximum max
strong extra ultraquick strength odor odour nsaid anti topical temporary
treat treats treatment frequent heartburn acid reducer relief reliever pain
aches muscle muscles joint joints back lower elbow hands hand feet foot
ear ears eye eyes nose nasal hearing protection noise reduction infection
infections prevention removal wax cleaner clean tool tools remedy earache
swimmer swimmers dog pets adults tablets multivitamins cap caps veggie
gels patch patches spray sprays cream creams rub rubs roll roller on
softgel lozenge lozenges pad pads wrap wraps heatwrap heatwraps bottle
bottles tablet capsule gummy chewable liquid powder oil drops eardrops
therapeutic therapy comfortable comfort fit stretch compact system
dynamic filtering filtered construction sleeping concerts hunting shooting
motorcycles lawn care storage box portable upgrade detachable safe scoop pick
made usa certified usda lab verified tested laboratory non gmo gluten sugar
free additive filler fillers artificial synthetic chemical chemicals
plant based plantbased animal derived animalderived color colors colour colours
dye dyes preservative preservatives flavor flavors flavour flavours fragrance
fragrances binders allergens sweetener sweeteners
""".split())
NOISE = CORE_NOISE | EXTRA_NOISE
FORMS = frozenset("capsule tablet softgel gummy patch spray cream gel lozenge pad liquid powder oil drops eardrops".split())


def remove_negative_claims(value: str) -> str:
    """Remove bounded negative lists, including both items in 'no X or Y'.

    Punctuation is a boundary before lexical normalization; a following positive
    clause (with/contains) is also retained. Negative claims must never provide
    either name-core or active-ingredient evidence.
    """
    value = str(value)
    boundaries = r"(?=\b(?:with|contains|containing|provides|providing)\b|[|;,().]|\s+[-–—]\s+|$)"
    terminal = r"(?:fillers?|colors?|colours?|dyes?|additives?|preservatives?|flavou?rs?|fragrances?|binders?|allergens?|sweeteners?)"
    body = r"(?:(?!\b(?:with|contains|containing|provides|providing)\b|[|;,().]|\s+[-–—]\s+).)*?"
    # A claim-object noun terminates the negative list even when a retailer
    # omits punctuation: "No artificial colors Zinc tablets" retains Zinc.
    # Coordinated claim objects remain part of the same negative list.
    value = re.sub(r"\b(?:no|without|free\s+from|free\s+of)\s+" + body + r"\b" + terminal
                   + r"\b(?:\s+(?:and|or)\s+" + body + r"\b" + terminal + r"\b)*",
                   " ", value, flags=re.I)
    value = re.sub(r"\b(?:no|without|free\s+from|free\s+of)\s+[^|;,().]+?" + boundaries,
                   " ", value, flags=re.I)
    # The compact suffix form is common on supplier templates.
    value = re.sub(r"\b(?:[\w]+(?:\s+(?:and|or)\s+[\w]+)*)[-\s]+free\b", " ", value, flags=re.I)
    return value


def comparison_name(value, strip_package_sizes=True) -> str:
    if value is None:
        return ""
    # Normalize x quantities before punctuation removes the multiplication sign.
    value = re.sub(r"\b\d+\s*[x×]\s*(\d+(?:\.\d+)?\s*(?:ml|oz|g))\b", r"\1", str(value), flags=re.I)
    value = remove_negative_claims(value)
    value = matching_text(value)
    # A letter after "vitamin" is substantive, while a bare conjunction/size
    # letter is not. Keep the feature attached to its lexical role.
    value = re.sub(r"\bvitamin\s+([abdekc]\d{0,2})\b", r"vitamin\1", value)
    aliases = {"lozenges": "lozenge", "pads": "pad", "wraps": "wrap", "heatwraps": "heatwrap",
               "micronised": "micronized", "flavoured": "flavored", "unflavoured": "unflavored",
               "odour": "odor", "max": "maximum", "advance": "advanced", "multivitamins": "multivitamin",
               "caps": "capsule", "cap": "capsule"}
    value = " ".join(aliases.get(t, t) for t in value.split())
    # Package counts can appear before or after a form, at either title end.
    units = r"(?:count|capsule|tablet|caplet|softgel|gummy|piece|bottle|pack|serving|lozenge|pad|wrap|each|ea)"
    value = re.sub(r"\bpack\s+(?:of\s+)?\d+(?:\.\d+)?\b", " ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\s*" + units + r"\b", " ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:day|month|year)s?\s+supply\b", " ", value)
    if strip_package_sizes:
        value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:fl\s+oz|oz|ml|g)\b", " ", value)
    value = re.sub(r"\b(?:per\s+pack|packaging\s+may\s+vary|total|each|ea)\b", " ", value)
    return " ".join(value.split())


def _core_word(token):
    return (token not in NOISE and len(token) > 1
            and not re.fullmatch(r"\d+(?:\.\d+)?", token))


@dataclass(frozen=True)
class RuleProductEvidence:
    name: str
    brand: str
    tokens: frozenset[str]
    core: frozenset[str]
    ordered_core: tuple[str, ...]
    head: tuple[str, ...]
    strengths: frozenset[str]
    branded_generic_line: bool
    anchors: frozenset[str]
    payload_key: str


def prepare_rule_products_v3(frame, split_config, config):
    """Use allowlisted fields only; learn repeated *body* advertising tokens.

    A token is removable only when it recurs across many different leading
    product families and rarely occurs in those families' leading core. Large
    legitimate families with the same head do not trigger this filter.
    """
    view = grouping_view(frame)
    rows = []
    for name, raw_brand in zip(view["ProductName"], view["ProductBrand"]):
        name = comparison_name(name, split_config.family_strip_package_sizes)
        brand = canonical_brand(raw_brand, split_config)
        # Match original declared brand words as well as its canonical suffix.
        brand_words = set(matching_text(raw_brand).split()) | set(brand.split())
        brand_stems = {t[:-1] if len(t) > 4 and t.endswith("s") else t for t in brand_words}
        ordered = tuple(dict.fromkeys(t for t in name.split() if _core_word(t) and t not in brand_words
                         and (t[:-1] if len(t) > 4 and t.endswith("s") else t) not in brand_stems))
        rows.append((name, brand, brand_words, ordered))
    by_brand = defaultdict(list)
    for i, (_, brand, _, _) in enumerate(rows):
        if brand:
            by_brand[brand].append(i)
    template_tokens = {}
    template_counts = Counter()
    for brand, indices in by_brand.items():
        unique = {rows[i][0]: rows[i][3] for i in indices}
        heads = {words[:config.line_anchor_tokens] for words in unique.values() if words}
        if len(unique) < config.template_min_titles or len(heads) < config.template_min_distinct_heads:
            continue
        df = Counter(t for words in unique.values() for t in set(words))
        head_df = Counter(t for words in unique.values() for t in set(words[:config.line_anchor_tokens]))
        removable = {t for t, count in df.items()
                     if count / len(unique) >= config.template_document_fraction
                     and head_df[t] / len(unique) <= config.template_head_fraction}
        if removable:
            template_tokens[brand] = removable
            template_counts["brands_with_repeated_body_template"] += 1
            template_counts["repeated_body_template_tokens"] += len(removable)
    products = []
    payloads = list(view[["ProductName", "ProductBrand", "ProductDescription", "ProductContents"]]
                    .itertuples(index=False, name=None))
    for i, (name, brand, brand_words, ordered) in enumerate(rows):
        ordered = tuple(t for t in ordered if t not in template_tokens.get(brand, ()))
        core = frozenset(ordered)
        head = ordered[:config.line_anchor_tokens]
        tokens = frozenset(name.split())
        strengths = frozenset(re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|iu|cfu)\b", name))
        brand_in_title = bool(brand and set(brand.split()).issubset(tokens))
        # A branded, exactly repeated formulation line can survive an empty
        # residual core; a generic slogan or unbranded form cannot.
        generic_line = bool(brand_in_title and any(t not in NOISE for t in brand.split())
                            and len(tokens - brand_words) >= config.generic_line_min_tokens
                            and tokens & FORMS
                            and re.search(r"\b(?:maximum strength|extra strength|delayed release|extended release|rapid release|time release)\b", name))
        anchors = {"word:" + t for t in core}
        if generic_line:
            anchors.add("branded_exact:" + brand + "|" + name)
        payload = tuple(str(value).strip() for value in payloads[i])
        payload_key = (hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                        separators=(",", ":")).encode()).hexdigest()
                       if any(value.casefold() not in {"", "null"}
                              for value in (payload[0], payload[2], payload[3])) else "")
        if payload_key:
            anchors.add("payload:" + payload_key)
        products.append(RuleProductEvidence(name, brand, tokens, core, ordered, head,
                                            strengths, generic_line, frozenset(anchors), payload_key))
    return products, dict(template_counts)

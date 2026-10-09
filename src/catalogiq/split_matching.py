"""Comparison-only product-family evidence shared by experimental v2 matchers.

No source correction, target labels, diagnostic pair IDs, or product-specific
alias table. The small ingredient vocabulary is a general lexical conflict
guard, not a medical ontology or a claim that missing signatures are equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import re
import unicodedata

import pandas as pd

from .splitting import grouping_view, jaccard

# Common OTC active names, independent of brands/products in the diagnostic set.
ACTIVE_NAMES = frozenset("""acetaminophen ibuprofen aspirin naproxen lidocaine
benzocaine hydrocortisone clotrimazole miconazole diphenhydramine doxylamine
loratadine cetirizine fexofenadine chlorpheniramine famotidine omeprazole
lansoprazole esomeprazole loperamide docusate bisacodyl senna glycerin
phenylephrine pseudoephedrine dextromethorphan guaifenesin menthol simethicone
carboxymethylcellulose polyethylene glycol bismuth subsalicylate""".split())
MINERALS = frozenset("magnesium calcium potassium zinc ferrous iron sodium".split())
SALTS = frozenset("oxide citrate glycinate bisglycinate carbonate gluconate fumarate chloride sulfate".split())
# Marketing scaffolds are not evidence of the named formula. Keep informative
# unknown words (including rare named formulas) without a vocabulary cap.
CORE_NOISE = frozenset("""a an the and or for with of in to at by from on as is it
your our their all no non not free plus extra maximum strength original natural
organic pure always premium advanced high quality dietary supplement supplements
vitamin vitamins nutrition nutritional ingredient ingredients active proven
support supports supporting health healthy wellness daily day month year supply
serving servings vegetarian vegan gluten gmo sugar additive additives filler
fillers lab verified laboratory tested certified kosher formula blend complex
capsule capsules tablet tablets caplet caplets softgel softgels gel gels gummy
gummies bottle bottles pack packs count ct piece pieces pcs total each ea refill
ounce ounces oz fl ml gram grams g milligram milligrams mg mcg iu cfu billion
pain relief reliever fever reducer medicine otc drug drugfree women woman womens
men man mens adult adults child children childrens kids baby chewable chewables
quick dissolve dissolving fast slow release sustained extended liquid powder
oil store visit inc incorporated llc ltd company co products product research
formulas made usa laboratory best value joint hair skin nail nails immunity
energy immune digestion digestive function cellular easy nonhabit forming
fake tooth teeth repair kit""".split())


@dataclass(frozen=True)
class ProductEvidence:
    name: str
    family: str
    brand: str
    tokens: frozenset[str]
    core: frozenset[str]
    ingredients: frozenset[str]


def matching_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.casefold() in {"", "null"}:
        return ""
    value = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    # Possessive punctuation must not create an extra one-letter word.
    value = re.sub(r"['’‘ʼ`]+", "", value)
    value = re.sub(r"\b([abdek])\s*[- ]\s*(\d{1,3})\b", r"\1\2", value)
    value = re.sub(r"[^\w.]+", " ", value)
    value = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", value)
    value = value.replace("_", " ")
    value = re.sub(r"(?<=\d)(?=(?:mg|mcg|ml|oz|iu|cfu|g|ct|count|capsules?|tablets?|caplets?|softgels?|gummies|packs?|pcs)\b)", " ", value)
    aliases = {"ct": "count", "counts": "count", "capsules": "capsule", "tablets": "tablet",
               "softgels": "softgel", "gummies": "gummy", "bottles": "bottle", "packs": "pack",
               "pcs": "piece", "pieces": "piece", "servings": "serving", "caplets": "caplet",
               "ounces": "oz", "ounce": "oz", "grams": "g", "milligrams": "mg", "micrograms": "mcg"}
    return " ".join(aliases.get(t, t) for t in value.split())


def family_text(value, config) -> str:
    value = matching_text(value)
    # Preserve strength units (mg/mcg/IU) and formulas; remove only quantities
    # describing packaging or total supply in this related-family view.
    value = re.sub(r"\bpack\s+(?:of\s+)?\d+(?:\.\d+)?\b", " ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:count|capsule|tablet|caplet|softgel|gummy|piece|bottle|pack|serving)\b", " ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:day|month|year)s?\s+supply\b", " ", value)
    if config.family_strip_package_sizes:
        value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:fl\s+oz|oz|ml|g)\b", " ", value)
    value = re.sub(r"\b(?:total|each|ea)\b", " ", value)
    return " ".join(value.split())


def canonical_brand(value, config) -> str:
    value = matching_text(value)
    value = re.sub(r"\s+(?:com|net|org)$", "", value)
    value = re.sub(r"(?<=[a-z])usa$", "", value) if len(value) >= 8 else value
    # Generic organizational suffixes only. Never reduce a brand to empty.
    suffixes = sorted(config.brand_alias_suffixes, key=lambda s: (-len(s.split()), s))
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if value.endswith(" " + suffix):
                root = value[:-(len(suffix) + 1)].strip()
                if root and any(t not in CORE_NOISE for t in root.split()):
                    value, changed = root, True
                    break
    return value


def brand_block_key(value):
    """Spacing variants can share candidate blocks without global brand maps."""
    return value.replace(" ", "")


def _ingredient_signature(name, contents):
    words = set(name.split())
    # A contents value is corroborating only when it explicitly identifies an
    # active section. Avoid attributing comparisons/marketing to the product.
    content = matching_text(contents)
    active = re.search(r"\bactive ingredients?\b", content)
    if active:
        section = content[active.end():].split("inactive ingredient", 1)[0]
        words.update(section.split())
    # Ignore immediate negated claims such as aspirin-free/no aspirin.
    negated = set(re.findall(r"\b(?:no|without|free from)\s+(\w+)\b", name))
    negated.update(re.findall(r"\b(\w+)\s+free\b", name))
    found = (words & ACTIVE_NAMES) - negated
    # Compound salt names distinguish formulations without an example-specific
    # botanical/product dictionary. Preserve the exact compound evidence.
    compounds = set()
    for mineral, salt in zip(name.split(), name.split()[1:]):
        if mineral in MINERALS and salt in SALTS:
            compounds.add(mineral + " " + salt)
    return frozenset(found | compounds)


def prepare_products(frame, config):
    view = grouping_view(frame)
    products = []
    for row in view.itertuples(index=False, name=None):
        values = dict(zip(view.columns, row))
        name = matching_text(values["ProductName"])
        family = family_text(values["ProductName"], config)
        brand = canonical_brand(values["ProductBrand"], config)
        tokens = frozenset(family.split())
        core = frozenset(t for t in tokens if t not in CORE_NOISE and t not in brand.split()
                         and not re.fullmatch(r"\d+(?:\.\d+)?", t))
        products.append(ProductEvidence(name, family, brand, tokens, core,
                         _ingredient_signature(name, values["ProductContents"])))
    return products


def brand_compatible(a, b, config) -> bool:
    if a.brand and brand_block_key(a.brand) == brand_block_key(b.brand):
        return True
    if not a.brand and not b.brand:
        return bool(a.family and a.family == b.family and a.core and b.core)
    return False


def _title_brand_override(a, b):
    # Evidence for a parent/child brand lives in the title. A conflicting brand
    # explicitly present in its own title is not silently treated as an alias.
    for marked, other in ((a, b), (b, a)):
        marker = set(marked.brand.split())
        if (marker and marker.issubset(marked.tokens) and marker.issubset(other.tokens)
                and (not other.brand or not set(other.brand.split()).issubset(other.tokens))):
            return True
    return False


def compatible(a, b, config, allow_brand_override=False) -> bool:
    if (a.ingredients and b.ingredients and not (a.ingredients & b.ingredients)):
        return False
    if a.core and b.core:
        overlap = len(a.core & b.core) / min(len(a.core), len(b.core))
        if overlap < config.core_min_overlap:
            return False
    if brand_compatible(a, b, config):
        return True
    return bool(allow_brand_override and _title_brand_override(a, b)
                and jaccard(a.tokens, b.tokens) >= config.brand_override_name_threshold)


def name_similarity(a, b):
    return jaccard(set(a.tokens), set(b.tokens))

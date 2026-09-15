# PUB LOG Columns + Code Keys

Simple reference for the **seven PUB LOG tables we are currently using**.

The goal here is not to document every DLA code in existence. It is to answer two questions:

1. **What does this column mean?**
2. **If the value is a code, what does that code mean?**

---

## `P_FLIS_NSN`

Basic identity of the government item.

| Column | Meaning | Key / value guide |
|---|---|---|
| `FSC` | Federal Supply Class — the item's category | 4-digit class code. Use `V_H2_FSC` to turn it into a readable category. |
| `NIIN` | National Item Identification Number — the exact catalog item | 9-digit item ID. No code key needed. |
| `INC` | Item Name Code — standardized code for the approved item name | 5-digit code. `77777` = non-approved item name. |
| `ITEM_NAME` | Human-readable item name | Plain text. No key needed. |
| `SOS` | Source of Supply | Routing/source code identifying the organization that supplies the item. There are many possible SOS codes, so this is a lookup rather than a small key. |

---

## `V_CHARACTERISTICS`

Physical and technical characteristics of an item.

| Column | Meaning | Key / value guide |
|---|---|---|
| `NIIN` | Item ID | 9-digit NIIN. |
| `MRC` | Master Requirement Code — identifies which characteristic is being described | 4-character code. There are many MRCs, so this is not a small fixed key. `REQUIREMENTS_STATEMENT` gives the readable meaning. |
| `REQUIREMENTS_STATEMENT` | The characteristic/question being described | Plain text. Example idea: material, diameter, surface treatment, etc. |
| `CLEAR_TEXT_REPLY` | The actual value/answer for that characteristic | Plain text. This is one of our main places to look for material names. |

---

## `V_FLIS_PART`

Connects a government item to manufacturer/reference part information.

| Column | Meaning | Key / value guide |
|---|---|---|
| `NIIN` | Item ID | 9-digit NIIN. |
| `PART_NUMBER` | Manufacturer/reference part number | Plain text. |
| `CAGE_CODE` | Company/organization ID | 5-character CAGE code. |
| `CAGE_STATUS` | Status of the CAGE record | DLA describes this as status such as active/restricted, but the current public page does not provide the full letter key. Use PUB LOG context help when we need the exact mapping. |
| `RNCC` | Reference Number Category Code — tells what kind of relationship this part/reference number has to the item | See key below. |
| `RNVC` | Reference Number Variation Code — tells whether the reference number actually identifies the item | See key below. |

### RNCC key

| Code | Easy meaning |
|---|---|
| `1` | Source-control reference |
| `2` | Definitive U.S. Government specification/standard |
| `3` | Design-control reference |
| `4` | Non-definitive U.S. Government specification/standard |
| `5` | Secondary reference |
| `6` | Informational reference |
| `7` | Vendor item drawing |
| `8` | U.S./NATO reproduced item identification |
| `A` | Packaging/logistics-data reference |
| `C` | Advisory reference |
| `D` | Drawing/document reference |
| `E` | Replaced reference number |

### RNVC key

| Code | Easy meaning |
|---|---|
| `1` | Reference does **not** identify the item by itself; more information is needed |
| `2` | Reference **does** identify the item |
| `3` | Vendor part number for a reparable source-control item |
| `8` | Reference to the original/replaced item after an NSN replacement |
| `9` | Inactive, obsolete, superseded, informational, or otherwise non-current reference |

---

## `V_CAGE_ADDRESS`

Company/organization identity and location.

| Column | Meaning | Key / value guide |
|---|---|---|
| `CAGE_CODE` | Company/organization ID | 5-character CAGE code. |
| `COMPANY_NAME` | Organization name | Plain text. |
| `CITY` | City | Plain text. |
| `STATE` | State/U.S. possession | Usually 2-letter postal abbreviation for U.S. records. |
| `ZIP` | ZIP/postal code | Plain value. |
| `COUNTRY` | Country | Plain text. |

---

## `V_CAGE_STATUS_AND_TYPE`

Extra information about the company/organization behind a CAGE code.

| Column | Meaning | Key / value guide |
|---|---|---|
| `CAGE_CODE` | Company/organization ID | 5-character CAGE code. |
| `STATUS` | Status of the CAGE/NCAGE record | DLA says this represents conditions such as active/restricted. Full official value key was not exposed on the current public page; use PUB LOG context help for exact letters. |
| `TYPE` | Type of organizational entity | 1-character code. Full current key not found on the public DLA page; use PUB LOG context help if needed. |
| `PARENT_CAGE` | Parent organization's CAGE | 5-character CAGE code. |
| `BUS_SIZE` | Business-size classification | 1-character code. Full current key not found in the public DLA docs we checked. |
| `PRIMARY_BUSINESS` | Primary business category | 1-character code. Full current key not found in the public DLA docs we checked. |
| `TYPE_OF_BUSINESS` | Type of business | 1-character code. Full current key not found in the public DLA docs we checked. |
| `WOMAN_OWNED` | Woman-owned business indicator | 1-character indicator. Exact PUB LOG key should be taken from PUB LOG context help rather than guessed. |

---

## `V_H2_FSC`

Turns an FSC category code into readable information.

| Column | Meaning | Key / value guide |
|---|---|---|
| `FSC` | Federal Supply Class | 4-digit category code. |
| `FSC_TITLE` | Name of the class | Plain text. |
| `FSC_NOTES` | Notes about the class | Plain text. |
| `FSC_INCLUSIONS` | Items that belong in the class | Plain text. |
| `FSC_EXCLUSIONS` | Items that do not belong in the class | Plain text. |

There is no small FSC key because there are hundreds of classes. This table **is the lookup key**.

---

## `V_FLIS_IDENTIFICATION`

Official flags and control information attached to an item.

| Column | Meaning | Key / value guide |
|---|---|---|
| `NIIN` | Item ID | 9-digit NIIN. |
| `INC` | Item Name Code | 5-digit code; `77777` = non-approved item name. |
| `CRIT_CD` | Criticality Code — tells whether the item has safety/technical critical features | See key below. |
| `DMIL` | Demilitarization Code — tells what must happen before an item leaves DoD control | See key below. |
| `DMIL_INT_CD` | DEMIL Integrity Code — tells how trustworthy/current the DEMIL assignment is | See key below. |
| `NIIN_ASGMT` | Date the NIIN was assigned | Date value. |
| `PMIC` | Precious Metals Indicator Code | See key below. |
| `HMIC` | Hazardous Material Indicator Code | See key below. |
| `HCC` | Hazardous Characteristics Code | See key below. |
| `IUID_INDICATOR` | Whether Item Unique Identification marking/tracking is required | `Y` = required; `N` = not required. |
| `LST_KWN_SOS` | Last Known Source of Supply | 3-character SOS routing/activity code. Many possible values; use an SOS lookup when needed. |

### CRIT_CD key

| Code | Easy meaning |
|---|---|
| blank | No criticality code assigned |
| `C` | Has critical features; nuclear-hardness status not determined |
| `E` | Aviation critical-safety item **and** nuclear hard |
| `F` | Aviation critical-safety item |
| `H` | Nuclear hard, but no other critical feature |
| `M` | Nuclear hard **and** has other critical features |
| `N` | No critical feature; nuclear-hardness status not determined |
| `S` | Non-aviation critical-safety item |
| `V` | Has not yet been reviewed for critical-safety/critical-application purposes |
| `X` | Not nuclear hard and no other critical feature |
| `Y` | Not nuclear hard, but has other critical feature(s) |

### DMIL key

| Code | Easy meaning |
|---|---|
| `G` | Ammunition/explosives; demilitarization required |
| `P` | Classified U.S. Munitions List item; demilitarization required |
| `F` | Demilitarization required; special instructions must come from the item/technical manager |
| `D` | Destroy item/components so they cannot be restored or repaired |
| `C` | Remove/destroy critical installed parts/key points |
| `E` | Special instructions from the DoD DEMIL Program Office |
| `B` | Munitions-list item; mutilate to scrap worldwide |
| `Q` | Commerce-controlled item; mutilation rules depend on location/integrity code |
| `A` | Low-risk EAR-controlled/EAR99 item; no DEMIL or mutilation required before release |

### DMIL_INT_CD key

| Code | Easy meaning |
|---|---|
| blank | DEMIL code has not been reviewed |
| `0` | Under review; recommended and current DEMIL codes disagree |
| `1` | Reviewed and accepted / no change needed |
| `2` | Was accepted, but the inventory manager later changed it |
| `3` | Sensitive/critical item; mutilation required worldwide |
| `4` | Could not be validated / not enough technical data |
| `5` | Service/agency coded or changed it without completing normal DDCMO collaboration |
| `6` | Non-critical/non-sensitive controlled item; mutilation required overseas |
| `7` | DDCMO forced a code change after no timely response |
| `8` | Inventory manager disagrees with DDCMO recommendation; unresolved |
| `9` | Reserved for future use |

### PMIC key

| Code | Meaning |
|---|---|
| `A` | No precious metal |
| `C` | Combination of two or more precious metals |
| `G` | Gold |
| `P` | Platinum-family metals |
| `S` | Silver |
| `U` | Precious-metal content unknown |
| `V` | Precious-metal type varies by manufacturer |

### HMIC key

| Code | Easy meaning |
|---|---|
| `Y` | Hazard information exists in HMIRS |
| `D` | No HMIRS record, but this kind of item normally should have a safety data sheet |
| `P` | No HMIRS record, but a safety data sheet may be required |
| `N` | No HMIRS record and this kind of item is not normally suspected of being hazardous |

### HCC key

| Code | Easy meaning |
|---|---|
| `A1` | Radioactive — licensed |
| `A2` | Radioactive — license exempt |
| `A3` | Radioactive — exempt/authorized |
| `B1` | Corrosive inorganic alkali |
| `B2` | Corrosive organic alkali |
| `B3` | Low-risk alkali |
| `C1` | Corrosive inorganic acid |
| `C2` | Corrosive organic acid |
| `C3` | Low-risk acid |
| `C4` | Inorganic acid + oxidizer |
| `C5` | Organic acid + oxidizer |
| `D1` | Oxidizer |
| `D2` | Oxidizer + poison |
| `D3` | Oxidizer + acidic corrosive |
| `D4` | Oxidizer + alkali corrosive |
| `E1` | Military explosive |
| `E2` | Low-risk explosive |
| `F1` | Highly flammable liquid |
| `F2` | Flammable liquid |
| `F3` | Flammable liquid, lower hazard level |
| `F4` | Flammable/combustible liquid, lower hazard level |
| `F5` | Flammable liquid + poison |
| `F6` | Flammable liquid + acidic corrosive |
| `F7` | Flammable liquid + alkali corrosive |
| `F8` | Flammable solid |
| `G1` | Poison gas, nonflammable |
| `G2` | Flammable gas |
| `G3` | Nonflammable gas |
| `G4` | Nonflammable oxidizing gas |
| `G5` | Nonflammable corrosive gas |
| `G6` | Poison + corrosive gas |
| `G7` | Poison + oxidizing gas |
| `G8` | Poison + flammable gas |
| `G9` | Poison + corrosive + oxidizing gas |
| `H1` | Hazard characteristics not yet determined |
| `K1` | Infectious substance |
| `K2` | Cytotoxic drug |
| `M1` | Magnetized material |
| `N1` | Not regulated as hazardous |
| `P1` | Regulated organic peroxide |
| `P2` | Low-risk organic peroxide |
| `R1` | Reactive + flammable chemical |
| `R2` | Water-reactive chemical |
| `T1` | Poison-inhalation hazard |
| `T2` | Poison — Packing Group I |
| `T3` | Poison — Packing Group II |
| `T4` | Poison — Packing Group III |
| `T5` | Low-risk pesticide |
| `T6` | Health hazard |
| `T7` | Carcinogen |
| `V1` | Miscellaneous hazardous material — Class 9 |
| `V2` | Nonflammable aerosol |
| `V3` | Flammable aerosol |
| `V4` | Combustible liquid |
| `V5` | High-flash-point liquid |
| `V6` | Petroleum product |
| `V7` | Environmental hazard |
| `X1` | Multiple hazards under one NSN |
| `Z1` | Article containing asbestos |
| `Z2` | Article containing mercury |
| `Z3` | Article containing PCB |
| `Z4` | Lead-acid battery, nonspillable |
| `Z5` | Nickel-cadmium battery, nonspillable |
| `Z6` | Lithium battery |
| `Z7` | Dry-cell battery |

---

# What matters most for our project

If we are trying to identify items worth investigating for critical-material recovery, these are the strongest fields to understand first:

| Field | Why it matters |
|---|---|
| `CLEAR_TEXT_REPLY` | Actual characteristic text; likely where material names appear |
| `REQUIREMENTS_STATEMENT` | Tells us what that characteristic means |
| `PMIC` | Direct signal for gold, silver, platinum-family metals, etc. |
| `HCC` | Can directly flag things like lithium, nickel-cadmium batteries, mercury, etc. |
| `HMIC` | General hazardous-material flag |
| `ITEM_NAME` | Helps us understand what the object actually is |
| `FSC` | Helps group similar item types |
| `PART_NUMBER` + `CAGE_CODE` | Connects the government item to a real part/company |

---

## Official sources used for code keys

- DLA PUB LOG / FLIS documentation
- DLA DLMS Data Dictionary
- DLA Demilitarization (DEMIL) Codes
- DLM 4000.25 Volume 2 for IUID indicator values
- DLA PUB LOG record layouts for CAGE and Reference data

For fields where a current compact public DLA value map was not found (`CAGE_STATUS`, `TYPE`, `BUS_SIZE`, `PRIMARY_BUSINESS`, `TYPE_OF_BUSINESS`, `WOMAN_OWNED`), this guide intentionally does **not** guess. PUB LOG itself has context help/literal descriptions that we can use to fill those in later.

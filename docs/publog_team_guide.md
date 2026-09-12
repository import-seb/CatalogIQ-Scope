# PUB LOG Data — Simple Team Guide

This is a quick explanation of what we found in the DLA PUB LOG download and how we are using it for the **Critical Materials Recovery** project.

The main thing to know: **PUB LOG is basically a giant catalog of government/military supply items.** It contains millions of items plus details about what they are, their characteristics, manufacturer part numbers, companies, supply classes, and other logistics information.

---

## 1. The basic IDs

### NSN — National Stock Number

An **NSN** is the full government stock number for an item.

It is 13 digits:

```text
FSC + NIIN
```

Example:

```text
1234-01-234-5678
```

- First 4 digits = **FSC**
- Last 9 digits = **NIIN**

---

### FSC — Federal Supply Class

The **FSC** tells us what general type/category of item it is.

Think:

```text
FSC = "What kind of thing is this?"
```

Examples could be categories for hardware, electrical equipment, aircraft parts, etc.

The table:

```text
V_H2_FSC
```

gives us the FSC title, notes, inclusions, and exclusions.

---

### NIIN — National Item Identification Number

The **NIIN** is the 9-digit identifier for the specific government catalog item.

Think:

```text
NIIN = "Which exact item is this?"
```

We use NIIN constantly because it connects most of the item-related PUB LOG tables together.

Example:

```text
NIIN 012345678
```

might represent one exact valve, pump, bolt, battery, aircraft component, etc.

---

### CAGE Code

A **CAGE code** identifies a company, manufacturer, supplier, or government organization.

Think:

```text
CAGE = "Which company/organization is associated with this?"
```

Example:

```text
NIIN:        012345678
PART_NUMBER: ABC-100
CAGE_CODE:   1XYZ9
```

That roughly means:

> Government item `012345678` is associated with manufacturer part `ABC-100`, from the organization identified by CAGE `1XYZ9`.

We can then use the CAGE tables to find information about that company.

---

## 2. What are all these `V_...` and `P_...` files?

The `.TAB` files are DLA's own PUB LOG data tables. They are **not normal tab-separated files** that we should open directly with Pandas.

DLA provides `Decomp.exe`, which lets us query/export these tables with simple SQL.

For example:

```text
V_CHARACTERISTICS.TAB
V_FLIS_PART.TAB
V_CAGE_ADDRESS.TAB
```

Each table stores a different type of information.

The DLA schema does not explicitly tell us what the letters `V_` and `P_` stand for, so we should not pretend we know. Practically, though:

- `P_FLIS_NSN` is a very useful item lookup table.
- The `V_...` tables contain detailed information we can join to those items.

---

## 3. The main tables we are using

### `P_FLIS_NSN`

This is our basic item lookup table.

Important fields:

```text
FSC
NIIN
INC
ITEM_NAME
SOS
```

Think of this as:

```text
"What item is this?"
```

Current September 2026 PUB LOG row count:

```text
17,005,582 rows
```

---

### `V_CHARACTERISTICS`

This is probably the **most important table for our critical-material work**.

It contains:

```text
NIIN
MRC
REQUIREMENTS_STATEMENT
CLEAR_TEXT_REPLY
```

This table answers:

```text
"What is this item physically or technically like?"
```

A single NIIN can have many characteristic rows.

For example, one item might have separate information for:

```text
MATERIAL
DIAMETER
SURFACE TREATMENT
TEMPERATURE RATING
CHEMICAL COMPOSITION
```

This is why `V_CHARACTERISTICS` has many more rows than the item table.

Current September 2026 PUB LOG row count:

```text
44,657,756 rows
```

For our project, this is where we expect to find evidence such as:

```text
NICKEL
COBALT
TITANIUM
TUNGSTEN
CHROMIUM
ALUMINUM ALLOY
...
```

Important distinction:

> `V_CHARACTERISTICS` tells us **what the item is made of / what properties it has**.

---

### `V_FLIS_PART`

This table does **not** replace `V_CHARACTERISTICS`.

It answers a different question:

```text
"What manufacturer part is associated with this government item?"
```

Important fields:

```text
NIIN
PART_NUMBER
CAGE_CODE
```

Example:

```text
NIIN:        012345678
PART_NUMBER: ABC-500
CAGE_CODE:   1XYZ9
```

So:

```text
V_CHARACTERISTICS
    -> What is the item like / what might it contain?

V_FLIS_PART
    -> What manufacturer part/company is associated with it?
```

Current September 2026 row count:

```text
16,586,217 rows
```

---

### `V_CAGE_ADDRESS`

Once `V_FLIS_PART` gives us a CAGE code, this table helps identify the organization.

Important fields include:

```text
CAGE_CODE
COMPANY_NAME
CITY
STATE
ZIP
COUNTRY
```

Think:

```text
CAGE 1XYZ9
    ->
Example Manufacturing Corp.
Miami, FL
United States
```

Current September 2026 row count:

```text
4,158,375 rows
```

---

### `V_CAGE_STATUS_AND_TYPE`

This gives extra information about the organization behind a CAGE code.

Fields include:

```text
STATUS
TYPE
PARENT_CAGE
BUS_SIZE
PRIMARY_BUSINESS
TYPE_OF_BUSINESS
WOMAN_OWNED
```

This is supporting company metadata, not our main material-detection source.

---

### `V_H2_FSC`

This is a small lookup table that explains the Federal Supply Class.

Fields include:

```text
FSC
FSC_TITLE
FSC_NOTES
FSC_INCLUSIONS
FSC_EXCLUSIONS
```

Current September 2026 row count:

```text
676 rows
```

Think:

```text
FSC -> readable category information
```

---

### `V_FLIS_IDENTIFICATION`

This contains additional official attributes about an item.

Fields we currently pull include:

```text
NIIN
INC
CRIT_CD
DMIL
DMIL_INT_CD
NIIN_ASGMT
PMIC
HMIC
HCC
IUID_INDICATOR
LST_KWN_SOS
```

This gives us extra item metadata such as criticality-related codes, demilitarization-related fields, hazardous-material indicators, assignment information, etc.

Current September 2026 row count:

```text
17,005,582 rows
```

---

## 4. How the important tables connect

The easiest mental model is:

```text
                    NIIN
                     |
              P_FLIS_NSN
              "What item?"
                     |
        +------------+------------+
        |                         |
        v                         v
V_CHARACTERISTICS           V_FLIS_PART
"What is it like?"          "What manufacturer
"What material?"             part is tied to it?"
                                  |
                                  v
                              CAGE_CODE
                                  |
                                  v
                         V_CAGE_ADDRESS
                         "What company?"
```

So if we were investigating one item:

```text
Government Item
NIIN 123456789
        |
        +-- P_FLIS_NSN
        |      ITEM_NAME -> some component
        |
        +-- V_CHARACTERISTICS
        |      MATERIAL -> NICKEL ALLOY
        |      COATING  -> CHROMIUM
        |
        +-- V_FLIS_PART
               PART_NUMBER -> XJ-774
               CAGE        -> 1XYZ9
                                  |
                                  v
                         V_CAGE_ADDRESS
                         Example Manufacturing Corp.
```

For **Critical Materials Recovery**:

- `V_CHARACTERISTICS` helps tell us **why an item might contain a critical material**.
- `V_FLIS_PART` helps connect that government item to a real-world manufacturer part.
- CAGE tables help tell us **who the associated organization is**.
- FSC tells us **what type/category of item it is**.

---

## 5. Why some tables have way more rows than others

Not every table is "one row per NIIN."

For example:

```text
P_FLIS_NSN
17.0 million rows
```

versus:

```text
V_CHARACTERISTICS
44.7 million rows
```

That happens because one item can have **many characteristics**.

Example:

```text
NIIN 123456789
    |
    +-- MATERIAL
    +-- DIAMETER
    +-- LENGTH
    +-- SURFACE TREATMENT
    +-- TEMPERATURE RATING
```

Those can each be separate characteristic records.

This is also why we have to be careful with joins. A giant naive join could duplicate rows many times.

---

## 6. How we are reading the DLA data

The PUB LOG ZIP includes an old desktop application plus the actual data.

We do **not** need to install/use the desktop application.

DLA provides:

```text
TOOLS/UTILITIES/Decomp.exe
```

`Decomp.exe` lets us query a PUB LOG table from the command line.

Example idea:

```text
select NIIN,MRC,REQUIREMENTS_STATEMENT,CLEAR_TEXT_REPLY
from V_CHARACTERISTICS
where NIIN='*'
```

The output is pipe-delimited text.

We then process that output with Python.

---

## 7. Monthly automated update design

The goal is for our PUB LOG data to update automatically every month.

Our current pipeline is:

```text
Download current PublogDVD.zip
        |
        v
Extract PUB LOG
        |
        v
Read DLA publication metadata
        |
        v
Is this a new release?
   |             |
   No            Yes
   |             |
   stop          v
            Export needed tables
                 |
                 v
            Process in chunks
                 |
                 v
            Save as Parquet
                 |
                 v
            Validate row counts
                 |
                 v
            Publish new release
```

DLA includes a built-in metadata table called:

```text
IMD_TABLES
```

It gives us:

```text
TABLE_NAME
ROWS
PUB_DATE
```

For the current package, every table we checked reports:

```text
PUB_DATE = SEP 2026
```

This gives us a simple way to know whether a monthly refresh is needed.

---

## 8. Why we use chunks instead of loading everything into Pandas

Some of these tables are huge.

For example:

```text
V_CHARACTERISTICS = 44,657,756 rows
```

Trying to load the whole thing into one Pandas DataFrame could use a massive amount of RAM.

Instead we process something like:

```text
250,000 rows
        |
        v
clean + transform
        |
        v
write Parquet
        |
        v
next 250,000 rows
```

So the whole table never has to exist in memory at once.

---

## 9. Why Parquet?

The raw DLA export is huge pipe-delimited text.

We convert it to **Parquet** because it is much better for the rest of the project:

- compressed
- faster to read
- column-based
- works well with Pandas and analytics tools
- we no longer have to repeatedly parse giant text files

The raw `.pipe` export is deleted after the Parquet version passes validation.

---

## 10. How we make sure a monthly update did not break

Before a new release becomes active, we compare the number of rows we processed with DLA's own `IMD_TABLES` row count.

For some tables, the counts may match exactly.

Example:

```text
P_FLIS_NSN

DLA expected:
17,005,582 rows

Our export:
17,005,582 rows
```

For other tables, the counts may be slightly different.

Example:

```text
V_CHARACTERISTICS

DLA expected:
44,657,756 rows

Our export:
44,630,707 rows
```

Instead of automatically treating every difference as a failure, the pipeline now reports the difference as a warning so we can investigate it.

The important checks are:

* Did the table export successfully?
* Did the transformation complete?
* Is the row count reasonable compared with DLA's reported total?
* Did anything fail or produce obviously incomplete data?

---

## 11. No database UPSERT yet

We considered doing monthly SQL UPSERTs like:

```sql
INSERT ... ON CONFLICT DO UPDATE
```

but we are **not doing that yet**.

Reason: NIIN is used to connect tables, but that does not mean every table has exactly one row per NIIN.

For example:

```text
one NIIN
    ->
many V_CHARACTERISTICS rows
```

and potentially multiple part/manufacturer records.

We would need to understand the true unique key for every table before doing safe row-level upserts.

For the MVP, a full monthly snapshot is simpler and safer.

---

## 12. How monthly releases work

We build each monthly dataset separately.

Example:

```text
data/
    CURRENT.json

    releases/
        SEP-2026/
            manifest.json
            parquet/
                p_flis_nsn/
                v_characteristics/
                v_flis_part/
                ...
```

The new release is built in a staging folder first.

Only after:

```text
export succeeds
+ transformation succeeds
+ row counts validate
```

do we switch:

```text
CURRENT.json
```

to the new release.

That means if the update crashes halfway through, the previous good dataset stays active.

---

## Quick cheat sheet

```text
NSN
= Full government stock number

FSC
= What category/type of item is this?

NIIN
= Which exact government catalog item is this?

CAGE
= Which company/organization is associated with it?

P_FLIS_NSN
= Basic item identity

V_CHARACTERISTICS
= Physical/technical characteristics
  -> especially important for material detection

V_FLIS_PART
= Manufacturer part number + CAGE relationship

V_CAGE_ADDRESS
= Company/location behind a CAGE code

V_CAGE_STATUS_AND_TYPE
= Extra organization metadata

V_H2_FSC
= FSC category descriptions

V_FLIS_IDENTIFICATION
= Extra official item attributes
```

## The most important distinction

If you remember only one thing:

```text
V_CHARACTERISTICS
    = What is the item made of / what are its properties?

V_FLIS_PART
    = What manufacturer part/company is tied to the item?
```

Those two tables are doing different jobs, and both connect back to the item through the NIIN.

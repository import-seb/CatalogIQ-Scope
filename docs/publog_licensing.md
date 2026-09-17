# PUB LOG — Rules on Use, Storage, and Redistribution

**Author:** Maria Garcia
**Last updated:** September 16, 2026
**Status:** Complete. README inside the ZIP has been checked.

## Question

Can we use PUB LOG data, keep a copy of it, share it inside the team, and republish it? I went looking for a license, a terms of use page, or any written rule from DLA that limits what we do with the data.

## Short answer

There is no license on PUB LOG and no terms of use page on the download path. I also checked the README inside the ZIP. It contains no redistribution restriction. The data is made by a federal agency, so it is not covered by U.S. copyright. We can store it, transform it, share it, and republish it without asking DLA for permission.

The real limits are not copyright limits. They come from four other places, listed below.

## What PUB LOG is

PUB LOG is a monthly data product from the Defense Logistics Agency (DLA). It has National Stock Number (NSN) data, CAGE codes, part numbers, characteristics data, and cross reference data. It is free, with no subscription and no CAC. DLA says it is open to the U.S. government, Foreign Military Sales customers, other foreign national customers, and the private sector.

The reason it is public is that DLA already removed the sensitive parts before publishing. PUB LOG does not include service unique data, proprietary data, or NATO data. Stock numbers flagged proprietary (MRC PRPY) are also left out of the cross reference and management files. The filtering happened upstream, before we downloaded anything.

## Why there is no copyright problem

Work created by federal employees as part of their job is not protected by U.S. copyright. That comes from 17 U.S.C. § 105. The Government Publishing Office says the same thing in plainer words: public documents can generally be reprinted with no legal restriction.

DLA also publishes PUB LOG through its FOIA electronic reading room, which is where agencies post records already released to the public.

## What the README actually says

The README in the ZIP has one license-like sentence, and it is the opening line:

> These tools include unlimited use within IMD.

That is a grant, not a restriction. It refers to the utilities in the TOOLS folder. There is no redistribution clause, no copyright notice, no terms of use, and no restriction on the data anywhere in the file. The rest of the README is technical documentation for the extraction utilities.

Two things to note about that sentence. It covers the tools, not the data. And "within IMD" is not defined, so it is best read as permission to use the utilities against this product, not as permission to ship them somewhere else.

## The four things that actually limit us

### 1. Never mix in FED LOG data

Most important one. FED LOG is the same family of data but it is not public. DLA calls it restricted and limits it to authorized users. The sponsorship terms say unauthorized distribution of the product or its contents is prohibited and may bring civil and criminal penalties. A contractor can only use it for the one contract it was approved for, and cannot keep it after that contract ends.

If anyone pulls FED LOG from FedMall and the two datasets land in the same table, the whole pipeline inherits those rules. Keep them physically separate and document which source each field came from.

### 2. The data is free, the software is not the same thing

The PUB LOG download is not only data. It also ships the Integrated Mobile Database (IMD) application, in a classic build and a newer IMD2 build (`IMD.EXE`, `IMD64.EXE`, `IMD2.EXE`), plus the TOOLS utilities. IMD is a commercial product from Synergetics. DLA describes this family of software as Commercial Off-The-Shelf.

The README confirms the software layer is shared with FED LOG. Its Java section refers to the FED LOG DVD and to "the discs," and the Java package namespace is `com.vitalaxiom.IMD`, not a DLA namespace.

DLA states the rule plainly for FED LOG: the software on the discs is licensed to DLA for specific services, and copying it without the owner's permission may break U.S. or international copyright law. The README's grant of unlimited use does not extend to redistribution.

**Rule for us: share the data files, not the installer or the TOOLS binaries.**

### 3. Trademarks and endorsement

Public domain covers copyright, not trademarks. DLA filed trademark applications for PUB LOG, FED LOG, and WebFLIS in July 2021 and uses the registered symbol on its own pages. We can say our product uses PUB LOG data. We cannot name our product PUB LOG, use the DLA seal, or word anything so it sounds like DLA approved us.

### 4. Accuracy, not permission

DLA says the only official versions are the printed or hard copy materials, and the hard copy wins if the two disagree. If we publish anything built on this data, we should add our own accuracy note.

Plan for the data changing. The reading room posts a notice that starting April 1, 2026, nine fields would stop being distributed in both FED LOG and PUB LOG: `FIIG`, `FIIG_TITLE`, `DELT_RSN_CD`, `ACTY_CD`, `QNTV_EXPRSN`, `SUPP_RCVR`, `AUTH_RCVR`, `SUBMTR`, `NATO_FMSN`. DLA also replaced the old FLIS legacy files in April 2023 and warns the old and new files do not map one to one.

**Monitoring hook:** the README says changes to the software or the data tables are published in `RELNOTES.TXT` and `POPUP.PDF`. Diff those two files on every monthly refresh. That is our early warning for schema changes.

## How to get the data out

The README documents a supported extraction path, so we do not have to reverse-engineer the file format.

- **`DECOMP`** runs simple SQL against one table at a time. `decomp.exe <path> "SQL" <output_file>`. A wildcard on the primary column pulls a whole table. A query list can be passed as `@queries.txt`, one per line. `BLANK` and `NONBLANK` are reserved words for null and non-null filtering. `IMD_TABLES` is a built-in table listing every table with its row count and publication date, which is the right place to start.
- **`MERGE`** joins two extracts on a shared column. Both files must be sorted first.
- **`ORDERBY`** sorts an extract by a column, with optional `NoDups` and `Header` flags.
- **`ALTER`** drops, moves, combines, or splits columns in pipe-delimited DECOMP output.
- **`FILTER`** keeps or drops rows by field value.
- **Bindings** exist for C/C++ (`DECOMP.LIB`, `DECOMP.DLL`), Java, and Python.

**Caution on the bindings:** the sample code targets Visual Studio 6.0, 2008, and 2012, and the Python sample is written for Python 2.7. The bit width of the interpreter has to match the bit width of the DLL. Do not build the pipeline on that sample. Prefer calling `decomp.exe` as a subprocess and parsing the pipe-delimited output.

## What I could not confirm

- I confirmed DLA filed the trademark applications for PUB LOG, FED LOG, and WebFLIS. I did not verify they are registered today.
- The April 2026 field change is what the reading room notice says. I did not open a current data file to confirm those fields are gone.
- No DLA document addresses bulk redistribution head on. As far as I can tell nothing exists to find. But "no rule exists" is a weaker finding than "a rule allows it."

## Recommendation

Treat PUB LOG as open data and use it. Keep FED LOG out of the same system and document the separation. Share the data files, not the installer or the TOOLS binaries. Add our own accuracy note when we publish. Diff `RELNOTES.TXT` each month so a dropped field does not break us.

If this ever goes into a contract or a compliance memo, email dlacontactcenter@dla.mil and get the answer in writing. One email turns a reading into a record. I am not a lawyer, so if real money rides on this, counsel should look at the FED LOG separation plan.

## Sources

- PUB LOG `README.TXT`, from the monthly ZIP download (checked September 2026).
- Defense Logistics Agency. (2017). *FED LOG sponsorship information.* https://www.dla.mil/Portals/104/Documents/InformationOperations/LogisticsInformationServices/Customer%20Outreach/Subscriptions/J6_FEDLOGSponsorshipInformation_20170407.pdf
- Defense Logistics Agency. (2022). *PUB LOG — Public logistics data.* https://www.dla.mil/Information-Operations/Services/Applications/PUB-LOG/
- Defense Logistics Agency. (2025). *How to install PUB LOG system.* https://www.dla.mil/Small-Business/Resources/Training/Details/Article/4180130/how-to-install-pub-log-system/
- Defense Logistics Agency. (2026). *FLIS data electronic reading room.* https://www.dla.mil/Information-Operations/FLIS-Data-Electronic-Reading-Room/
- Synergetics. (n.d.). *Integrated Mobile Database (IMD).* https://synergetics.com/imd/
- U.S. Government Publishing Office. (n.d.). *Copyright and use policies of govinfo content.*
- Copyright Act of 1976, 17 U.S.C. § 105.

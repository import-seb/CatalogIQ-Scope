# CatalogIQ Dataset Profile

This report profiles the supplied training and target datasets before cleaning or modeling.

## Dataset Sizes

- Training: **84,552 rows × 32 columns**
- Target: **28,082 rows × 32 columns**

## Column Names

### Training
- `Exclude`
- `JoiningKey`
- `Retailer`
- `Category`
- `Mnfr`
- `Brand`
- `Platform`
- `Segment`
- `Sub-Segment`
- `TargetAgeGroup`
- `Sun1`
- `Sun2`
- `Sun3`
- `Sun4`
- `Sun5`
- `Notes`
- `ProductCategory`
- `ProductBrand`
- `ProductName`
- `ProductRating`
- `ProductReviewsCount`
- `XRatXRev`
- `ReviewsCount`
- `Sku`
- `Upc`
- `ProductModelNumber`
- `ProductUrl`
- `ProductImageUrl`
- `MDM_InsertDateTime`
- `MDM_Id`
- `ProductDescription`
- `ProductContents`

### Target
- `Exclude`
- `JoiningKey`
- `Retailer`
- `Category`
- `Mnfr`
- `Brand`
- `Platform`
- `Segment`
- `Sub-Segment`
- `TargetAgeGroup`
- `Sun1`
- `Sun2`
- `Sun3`
- `Sun4`
- `Sun5`
- `Notes`
- `ProductCategory`
- `ProductBrand`
- `ProductName`
- `ProductRating`
- `ProductReviewsCount`
- `XRatXRev`
- `ReviewsCount`
- `Sku`
- `Upc`
- `ProductModelNumber`
- `ProductUrl`
- `ProductImageUrl`
- `MDM_InsertDateTime`
- `MDM_Id`
- `ProductDescription`
- `ProductContents`

## Data Types

### Training
- `Exclude`: `str`
- `JoiningKey`: `str`
- `Retailer`: `str`
- `Category`: `str`
- `Mnfr`: `str`
- `Brand`: `str`
- `Platform`: `str`
- `Segment`: `str`
- `Sub-Segment`: `str`
- `TargetAgeGroup`: `str`
- `Sun1`: `str`
- `Sun2`: `str`
- `Sun3`: `str`
- `Sun4`: `str`
- `Sun5`: `str`
- `Notes`: `str`
- `ProductCategory`: `str`
- `ProductBrand`: `str`
- `ProductName`: `str`
- `ProductRating`: `str`
- `ProductReviewsCount`: `str`
- `XRatXRev`: `str`
- `ReviewsCount`: `str`
- `Sku`: `str`
- `Upc`: `str`
- `ProductModelNumber`: `str`
- `ProductUrl`: `str`
- `ProductImageUrl`: `str`
- `MDM_InsertDateTime`: `str`
- `MDM_Id`: `str`
- `ProductDescription`: `str`
- `ProductContents`: `str`

### Target
- `Exclude`: `str`
- `JoiningKey`: `str`
- `Retailer`: `str`
- `Category`: `float64`
- `Mnfr`: `float64`
- `Brand`: `float64`
- `Platform`: `float64`
- `Segment`: `float64`
- `Sub-Segment`: `float64`
- `TargetAgeGroup`: `float64`
- `Sun1`: `str`
- `Sun2`: `str`
- `Sun3`: `str`
- `Sun4`: `str`
- `Sun5`: `str`
- `Notes`: `str`
- `ProductCategory`: `str`
- `ProductBrand`: `str`
- `ProductName`: `str`
- `ProductRating`: `str`
- `ProductReviewsCount`: `str`
- `XRatXRev`: `str`
- `ReviewsCount`: `str`
- `Sku`: `str`
- `Upc`: `str`
- `ProductModelNumber`: `str`
- `ProductUrl`: `str`
- `ProductImageUrl`: `str`
- `MDM_InsertDateTime`: `str`
- `MDM_Id`: `str`
- `ProductDescription`: `str`
- `ProductContents`: `str`

## Missing Values

### Training
- `Notes`: 84,551 missing (100.00%)
- `Sun4`: 84,550 missing (100.00%)
- `Sun2`: 84,550 missing (100.00%)
- `Sun3`: 84,550 missing (100.00%)
- `Sun5`: 84,550 missing (100.00%)
- `Sun1`: 84,549 missing (100.00%)
- `Platform`: 83,447 missing (98.69%)
- `Exclude`: 79,255 missing (93.74%)
- `ProductModelNumber`: 72,565 missing (85.82%)
- `Upc`: 67,477 missing (79.81%)
- `ProductDescription`: 57,285 missing (67.75%)
- `ProductContents`: 50,844 missing (60.13%)
- `Sub-Segment`: 7,515 missing (8.89%)
- `TargetAgeGroup`: 7,077 missing (8.37%)
- `Mnfr`: 5,698 missing (6.74%)
- `Segment`: 4,789 missing (5.66%)
- `ProductImageUrl`: 754 missing (0.89%)
- `Brand`: 322 missing (0.38%)
- `ProductUrl`: 228 missing (0.27%)
- `ProductName`: 151 missing (0.18%)
- `ProductReviewsCount`: 139 missing (0.16%)
- `MDM_InsertDateTime`: 118 missing (0.14%)
- `ProductBrand`: 100 missing (0.12%)
- `MDM_Id`: 66 missing (0.08%)
- `ProductCategory`: 29 missing (0.03%)
- `Sku`: 6 missing (0.01%)
- `XRatXRev`: 5 missing (0.01%)
- `ReviewsCount`: 4 missing (0.00%)
- `ProductRating`: 4 missing (0.00%)
- `Category`: 1 missing (0.00%)
- `Retailer`: 0 missing (0.00%)
- `JoiningKey`: 0 missing (0.00%)

### Target
- `Category`: 28,082 missing (100.00%)
- `Platform`: 28,082 missing (100.00%)
- `Brand`: 28,082 missing (100.00%)
- `Mnfr`: 28,082 missing (100.00%)
- `Segment`: 28,082 missing (100.00%)
- `Sub-Segment`: 28,082 missing (100.00%)
- `TargetAgeGroup`: 28,082 missing (100.00%)
- `Sun2`: 28,081 missing (100.00%)
- `Sun5`: 28,081 missing (100.00%)
- `Sun4`: 28,081 missing (100.00%)
- `Notes`: 28,081 missing (100.00%)
- `Sun3`: 28,081 missing (100.00%)
- `Sun1`: 28,080 missing (99.99%)
- `Exclude`: 26,228 missing (93.40%)
- `ProductModelNumber`: 24,018 missing (85.53%)
- `Upc`: 22,336 missing (79.54%)
- `ProductDescription`: 19,070 missing (67.91%)
- `ProductContents`: 16,822 missing (59.90%)
- `ProductImageUrl`: 265 missing (0.94%)
- `ProductUrl`: 85 missing (0.30%)
- `ProductReviewsCount`: 49 missing (0.17%)
- `ProductName`: 43 missing (0.15%)
- `MDM_InsertDateTime`: 37 missing (0.13%)
- `MDM_Id`: 31 missing (0.11%)
- `ProductBrand`: 22 missing (0.08%)
- `ProductCategory`: 7 missing (0.02%)
- `Sku`: 5 missing (0.02%)
- `ReviewsCount`: 2 missing (0.01%)
- `ProductRating`: 2 missing (0.01%)
- `XRatXRev`: 2 missing (0.01%)
- `Retailer`: 0 missing (0.00%)
- `JoiningKey`: 0 missing (0.00%)

## Duplicate Rows

- Training duplicate rows: **0**
- Target duplicate rows: **0**

## Malformed / Shifted Row Checks

- Training rows with the wrong CSV field count found in the first scan: **0**
- Target rows with the wrong CSV field count found in the first scan: **0**
- Training rows with at least 50% missing fields: **90**
- Target rows with at least 50% missing fields: **65**

## Training Target Labels

### Mnfr

Unique labels: **4**

- ` aids in collagen formation* GRASS-FED COLLAGEN: Super Collagen + Vitamin C & Biotin is Keto certified`
- ` coffee`
- `All others`
- `J&J`

### Brand

Unique labels: **146**

- ` gluten-free`
- ` tea`
- `21st Century`
- `Advil`
- `Airborne`
- `Aleve`
- `Alka Seltzer`
- `Allegra`
- `Amazing Nutrition`
- `Amazon Basic Care`
- `Ancient Nutrition`
- `Andrew Lessman`
- `Ark Labs`
- `Astepro`
- `Bayer`
- `Benadryl`
- `Bengay`
- `Best Naturals`
- `Better Not Younger`
- `Biofreeze`
- `Biotrue`
- `Bluebonnet`
- `Boiron`
- `Bonafide`
- `Botanic Choice`
- `Bronson`
- `BulkSupplements`
- `CVS Health`
- `Carlson`
- `Carlyle`
- `Centrum`
- `Claritin`
- `Codeage`
- `Country Life`
- `Culturelle`
- `Designs for health`
- `Dimetapp`
- `Doctor's Best`
- `Dr. Mercola`
- `Emergen-C`
- `Equate`
- `Estroven`
- `Excedrin`
- `Flonase`
- `Force Factor`
- `GNC`
- `Gaia Herbs`
- `Garden of Life`
- `Generic`
- `Genexa`
- `HALLS`
- `HAWAIIPHARM`
- `HUM`
- `Health & Her`
- `Herb Pharm`
- `Hers`
- `Horbäach`
- `Hyland's`
- `IDEAL PERFORMANCE`
- `Icy Hot`
- `Imodium`
- `Irwin Naturals`
- `Jarrow Formulas`
- `KAL`
- `Kirkland Signature`
- `Lactaid`
- `Life Extension`
- `Love Wellness`
- `MaryRuth Organics`
- `Mason Natural`
- `MegaFood`
- `Meno`
- `Metagenics`
- `Metamucil`
- `Micro Ingredients`
- `MiraLax`
- `Mommy's Bliss`
- `Motrin`
- `Mucinex`
- `NOW`
- `Nasonex`
- `Natrol`
- `NaturaNectar`
- `Natural Factors`
- `Nature Made`
- `Nature's Answer`
- `Nature's Bounty`
- `Nature's Nutrition`
- `Nature's Plus`
- `Nature's Sunshine`
- `Nature's Truth`
- `Nature's Way`
- `Natures Craft`
- `NeilMed`
- `New Chapter`
- `Nexium`
- `Nordic Naturals`
- `NusaPure`
- `Nutricost`
- `Olly`
- `One A Day`
- `Other Brands`
- `Pedialyte`
- `Pepcid`
- `Pepto Bismol`
- `Pipping Rock`
- `Pure Encapsulations`
- `Pure Organic Ingredients`
- `Puritan's Pride`
- `Qunol`
- `Rae Wellness`
- `ReNew Life`
- `Refresh`
- `Ricola`
- `Robitussin`
- `Rolaids`
- `SOLARAY`
- `SmartyPants`
- `Solgar`
- `Source Naturals`
- `Spring Valley`
- `Standard Process Inc.`
- `Starwest Botanicals`
- `Sudafed`
- `Sunwarrior`
- `Swanson`
- `TUMS`
- `Theraflu`
- `Thorne Research`
- `Tylenol`
- `UnItemised brand`
- `Vicks`
- `Visine`
- `Vitafusion`
- `Vital Proteins`
- `Vitamatic`
- `Vitauthority`
- `Voltaren`
- `Walgreens`
- `Wile`
- `Womaness`
- `Xyzal`
- `Zarbee's`
- `Zicam`
- `Zyrtec`
- `up & up`

### Platform

Unique labels: **63**

- ` grass fed`
- ` orange juice or a smoothie daily; unflavored GRASS-FED: Super Collagen Peptides powder is grass-fed`
- `12 Hour Relief`
- `24 Hour Relief`
- `Allergy`
- `Allergy Plus Congestion`
- `Arthritis`
- `Arthritis Pain Relief`
- `Chewables`
- `Children`
- `Coated Tabs`
- `Cold + Cough + Runny Nose`
- `Cold + Flu`
- `Cold Max`
- `Complete`
- `Concentrated Drops`
- `Cough + Cold + Sore Throat`
- `Cream`
- `Dissolve Packs`
- `Dissolve Tablets`
- `Dry Eye`
- `Dual Action`
- `Effective in 15 Minutes`
- `Eight Hour Arthritis Pain`
- `Eight Hour Muscle Aches`
- `Extra Strength`
- `Gel`
- `Head`
- `Head + Cold`
- `Head Care`
- `Immune`
- `Imodium A-D`
- `Liquid Gels`
- `Maximum Strength`
- `Migraine Caplets`
- `Motrin IB`
- `Motrin PM`
- `Multi-Symptom`
- `Nasal`
- `Oral Solution`
- `Oral Suspension`
- `Original Strength`
- `PM`
- `Pain + Fever`
- `Pain Relief`
- `Patch`
- `Precise`
- `Rapid Release Gels`
- `Red Eye`
- `Regular Strength`
- `Simply Sleep`
- `Sinus`
- `Sinus + Headache`
- `Sinus Plus`
- `Sinus Severe`
- `Sleep`
- `SmartCheck`
- `Syrup`
- `Tension`
- `Throat`
- `Tylenol PM`
- `Ultratabs`
- `Vits & Supps`

### Segment

Unique labels: **10**

- ` IGEN Non-GMO tested.* No soy wheat`
- ` Keto certified`
- `Allergy`
- `CCFS`
- `Digestive Health`
- `External Analgesics`
- `Internal Analgesics`
- `Lifestyle CHC`
- `Other Self Care`
- `Vitamins, Minerals & Supplements`

### Sub-Segment

Unique labels: **53**

- ` gluten-free`
- ` lactose`
- `Acid Relief`
- `All Other Digestive Health`
- `Allergy INS`
- `Anti-Diarrheal`
- `Anti-Gas`
- `Anti-Nausea`
- `Cold & Heat Wraps`
- `Cold / Flu`
- `Cold/Flu`
- `Constipation Remedies`
- `Cough`
- `Creams & Gels (Rubs)`
- `Creams/Gels & Medicated Patches`
- `Daily oral contracception`
- `Dermatologicals`
- `Devices/Refills`
- `Digestive enzymes`
- `Ear Care`
- `External Upper Respiratory`
- `Eye Care`
- `Hemorrhoid Remedies`
- `Herbal & Natural (H&N)`
- `Internal Pain`
- `Lactose Intolerance`
- `Menopausal Support`
- `Minerals`
- `Motion Sickness`
- `Multivitamins`
- `Nasal Sprays`
- `Obesity treatments`
- `Oral Antihistamine`
- `Other Allergy`
- `Other External Analgesics`
- `Other Lifestyle`
- `Other Lifestyle CHC`
- `Other Self Care`
- `Pregnancy Vitamins`
- `Probiotics`
- `Prostate Remedies`
- `Remaining Internal Pain`
- `Single Vitamins`
- `Sinus excl Sprays`
- `Sleep Aids`
- `Sore Throat / Cough Drops`
- `Speciality Pain`
- `Stimulants`
- `Supplements`
- `Systemic Cardiovasculars`
- `Temperature Packs`
- `Topical Itch`
- `Urinary products`

### TargetAgeGroup

Unique labels: **6**

- ` Paleo friendly`
- ` starch or artificial flavors; 3 tablets daily NeoCell Super Collagen + Vitamin C and Biotin is made with grass-fed hydrolyzed collagen`
- ` starch or artificial flavors; 3 tablets daily"`
- `Adult`
- `Children`
- `Infant`

## Training Target Class Counts

### Mnfr

- `All others`: 77821
- `<MISSING>`: 5698
- `J&J`: 1030
- ` aids in collagen formation* GRASS-FED COLLAGEN: Super Collagen + Vitamin C & Biotin is Keto certified`: 2
- ` coffee`: 1

### Brand

- `UnItemised brand`: 54059
- `Other Brands`: 3485
- `NOW`: 1253
- `Nature Made`: 894
- `Swanson`: 820
- `Nature's Bounty`: 792
- `Walgreens`: 721
- `CVS Health`: 607
- `Generic`: 557
- `Nature's Way`: 538
- `Equate`: 518
- `SOLARAY`: 480
- `Vicks`: 458
- `Nutricost`: 457
- `Carlyle`: 420
- `BulkSupplements`: 412
- `Solgar`: 404
- `Garden of Life`: 390
- `Puritan's Pride`: 373
- `Life Extension`: 351
- `Tylenol`: 324
- `<MISSING>`: 322
- `Source Naturals`: 320
- `Natrol`: 320
- `Olly`: 303
- `Best Naturals`: 292
- `GNC`: 287
- `Mucinex`: 278
- `Pure Encapsulations`: 274
- `Nature's Plus`: 264
- `Centrum`: 263
- `Spring Valley`: 263
- `Horbäach`: 262
- `Nature's Truth`: 261
- `Advil`: 258
- `Nordic Naturals`: 252
- `Pipping Rock`: 249
- `up & up`: 242
- `Carlson`: 239
- `Bluebonnet`: 228
- `Amazing Nutrition`: 221
- `Jarrow Formulas`: 219
- `One A Day`: 209
- `Doctor's Best`: 206
- `MaryRuth Organics`: 203
- `New Chapter`: 202
- `Natures Craft`: 201
- `TUMS`: 200
- `Kirkland Signature`: 198
- `Vitafusion`: 196
- `HALLS`: 195
- `Pure Organic Ingredients`: 184
- `Designs for health`: 182
- `Hyland's`: 182
- `Nature's Nutrition`: 178
- `Natural Factors`: 174
- `Codeage`: 170
- `Herb Pharm`: 166
- `Gaia Herbs`: 161
- `21st Century`: 160
- `Boiron`: 160
- `Country Life`: 159
- `ReNew Life`: 155
- `IDEAL PERFORMANCE`: 155
- `MegaFood`: 154
- `Vitamatic`: 154
- `Starwest Botanicals`: 150
- `Airborne`: 148
- `Nature's Answer`: 146
- `Ricola`: 146
- `Emergen-C`: 146
- `Aleve`: 146
- `Mason Natural`: 143
- `Alka Seltzer`: 143
- `Amazon Basic Care`: 138
- `Zarbee's`: 138
- `Metamucil`: 137
- `Thorne Research`: 136
- `NeilMed`: 135
- `KAL`: 134
- `Benadryl`: 129
- `Bronson`: 129
- `Vital Proteins`: 129
- `Standard Process Inc.`: 128
- `Icy Hot`: 125
- `Claritin`: 122
- `Refresh`: 122
- `Andrew Lessman`: 121
- `HAWAIIPHARM`: 121
- `Force Factor`: 117
- `NusaPure`: 116
- `Dr. Mercola`: 115
- `Ark Labs`: 112
- `SmartyPants`: 110
- `Culturelle`: 109
- `Motrin`: 108
- `Bayer`: 107
- `Zyrtec`: 106
- `Micro Ingredients`: 104
- `Irwin Naturals`: 99
- `Robitussin`: 98
- `Nature's Sunshine`: 98
- `Metagenics`: 97
- `Ancient Nutrition`: 97
- `Biofreeze`: 93
- `Pepcid`: 92
- `Botanic Choice`: 90
- `Zicam`: 88
- `Pepto Bismol`: 82
- `Allegra`: 78
- `Excedrin`: 75
- `Mommy's Bliss`: 75
- `Qunol`: 73
- `Flonase`: 72
- `Lactaid`: 60
- `MiraLax`: 60
- `Visine`: 55
- `Nexium`: 52
- `Theraflu`: 50
- `Sudafed`: 48
- `Imodium`: 45
- `Genexa`: 42
- `Pedialyte`: 37
- `HUM`: 36
- `Bengay`: 34
- `Rae Wellness`: 32
- `Xyzal`: 31
- `Sunwarrior`: 31
- `Biotrue`: 27
- `Vitauthority`: 24
- `Love Wellness`: 22
- `Dimetapp`: 21
- `Rolaids`: 20
- `Astepro`: 18
- `Nasonex`: 11
- `NaturaNectar`: 8
- `Estroven`: 7
- `Voltaren`: 7
- `Wile`: 6
- `Bonafide`: 2
- ` gluten-free`: 2
- `Better Not Younger`: 2
- `Hers`: 1
- ` tea`: 1
- `Health & Her`: 1
- `Meno`: 1
- `Womaness`: 1

### Platform

- `<MISSING>`: 83447
- `Extra Strength`: 84
- `Children`: 72
- `Pain + Fever`: 51
- `Oral Suspension`: 49
- `Motrin IB`: 47
- `24 Hour Relief`: 46
- `Eight Hour Arthritis Pain`: 43
- `Throat`: 40
- `Maximum Strength`: 39
- `Complete`: 38
- `Liquid Gels`: 37
- `Sleep`: 35
- `Imodium A-D`: 32
- `Immune`: 31
- `Sinus`: 30
- `Red Eye`: 28
- `Chewables`: 27
- `Vits & Supps`: 22
- `Cold + Flu`: 21
- `Ultratabs`: 20
- `Cream`: 19
- `Allergy`: 19
- `Migraine Caplets`: 19
- `Regular Strength`: 18
- `Sinus Severe`: 15
- `Allergy Plus Congestion`: 14
- `Dual Action`: 13
- `Original Strength`: 13
- `Pain Relief`: 13
- `Dry Eye`: 12
- `Tylenol PM`: 12
- `Rapid Release Gels`: 10
- `Gel`: 10
- `Dissolve Packs`: 9
- `Multi-Symptom`: 8
- `Nasal`: 8
- `Motrin PM`: 8
- `Concentrated Drops`: 7
- `Head`: 7
- `Coated Tabs`: 7
- `Dissolve Tablets`: 7
- `Cold + Cough + Runny Nose`: 6
- `PM`: 6
- `Oral Solution`: 5
- `Simply Sleep`: 5
- `Head + Cold`: 5
- `Cough + Cold + Sore Throat`: 4
- `Arthritis Pain Relief`: 4
- `Tension`: 4
- `Patch`: 4
- `Sinus + Headache`: 4
- `Eight Hour Muscle Aches`: 3
- `Arthritis`: 2
- ` grass fed`: 2
- `Syrup`: 2
- `Head Care`: 2
- `Sinus Plus`: 1
- `Precise`: 1
- `12 Hour Relief`: 1
- ` orange juice or a smoothie daily; unflavored GRASS-FED: Super Collagen Peptides powder is grass-fed`: 1
- `Cold Max`: 1
- `SmartCheck`: 1
- `Effective in 15 Minutes`: 1

### Segment

- `Vitamins, Minerals & Supplements`: 49091
- `Digestive Health`: 7102
- `External Analgesics`: 6069
- `Internal Analgesics`: 4808
- `<MISSING>`: 4789
- `Lifestyle CHC`: 4551
- `CCFS`: 4267
- `Allergy`: 2737
- `Other Self Care`: 1135
- ` IGEN Non-GMO tested.* No soy wheat`: 2
- ` Keto certified`: 1

### Sub-Segment

- `Supplements`: 19535
- `Herbal & Natural (H&N)`: 10565
- `Minerals`: 7556
- `<MISSING>`: 7515
- `Single Vitamins`: 4936
- `Cold & Heat Wraps`: 3818
- `Multivitamins`: 3426
- `Ear Care`: 3095
- `All Other Digestive Health`: 2141
- `Internal Pain`: 1977
- `Speciality Pain`: 1755
- `Constipation Remedies`: 1733
- `Sore Throat / Cough Drops`: 1319
- `Cold/Flu`: 1302
- `Acid Relief`: 1154
- `Sleep Aids`: 1065
- `Eye Care`: 1029
- `Creams/Gels & Medicated Patches`: 1009
- `Other Allergy`: 882
- `Oral Antihistamine`: 823
- `Other External Analgesics`: 783
- `Probiotics`: 656
- `Cough`: 594
- `Allergy INS`: 581
- `Systemic Cardiovasculars`: 521
- `Nasal Sprays`: 464
- `Temperature Packs`: 436
- `Sinus excl Sprays`: 430
- `Motion Sickness`: 408
- `Digestive enzymes`: 393
- `Obesity treatments`: 393
- `Anti-Gas`: 373
- `Pregnancy Vitamins`: 359
- `Anti-Diarrheal`: 237
- `Prostate Remedies`: 233
- `Hemorrhoid Remedies`: 210
- `Other Lifestyle CHC`: 202
- `External Upper Respiratory`: 154
- `Menopausal Support`: 113
- `Other Self Care`: 102
- `Urinary products`: 92
- `Anti-Nausea`: 57
- `Lactose Intolerance`: 42
- `Topical Itch`: 32
- `Devices/Refills`: 14
- `Remaining Internal Pain`: 11
- `Stimulants`: 10
- `Cold / Flu`: 4
- `Dermatologicals`: 4
- `Other Lifestyle`: 3
- ` lactose`: 2
- `Daily oral contracception`: 2
- `Creams & Gels (Rubs)`: 1
- ` gluten-free`: 1

### TargetAgeGroup

- `Adult`: 72995
- `<MISSING>`: 7077
- `Children`: 3718
- `Infant`: 759
- ` Paleo friendly`: 1
- ` starch or artificial flavors; 3 tablets daily"`: 1
- ` starch or artificial flavors; 3 tablets daily NeoCell Super Collagen + Vitamin C and Biotin is made with grass-fed hydrolyzed collagen`: 1

## Obvious Data-Quality Problems

### Training
- Columns with leading/trailing whitespace: [('Exclude', 3), ('JoiningKey', 4), ('Retailer', 4), ('Category', 3), ('Mnfr', 3), ('Brand', 3), ('Platform', 3), ('Segment', 3), ('Sub-Segment', 3), ('TargetAgeGroup', 3), ('Sun1', 2), ('Sun2', 2), ('Sun3', 2), ('Sun4', 2), ('Sun5', 1), ('Notes', 1), ('ProductCategory', 1), ('ProductBrand', 1), ('ProductName', 9), ('ProductRating', 430), ('ProductReviewsCount', 276), ('XRatXRev', 192), ('ReviewsCount', 121), ('Sku', 68), ('Upc', 54), ('ProductModelNumber', 23), ('ProductUrl', 8), ('ProductImageUrl', 5), ('MDM_InsertDateTime', 1), ('ProductDescription', 13), ('ProductContents', 1000)]

### Target
- Columns with leading/trailing whitespace: [('Exclude', 2), ('JoiningKey', 2), ('Retailer', 2), ('Sun1', 1), ('Sun2', 1), ('Sun3', 1), ('Sun4', 1), ('Sun5', 1), ('Notes', 1), ('ProductCategory', 1), ('ProductBrand', 1), ('ProductName', 3), ('ProductRating', 158), ('ProductReviewsCount', 103), ('XRatXRev', 73), ('ReviewsCount', 54), ('Sku', 37), ('Upc', 24), ('ProductModelNumber', 12), ('ProductUrl', 4), ('ProductImageUrl', 4), ('ProductDescription', 5), ('ProductContents', 345)]

### Missing Training Target Labels

- `Mnfr`: 5,698
- `Brand`: 322
- `Platform`: 83,447
- `Segment`: 4,789
- `Sub-Segment`: 7,515
- `TargetAgeGroup`: 7,077

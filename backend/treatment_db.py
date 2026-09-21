"""
Treatment / management recommendations for each class the model predicts.

Content is condensed from IRRI Rice Knowledge Bank fact sheets, UC IPM, CABI Compendium,
and other agricultural-extension sources (see project report for full citations). This is
general guidance, not a substitute for a local agricultural extension officer -- the app
says so explicitly in the UI.

Khmer translations are provided for a Cambodian farmer-facing audience; they should be
reviewed by a native-speaking agronomist before any real-world deployment.
"""

TREATMENT_DB = {
    "bacterial_leaf_blight": {
        "display_name_en": "Bacterial Leaf Blight",
        "display_name_km": "ជំងឺស្លឹកឆេះលឿងដោយបាក់តេរី (Bacterial Leaf Blight)",
        "category": "bacterial disease",
        "severity": "high",
        "advice_en": [
            "Use certified, disease-free seed and resistant/tolerant varieties where available.",
            "Avoid excess nitrogen fertilizer; split nitrogen applications instead of one large dose.",
            "Drain and dry the field periodically -- avoid continuous deep flooding, which favors spread.",
            "Remove and destroy infected stubble and weed hosts after harvest.",
            "In severe outbreaks, copper-based bactericides can reduce spread, but resistant varieties and field sanitation are more reliable long-term.",
        ],
        "advice_km": [
            "ប្រើគ្រាប់ពូជស្អាត គ្មានជំងឺ និងប្រភេទពូជធន់ទ្រាំបើមាន។",
            "កុំប្រើជីអាសូតច្រើនពេក គួរបែងចែកការដាក់ជីជាច្រើនដង។",
            "បង្ហូរទឹក និងធ្វើឲ្យស្រែស្ងួតជាបន្តបន្ទាប់ កុំបំពេញទឹកជ្រៅជាប់រហូត។",
            "ដកចេញ និងបំផ្លាញកម្ទេច ស្មៅ និងដំណាំចាស់ដែលមានជំងឺបន្ទាប់ពីច្រូតកាត់។",
            "ក្នុងករណីធ្ងន់ធ្ងរ អាចប្រើថ្នាំសំលាប់បាក់តេរីមានផ្សំទង់ដែង ប៉ុន្តែពូជធន់ និងអនាម័យស្រែជាដំណោះស្រាយប្រសើរជាងរយៈពេលវែង។",
        ],
    },
    "bacterial_leaf_streak": {
        "display_name_en": "Bacterial Leaf Streak",
        "display_name_km": "ជំងឺឆួងស្លឹកដោយបាក់តេរី (Bacterial Leaf Streak)",
        "category": "bacterial disease",
        "severity": "medium",
        "advice_en": [
            "Plant tolerant varieties; avoid known-susceptible cultivars in outbreak areas.",
            "Treat seed (e.g. hot-water treatment around 52°C for ~30 min) to reduce seed-borne bacteria.",
            "Manage nitrogen carefully and avoid prolonged leaf wetness / dense canopies.",
            "Early-stage foliar bactericide/copper sprays can slow spread but are not a full cure.",
            "Rotate with non-rice crops and clear infected stubble to reduce carry-over.",
        ],
        "advice_km": [
            "ដាំពូជធន់ទ្រាំ និងជៀសវាងពូជដែលងាយប៉ះពាល់ក្នុងតំបន់មានជំងឺរាតត្បាត។",
            "ព្យាបាលគ្រាប់ពូជ (ឧ. ជ្រលក់ទឹកក្តៅប្រមាណ ៥២°C រយៈពេល ៣០នាទី) ដើម្បីកាត់បន្ថយបាក់តេរីជាប់នឹងគ្រាប់។",
            "គ្រប់គ្រងជីអាសូតឲ្យសមស្រប និងជៀសវាងស្លឹកសើមយូរ ឬដាំដុះក្រាស់ពេក។",
            "ការបាញ់ថ្នាំមានផ្សំទង់ដែងនៅដំណាក់កាលដំបូងអាចជួយពន្យាការរាលដាល ប៉ុន្តែមិនមែនជាការព្យាបាលទាំងស្រុងទេ។",
            "ធ្វើបំរើដំណាំផ្សេងឡើងវិញ និងសម្អាតស្មៅចាស់ដែលមានជំងឺ។",
        ],
    },
    "bacterial_panicle_blight": {
        "display_name_en": "Bacterial Panicle Blight",
        "display_name_km": "ជំងឺឆេះកួរស្រូវដោយបាក់តេរី (Bacterial Panicle Blight)",
        "category": "bacterial disease",
        "severity": "high",
        "advice_en": [
            "Use certified seed and, where available, cultivars with partial resistance.",
            "Avoid excessive nitrogen and very high seeding rates, which increase severity.",
            "Favor early-maturing varieties to reduce exposure to high-temperature flowering conditions that favor this disease.",
            "Copper-based bactericides or oxolinic acid can help, though resistance has developed in some regions -- rotate modes of action.",
            "Prioritize field sanitation and avoid planting into infected residue.",
        ],
        "advice_km": [
            "ប្រើគ្រាប់ពូជស្អាត និងពូជដែលមានភាពធន់ទ្រាំមួយផ្នែកបើមាន។",
            "ជៀសវាងការប្រើជីអាសូតច្រើនពេក និងសម្ពោធគ្រាប់ក្រាស់ពេក ព្រោះវាបង្កើនកម្រិតធ្ងន់ធ្ងរនៃជំងឺ។",
            "ប្រើពូជទុំលឿន ដើម្បីកាត់បន្ថយការប៉ះពាល់នឹងសីតុណ្ហភាពខ្ពស់ពេលចេញផ្កា។",
            "ថ្នាំសំលាប់បាក់តេរីមានផ្សំទង់ដែង ឬ oxolinic acid អាចជួយបាន ប៉ុន្តែគួរឆ្លាស់ប្រភេទថ្នាំព្រោះមានភាពធន់នៅតំបន់ខ្លះ។",
            "ផ្តោតលើអនាម័យក្នុងស្រែ និងជៀសវាងដាំលើសំណល់ដំណាំចាស់ដែលមានជំងឺ។",
        ],
    },
    "blast": {
        "display_name_en": "Rice Blast",
        "display_name_km": "ជំងឺផ្កាឆេះ / ប្លាស (Blast)",
        "category": "fungal disease",
        "severity": "high",
        "advice_en": [
            "Use resistant varieties where available; rotate varieties across seasons to slow pathogen adaptation.",
            "Avoid excess nitrogen fertilizer, which strongly increases susceptibility.",
            "Maintain continuous, even flooding where feasible -- alternating wet/dry stress favors the disease.",
            "Scout fields closely near booting stage; a fungicide (e.g. a strobilurin such as azoxystrobin) applied at boot-to-early-heading can prevent the more destructive neck blast if disease pressure is high.",
            "Use certified seed to avoid introducing the fungus on seed.",
        ],
        "advice_km": [
            "ប្រើពូជធន់ទ្រាំបើមាន និងផ្លាស់ប្តូរពូជតាមរដូវដើម្បីពន្យារពេលការសម្របខ្លួនរបស់ផ្សិត។",
            "ជៀសវាងជីអាសូតច្រើនពេក ព្រោះវាបង្កើនភាពងាយប៉ះពាល់យ៉ាងខ្លាំង។",
            "រក្សាទឹកជាប់ស្មើៗគ្នាតាមដែលអាចធ្វើបាន ព្យោះការប្តូរសើម-ស្ងួតញឹកញាប់អាចបង្កើនជំងឺ។",
            "ត្រួតពិនិត្យស្រែយ៉ាងដិតដល់នៅដំណាក់កាលចេញកួរ ថ្នាំកូវីនអាចការពារជំងឺឆេះកតួដែលធ្ងន់ធ្ងរជាង។",
            "ប្រើគ្រាប់ពូជស្អាតដើម្បីជៀសវាងការនាំចូលផ្សិតតាមរយៈគ្រាប់ពូជ។",
        ],
    },
    "brown_spot": {
        "display_name_en": "Brown Spot",
        "display_name_km": "ជំងឺចំណុចត្នោត (Brown Spot)",
        "category": "fungal disease",
        "severity": "medium",
        "advice_en": [
            "Brown spot is strongly linked to poor soil fertility, especially potassium and silicon deficiency -- balanced fertilization is often the most effective control.",
            "Use certified, fungicide-treated seed; the fungus is commonly seed-borne.",
            "Avoid water stress and nutrient-poor, marginal soils where possible.",
            "Fungicide sprays (e.g. propiconazole or similar) can help when disease pressure is high, particularly near heading.",
            "Remove and destroy infected residue after harvest.",
        ],
        "advice_km": [
            "ជំងឺនេះទាក់ទងយ៉ាងខ្លាំងជាមួយដីខ្សត់ជីជាតិ ជាពិសេសកង្វះប៉ូតាស្យូម និងស៊ីលីស៊ូម ការដាក់ជីឲ្យសមតុល្យជាវិធីព្យាបាលមានប្រសិទ្ធភាពបំផុត។",
            "ប្រើគ្រាប់ពូជស្អាតដែលបានព្យាបាលដោយថ្នាំកូវីន ព្រោះផ្សិតនេះច្រើនតែជាប់មកជាមួយគ្រាប់ពូជ។",
            "ជៀសវាងការខ្វះទឹក និងដីខ្សត់ជីជាតិតាមដែលអាចធ្វើទៅបាន។",
            "ការបាញ់ថ្នាំកូវីនអាចជួយបានក្នុងករណីមានការឆ្លងខ្លាំង ជាពិសេសនៅជិតដំណាក់កាលចេញកួរ។",
            "ដកចេញ និងបំផ្លាញសំណល់ដំណាំដែលមានជំងឺបន្ទាប់ពីច្រូតកាត់។",
        ],
    },
    "dead_heart": {
        "display_name_en": "Dead Heart (Stem Borer Damage)",
        "display_name_km": "ដើមស្រូវងាប់ (ដងវែក/ដង្កូវប៉ោង Stem Borer)",
        "category": "pest damage",
        "severity": "medium",
        "advice_en": [
            "This is insect (stem borer) damage, not a disease -- the central shoot dies because larvae bore into and feed inside the stem.",
            "Field-flood after harvest and manage stubble to kill overwintering larvae; avoid leaving infested stubble in the field.",
            "Release egg parasitoids (e.g. Trichogramma spp.) early in the season as a biological control option.",
            "Use pheromone traps to monitor adult moth activity and time interventions.",
            "If thresholds are exceeded, targeted insecticides (e.g. chlorantraniliprole, fipronil) applied at early larval stages are most effective -- broad, calendar-based spraying is discouraged as it harms natural enemies.",
        ],
        "advice_km": [
            "នេះជាការខូចខាតដោយសត្វល្អិត (ដងវែក/ដង្កូវប៉ោង) មិនមែនជំងឺទេ ដើមកណ្តាលងាប់ព្រោះដង្កូវចូលទៅក្នុងដើម ស្រូបចិញ្ចឹមខ្លួន។",
            "បំពេញទឹកស្រែបន្ទាប់ពីច្រូតកាត់ និងគ្រប់គ្រងសំណល់ដើមស្រូវដើម្បីសម្លាប់ដង្កូវដែលស្នាក់នៅ កុំទុកសំណល់ដែលមានដង្កូវនៅក្នុងស្រែ។",
            "លែងសត្វសត្រូវធម្មជាតិ (ដូចជា Trichogramma) នៅដើមរដូវជាវិធីគ្រប់គ្រងជីវសាស្ត្រ។",
            "ប្រើអន្ទាក់ភេរម៉ូនដើម្បីតាមដានសកម្មភាពមេអំបៅ និងកំណត់ពេលធ្វើអន្តរាគមន៍។",
            "ប្រសិនបើលើសកម្រិតព្រមាន គួរប្រើថ្នាំសម្លាប់សត្វល្អិតដែលកំណត់គោលដៅច្បាស់លាស់ ជៀសវាងការបាញ់ថ្នាំទូទៅតាមកាលវិភាគ ព្រោះវាបំផ្លាញសត្រូវធម្មជាតិ។",
        ],
    },
    "downy_mildew": {
        "display_name_en": "Downy Mildew",
        "display_name_km": "ជំងឺរបុយស (Downy Mildew)",
        "category": "fungal-like (oomycete) disease",
        "severity": "medium",
        "advice_en": [
            "Use resistant varieties where available and prioritize good nursery hygiene -- seedlings are most vulnerable.",
            "Improve drainage; avoid waterlogged soil and prolonged leaf wetness.",
            "Space plants to reduce canopy density and humidity, and avoid overhead irrigation.",
            "Remove and destroy crop residue after harvest; rotate with non-host crops.",
            "Copper-based fungicides can be used preventively in high-pressure situations.",
        ],
        "advice_km": [
            "ប្រើពូជធន់ទ្រាំបើមាន និងផ្តោតលើអនាម័យក្នុងកន្លែងដាំគ្រាប់ ព្រោះដើមកូនងាយប៉ះពាល់បំផុត។",
            "កែលម្អការបង្ហូរទឹក ជៀសវាងដីជោគជាំទឹក និងស្លឹកសើមយូរ។",
            "ដាំដុះកុំឲ្យក្រាស់ពេកដើម្បីកាត់បន្ថយសំណើម និងជៀសវាងការស្រោចទឹកលើស្លឹក។",
            "ដកចេញ និងបំផ្លាញសំណល់ដំណាំបន្ទាប់ពីច្រូតកាត់ ធ្វើបំរើដំណាំផ្សេងឡើងវិញ។",
            "ថ្នាំកូវីនមានផ្សំទង់ដែងអាចប្រើបង្ការនៅពេលមានហានិភ័យខ្ពស់។",
        ],
    },
    "hispa": {
        "display_name_en": "Rice Hispa",
        "display_name_km": "សត្វជីងគ្រួចស្រូវ (Hispa)",
        "category": "pest damage",
        "severity": "medium",
        "advice_en": [
            "This is insect (beetle) damage -- adults scrape leaf surfaces leaving white parallel streaks; larvae tunnel inside leaves.",
            "Plant early in the season to avoid peak hispa activity, and avoid excess nitrogen.",
            "Clip and destroy leaf tips before transplanting to remove eggs; remove ratoons and volunteer rice in the off-season.",
            "Hand-collect adult beetles where feasible, and conserve natural predators.",
            "If infestation is heavy, targeted insecticides (e.g. thiamethoxam, lambda-cyhalothrin, fipronil-based products) can be used according to local recommendations.",
        ],
        "advice_km": [
            "នេះជាការខូចខាតដោយសត្វល្អិត (ជីងគ្រួច) មេវែកកោសផ្ទៃស្លឹកទុកស្នាមខ្សែស ដង្កូវក្រៀមចូលក្នុងស្លឹក។",
            "ដាំដើមរដូវដើម្បីជៀសវាងកំពូលសកម្មភាពរបស់សត្វនេះ និងជៀសវាងជីអាសូតច្រើនពេក។",
            "កាត់ និងបំផ្លាញចុងស្លឹកមុនផ្សាំដើម្បីដកពងចេញ ដកចេញចំការស្រូវសំណល់ក្រៅរដូវ។",
            "ចាប់សម្លាប់មេវែកដោយដៃបើអាចធ្វើបាន និងថែរក្សាសត្រូវធម្មជាតិ។",
            "ប្រសិនបើឆ្លងខ្លាំង គួរប្រើថ្នាំសម្លាប់សត្វល្អិតដែលកំណត់គោលដៅតាមអនុសាសន៍មូលដ្ឋាន។",
        ],
    },
    "normal": {
        "display_name_en": "Healthy",
        "display_name_km": "ស្លឹកមានសុខភាពល្អ (Healthy)",
        "category": "healthy",
        "severity": "none",
        "advice_en": [
            "No disease or pest damage detected -- this leaf looks healthy.",
            "Continue routine field monitoring, especially around key growth stages (tillering, booting, heading).",
            "Maintain balanced fertilization and good water management to keep plants resilient.",
        ],
        "advice_km": [
            "មិនមានជំងឺ ឬការខូចខាតដោយសត្វល្អិតត្រូវបានរកឃើញទេ ស្លឹកមើលទៅមានសុខភាពល្អ។",
            "បន្តត្រួតពិនិត្យស្រែជាប្រចាំ ជាពិសេសនៅដំណាក់កាលសំខាន់ៗ (ដុះគល់ ចេញកួរ ចេញផ្កា)។",
            "រក្សាការដាក់ជីឲ្យសមតុល្យ និងការគ្រប់គ្រងទឹកល្អ ដើម្បីរក្សាភាពរឹងមាំរបស់ដំណាំ។",
        ],
    },
    "tungro": {
        "display_name_en": "Tungro Virus",
        "display_name_km": "ជំងឺទុងរ៉ូ (Tungro)",
        "category": "viral disease (leafhopper-transmitted)",
        "severity": "high",
        "advice_en": [
            "Tungro is spread by green leafhoppers -- managing the insect vector is central to control, not just the plant.",
            "Plant tungro-resistant varieties where available.",
            "Synchronize planting dates across neighboring farms and avoid staggered planting, which keeps leafhopper populations active year-round.",
            "Limit rice cropping to at most two seasons per year where feasible, with a non-rice or fallow break to interrupt the vector's life cycle.",
            "Control weeds and volunteer rice that host leafhoppers; if vector pressure is high, use recommended insecticides (e.g. buprofezin, pymetrozine) rather than broad-spectrum products that leafhoppers have developed resistance to.",
        ],
        "advice_km": [
            "ជំងឺទុងរ៉ូរាលដាលដោយសត្វខ្យង​ត្រចៀកបៃតង ការគ្រប់គ្រងសត្វល្អិតជាចំណុចសំខាន់ មិនមែនត្រឹមតែរុក្ខជាតិទេ។",
            "ដាំពូជធន់ទ្រាំនឹងទុងរ៉ូបើមាន។",
            "កំណត់ពេលដាំឲ្យស្របគ្នារវាងចំការជិតខាង ជៀសវាងការដាំមិនស្របគ្នា ព្រោះវារក្សាចំនួនសត្វខ្យងឲ្យសកម្មពេញឆ្នាំ។",
            "កំណត់ដំណាំស្រូវឲ្យបានតែពីររដូវក្នុងមួយឆ្នាំតាមដែលអាចធ្វើបាន ដោយឲ្យមានដំណាំផ្សេង ឬស្រែទំនេរដើម្បីកាត់ផ្តាច់វដ្តជីវិតរបស់សត្វខ្យង។",
            "កម្ចាត់ស្មៅ និងស្រូវសំណល់ដែលជាកន្លែងជ្រកកោនសត្វខ្យង ប្រសិនបើមានសម្ពាធខ្ពស់ ប្រើថ្នាំសម្លាប់សត្វល្អិតដែលបានណែនាំ។",
        ],
    },
}


def get_treatment(label: str):
    return TREATMENT_DB.get(label)

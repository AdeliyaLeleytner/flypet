"""Stimulus and behaviour catalogs.

Stimuli are groups of sensory neurons we can drive with Poisson input; readouts are motor /
descending neurons whose firing we interpret as behaviour. Every entry is defined through the
Schlegel et al. 2024 annotations, except a few groups taken verbatim from Shiu et al. 2024
(the sugar / bitter / water GRNs and the antennal-grooming neurons they validated in vivo).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import random
from . import connectome as C

# --- neuron groups validated in Shiu et al. 2024 (FlyWire v630 IDs that survive in v783) ----
SHIU_SUGAR = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]
SHIU_BITTER = [
    720575940621778381,
    720575940602353632,
    720575940617094208,
    720575940619197093,
    720575940626287336,
    720575940618600651,
    720575940627692048,
    720575940630195909,
    720575940646212996,
    720575940610483162,
    720575940645743412,
    720575940627578156,
    720575940622298631,
    720575940621008895,
    720575940629146711,
    720575940610259370,
    720575940610481370,
    720575940619028208,
    720575940614281266,
    720575940613061118,
    720575940604027168,
]
SHIU_WATER = [
    720575940612950568,
    720575940631898285,
    720575940606002609,
    720575940612579053,
    720575940622902535,
    720575940616177458,
    720575940660292225,
    720575940622486922,
    720575940613786774,
    720575940629852866,
    720575940625861168,
    720575940613996959,
    720575940617857694,
    720575940644965399,
    720575940625203504,
    720575940630553415,
    720575940635172191,
    720575940634796536,
]
SHIU_ADN1 = (
    720575940616185531  # "DN1" in Shiu et al. Fig. 5 (antennal grooming DN), DNg62 in annotations
)
SHIU_ADN2 = 720575940629806974  # "DN2" in Shiu et al. Fig. 5, DNge078 in annotations
SHIU_ABN1 = 720575940630907434  # aBN1 (antennal grooming brain neuron), SAD093 in annotations
MN9_RIGHT = 720575940660219265  # MN9 used in Shiu et al.; cell_type CB0701


@dataclass
class Stimulus:
    key: str
    label_ru: str
    description: str  # what it models, for the LLM
    default_rate_hz: float
    expected: str  # behaviour reported in the literature for this input
    selector: dict = field(default_factory=dict)
    ids: list[int] = field(default_factory=list)
    max_neurons: int | None = None  # subsample huge groups (photoreceptors) to keep the run fast

    def resolve(self, side: str | None = None, seed: int = 0) -> list[int]:
        flyid2i, _ = C.id_maps()
        if self.ids:
            ids = [i for i in self.ids if i in flyid2i]
            if side:
                ann = C.annotations()
                ids = [i for i in ids if ann.loc[i, "side"] == side]
        else:
            ids = C.select(side=side, **self.selector)
        if self.max_neurons and len(ids) > self.max_neurons:
            rng = random.Random(seed)
            ids = sorted(rng.sample(ids, self.max_neurons))
        return ids


STIMULI: dict[str, Stimulus] = {
    s.key: s
    for s in [
        # ---- taste (labellum = mouthparts, tarsi = feet) ----
        Stimulus(
            "sugar",
            "сахар на хоботке",
            "labellar sugar-sensing gustatory receptor neurons (Gr64f class), the set validated in Shiu et al. 2024",
            150,
            "proboscis extension (MN9 firing) - feeding initiation",
            ids=SHIU_SUGAR,
        ),
        Stimulus(
            "water",
            "вода на хоботке",
            "labellar water-sensing GRNs (ppk28), set from Shiu et al. 2024",
            150,
            "proboscis extension, weaker than sugar",
            ids=SHIU_WATER,
        ),
        Stimulus(
            "bitter",
            "горечь на хоботке",
            "labellar bitter-sensing GRNs (Gr66a class), set from Shiu et al. 2024",
            150,
            "suppression of feeding; when given together with sugar it reduces MN9 firing",
            ids=SHIU_BITTER,
        ),
        Stimulus(
            "sugar_all_labellar",
            "сахар/вода на всём хоботке",
            "all labellar sugar/water GRNs, both sides (annotation cell_sub_class sugar/water, LB types)",
            150,
            "strong proboscis extension",
            selector=dict(
                cell_class="gustatory", cell_sub_class="sugar/water", cell_type=r"LB.*", regex=True
            ),
        ),
        Stimulus(
            "sugar_tarsal",
            "сахар на лапках",
            "tarsal (leg) sugar/water GRNs that project to the brain (claw_tpGRN, dorsal_tpGRN)",
            150,
            "proboscis extension after tarsal contact with sugar",
            selector=dict(cell_class="gustatory", cell_type=["claw_tpGRN", "dorsal_tpGRN"]),
        ),
        Stimulus(
            "salt_low",
            "слабая соль",
            "low-salt gustatory neurons (IR94e class, appetitive at low concentration)",
            150,
            "mild appetitive response",
            selector=dict(cell_class="gustatory", cell_sub_class="low-salt"),
        ),
        Stimulus(
            "pharyngeal",
            "вкус в глотке",
            "pharyngeal GRNs (PhG types), taste of food already being ingested",
            150,
            "modulation of ingestion",
            selector=dict(cell_class="gustatory", cell_type=r"PhG.*", regex=True),
        ),
        # ---- mechanosensation ----
        Stimulus(
            "antenna_touch",
            "касание антенн / ветер",
            "Johnston's organ C/E neurons (wind and gravity sensing, annotation wind_gravity); Shiu et al. drove JO-CE/F to get antennal grooming",
            200,
            "antennal grooming via aDN1/aDN2 and aBN1",
            selector=dict(cell_class="mechanosensory", cell_sub_class="wind_gravity"),
        ),
        Stimulus(
            "sound",
            "звук",
            "Johnston's organ A/B auditory neurons (annotation auditory)",
            200,
            "auditory processing; courtship-song circuits",
            selector=dict(cell_class="mechanosensory", cell_sub_class="auditory"),
        ),
        Stimulus(
            "head_touch",
            "касание головы (щетинки)",
            "head bristle mechanosensory neurons",
            150,
            "head grooming",
            selector=dict(cell_class="mechanosensory", cell_sub_class="head bristle"),
            max_neurons=120,
        ),
        Stimulus(
            "eye_touch",
            "касание глаза (щетинки между омматидиями)",
            "interommatidial bristle mechanosensory neurons (BM_InOm)",
            150,
            "eye grooming",
            selector=dict(cell_class="mechanosensory", cell_sub_class="eye bristle"),
            max_neurons=150,
        ),
        Stimulus(
            "grooming_bristles",
            "щетинки, запускающие груминг",
            "mechanosensory neurons annotated as grooming",
            150,
            "grooming",
            selector=dict(cell_class="mechanosensory", cell_sub_class="grooming"),
        ),
        # ---- olfaction (ORN types by glomerulus) ----
        Stimulus(
            "smell_vinegar",
            "запах уксуса / фруктов",
            "ORNs of glomeruli DM1 (Or42b), DM2 (Or22a), VA2 (Or92a) and DM4 (Or59b): attractive food odours such as vinegar, ethyl acetate, fruit esters",
            60,
            "attraction, approach, feeding readiness",
            selector=dict(
                cell_class="olfactory", cell_type=["ORN_DM1", "ORN_DM2", "ORN_VA2", "ORN_DM4"]
            ),
        ),
        Stimulus(
            "smell_geosmin",
            "запах плесени (геосмин)",
            "ORN_DA2 (Or56a) geosmin-sensing neurons",
            60,
            "innate aversion, avoidance",
            selector=dict(cell_class="olfactory", cell_type="ORN_DA2"),
        ),
        Stimulus(
            "smell_co2",
            "углекислый газ",
            "ORN_V (Gr21a/Gr63a) CO2-sensing neurons",
            60,
            "innate avoidance",
            selector=dict(cell_class="olfactory", cell_type="ORN_V"),
        ),
        Stimulus(
            "smell_cva",
            "феромон cVA (другая муха)",
            "ORN_DA1 (Or67d) cVA pheromone neurons",
            60,
            "social response: courtship suppression in males, receptivity in females, aggression",
            selector=dict(cell_class="olfactory", cell_type="ORN_DA1"),
        ),
        Stimulus(
            "smell_fly_pheromone",
            "феромоны других мух (Or47b)",
            "ORN_VA1v (Or47b) fly-odour neurons",
            60,
            "attraction to other flies",
            selector=dict(cell_class="olfactory", cell_type="ORN_VA1v"),
        ),
        Stimulus(
            "smell_acid",
            "кислый запах",
            "ORN_DC4 / DP1l acid-sensing neurons (Ir64a)",
            60,
            "avoidance of acids",
            selector=dict(cell_class="olfactory", cell_type=["ORN_DC4", "ORN_DP1l"]),
        ),
        # ---- temperature / humidity ----
        Stimulus(
            "heat",
            "жар",
            "TRN_VP2 heating-sensing thermoreceptor neurons",
            150,
            "avoidance of heat",
            selector=dict(cell_class="thermosensory", cell_sub_class="heating"),
        ),
        Stimulus(
            "cold",
            "холод",
            "TRN_VP3 cold-sensing thermoreceptor neurons",
            150,
            "avoidance of cold",
            selector=dict(cell_class="thermosensory", cell_sub_class="cold"),
        ),
        Stimulus(
            "dry_air",
            "сухой воздух",
            "HRN_VP4 dry-air hygroreceptor neurons",
            150,
            "seeking humidity",
            selector=dict(cell_class="hygrosensory", cell_sub_class="dry"),
        ),
        Stimulus(
            "moist_air",
            "влажный воздух",
            "HRN_VP5 moist-air hygroreceptor neurons",
            150,
            "humidity preference behaviour",
            selector=dict(cell_class="hygrosensory", cell_sub_class="moist"),
        ),
        # ---- vision ----
        Stimulus(
            "light",
            "свет",
            "outer photoreceptors R1-6 (random subsample of 400 of 8452)",
            100,
            "phototaxis, optomotor circuits",
            selector=dict(cell_class="visual", cell_type="R1-6"),
            max_neurons=400,
        ),
        Stimulus(
            "uv_light",
            "ультрафиолет",
            "R7 photoreceptors (UV sensitive), subsample",
            100,
            "UV attraction",
            selector=dict(cell_class="visual", cell_type="R7"),
            max_neurons=300,
        ),
        Stimulus(
            "ocelli_light",
            "яркий свет сверху (глазки)",
            "ocellar photoreceptors",
            100,
            "flight stabilisation, light-on response",
            selector=dict(cell_class="visual", cell_sub_class="ocellar"),
        ),
        Stimulus(
            "looming",
            "надвигающийся объект",
            "LC4 and LPLC2 visual projection neurons (looming detectors)",
            200,
            "escape: giant fibre (DNp01) firing, takeoff",
            selector=dict(super_class="visual_projection", cell_type=["LC4", "LPLC2"]),
        ),
        Stimulus(
            "moving_object",
            "движущийся маленький объект",
            "LC11 / LC10 small-object detectors",
            150,
            "orientation / tracking of small objects",
            selector=dict(
                super_class="visual_projection",
                cell_type=["LC11", "LC10a", "LC10b", "LC10c", "LC10d"],
            ),
        ),
    ]
}


@dataclass
class Readout:
    key: str
    label_ru: str
    description: str
    selector: dict = field(default_factory=dict)
    ids: list[int] = field(default_factory=list)

    def resolve(self) -> list[int]:
        flyid2i, _ = C.id_maps()
        if self.ids:
            return [i for i in self.ids if i in flyid2i]
        return C.select(**self.selector)


READOUTS: dict[str, Readout] = {
    r.key: r
    for r in [
        Readout(
            "proboscis_extension",
            "вытягивание хоботка (кормление)",
            "MN9 proboscis motor neurons (cell_type CB0701); their firing = feeding initiation, validated in Shiu et al. 2024",
            selector=dict(cell_type="CB0701"),
        ),
        Readout(
            "antennal_grooming",
            "чистка антенн",
            "aDN1/aDN2 antennal grooming descending neurons and aBN1, the pathway Shiu et al. 2024 reproduced from JON input",
            ids=[SHIU_ADN1, SHIU_ADN2, SHIU_ABN1],
        ),
        Readout(
            "escape_takeoff",
            "побег / взлёт",
            "giant fibre DNp01 and looming descending neurons DNp02, DNp04, DNp06, DNp11",
            selector=dict(
                super_class="descending", cell_type=["DNp01", "DNp02", "DNp04", "DNp06", "DNp11"]
            ),
        ),
        Readout(
            "walk_forward",
            "ходьба вперёд",
            "DNp09 (P9) forward-walking descending neurons",
            selector=dict(cell_type="DNp09"),
        ),
        Readout(
            "walk_backward",
            "ходьба назад",
            "MDN moonwalker descending neurons",
            selector=dict(cell_type="MDN"),
        ),
        Readout(
            "turn",
            "поворот",
            "DNa01 / DNa02 steering descending neurons (side tells the turn direction)",
            selector=dict(cell_type=["DNa01", "DNa02"]),
        ),
        Readout(
            "head_neck_movement",
            "движение головы",
            "neck motor neurons (CB0705 class)",
            selector=dict(cell_type="CB0705"),
        ),
        Readout(
            "other_motor",
            "прочие мотонейроны мозга",
            "all other brain motor neurons (proboscis, antennal, pharyngeal muscles)",
            selector=dict(super_class="motor"),
        ),
        Readout(
            "descending_all",
            "все нисходящие нейроны",
            "all 1303 descending neurons (commands sent to the ventral nerve cord)",
            selector=dict(super_class="descending"),
        ),
    ]
}

COL_STARTDATE = "survival_hovk_begindatum"
COL_ENDDATE = "survival_hovk_einddatum"
COL_ID_HOVK = "bk_huurovereenkomst"
COL_ID_EENHEID = "bk_eenheid"
COL_LABEL_DURATION = "huurduur_dagen"
COL_LABEL_EVENT = "hovk_beeindigd_indicator"
COL_HOVK_STATUS = "huurovereenkomst_statusnaam"

NUM_COLUMNS = [
    # "startjaar_huurovereenkomst",
    "aantal_kamers",
    "etagenummer",
    "lift_aanwezig_indicator",
    "gebruiksoppervlak",
    "percentage_man",
    "aantal_contractant_medebewoner",
    # 'aanvangshuurbedrag' > nu te veel missing values
    "leeftijd_woning",
    "min_leeftijd",
    "max_leeftijd",
    COL_LABEL_DURATION,
]

CAT_COLUMNS = [
    "daebnaam",
    "debiteur_type",
    "vestigingsnaam",
    "opleverjaarcategorie",
    "woningtype",
    # "eenheiddetailsoortnaam", > waarschijnlijk te gedetailleerd & imbalanced, we gebruiken woningtype
    # 'cbs_wijknaam', > overkill nu we gemeentenaam hebben, wellicht later toevoegen
    # 'cbs_buurtnaam', > overkill nu we gemeentenaam hebben, wellicht later toevoegen
    # 'gemeentenaam', > waarschijnlijk te gedetailleerd & imbalanced, we gebruiken vestigingsnaam (=regio)
    # 'huurklasse_code_aanvang', > nu te veel missing values
]

DATE_COLUMNS = [
    "survival_hovk_begindatum",
    "survival_hovk_einddatum",
    "opleverdatum",
    "min_geboortedatum",
    "max_geboortedatum",
]

FEATURE_COLUMNS = NUM_COLUMNS + CAT_COLUMNS

SELECT
        hovk.[d_huurovereenkomst]
    ,   hovk.[bk_huurovereenkomst]
    ,   hovk.[survival_hovk_begindatum]
    ,   YEAR(hovk.[survival_hovk_begindatum]) AS startjaar_huurovereenkomst
    ,   hovk.[survival_hovk_einddatum]
    ,   hovk.[huurovereenkomst_statusnaam]
    ,   hovk.[debiteur_type]
    ,   eenh.[bk_eenheid]
    ,   eenh.[eenheidnaam]
    ,   eenh.[eenheiddetailsoortnaam]
    ,   eenh.[aantal_kamers]
    ,   eenh.[woningtype]
    ,   eenh.[opleverdatum]
    ,   eenh.[opleverjaarcategorie]
    ,   eenh.[gemeentenaam]
    ,   eenh.[cbs_wijknaam]
    ,   eenh.[cbs_buurtnaam]
    ,   eenh.[etagenummer]
    ,   eenh.[daebnaam]
    ,   eenh.[vestigingsnaam]
    ,   eenh.[lift_aanwezig_indicator]
    ,   eenh.[gebruiksoppervlak]
    ,   contrpers.[min_geboortedatum]
    ,   contrpers.[max_geboortedatum]
    ,   contrpers.[percentage_man]
    ,   contrpers.[aantal_contractant_medebewoner]
    ,   aanvhuur.[aanvangshuurbedrag]
    ,   aanvhuur.[huurklasse_code_aanvang]
FROM
    (
        -- BEDOELING: 
        -- Alle relevante en betrouwbare informatie over huurovereenkomsten met informatie
        -- die beschikbaar was bij de startdatum hovk. Omdat we geen informatie willen lekken
        -- die voor de predictset niet beschikbaar is, nemen we alleen informatie mee die 
        -- 1) Vaststond bij start huurovereenkomst;
        -- 2) Of nodig is om een target+censoring te berekenen (d_dag!huurovereenkomst_einddatum).
        SELECT
                fh.[d_huurovereenkomst]
            ,   fh.[d_eenheid]
            ,   dh.[bk_huurovereenkomst]
            ,   CASE 
                    WHEN fh.[d_dag!huurovereenkomst_begindatum] = '2999-12-31' THEN CONVERT(datetime, '2199-12-31')
                    ELSE CONVERT(datetime, fh.[d_dag!huurovereenkomst_begindatum])
                END AS survival_hovk_begindatum
            ,   CASE
                    -- actief blijkend uit statusnaam & hovk-einddatum
                    WHEN    (dh.huurovereenkomst_statusnaam = 'Actief'
                            AND
                            fh.[d_dag!huurovereenkomst_einddatum] > GETDATE()
                            )
                    THEN GETDATE()
                    -- beëindigd/historisch blijkend uit statusnaam & hovk-einddatum max 31 dagen van vandaag
                    WHEN    (dh.huurovereenkomst_statusnaam IN ('Beëindigd', 'Historisch')
                            AND
                            fh.[d_dag!huurovereenkomst_einddatum] < DATEADD(day, 31, GETDATE())
                            ) 
                    THEN fh.[d_dag!huurovereenkomst_einddatum] 
                    -- opgezegd maar einddatum niet in nabije toekomst beschouwen we als actief
                    WHEN    (dh.huurovereenkomst_statusnaam = 'Opgezegd' 
                            AND 
                            fh.[d_dag!huurovereenkomst_einddatum] > DATEADD(month, 3, GETDATE())
                            ) 
                    THEN GETDATE()       
                    -- opgezegd met einddatum in nabije toekomst beschouwen we als beeindigd
                    WHEN    (dh.huurovereenkomst_statusnaam = 'Opgezegd' 
                            AND 
                            fh.[d_dag!huurovereenkomst_einddatum] <= DATEADD(month, 3, GETDATE())
                            ) 
                    THEN fh.[d_dag!huurovereenkomst_einddatum]
                    -- beëindigd met einddatum in nabije toekomst (is vreemd! maar) beschouwen we als beeindigd
                    WHEN    (dh.huurovereenkomst_statusnaam = 'Beëindigd' 
                            AND 
                            fh.[d_dag!huurovereenkomst_einddatum] <= DATEADD(month, 3, GETDATE())
                            ) 
                    THEN fh.[d_dag!huurovereenkomst_einddatum]                  
                    ELSE NULL
                END AS survival_hovk_einddatum
            ,   dh.huurovereenkomst_statusnaam
            ,   CASE WHEN dd.[debiteur_type] IN ('(Onbekend)', '(Leeg)') THEN NULL ELSE dd.[debiteur_type] END AS debiteur_type
        FROM dm_gold.f_huurovereenkomst fh

        INNER JOIN dm_gold.d_huurovereenkomst dh
        ON dh.id = fh.d_huurovereenkomst

        INNER JOIN dm_gold.d_eenheid de 
        on de.id = fh.[d_eenheid]

        INNER JOIN dm_gold.d_debiteur dd
        ON dd.id = fh.d_debiteur

        WHERE 
            -- alleen laatste snapshot
            fh.d_dag = (SELECT MAX(d_dag) FROM dm_gold.f_huurovereenkomst)
            -- alleen woningen (geen parkeerplekken, BOG, MOG, etc.)
            AND ((de.eenheidsoortnaam = 'Woningen') OR (de.eenheidsoortnaam='Woonruimte' AND de.woonvormnaam='Woningen'))
            -- geen voorlopige huurovereenkomsten
            AND dh.huurovereenkomst_statusnaam != 'Voorlopig'
            -- geen hovk die op actief staat maar een einddatum in het verleden heeft
            AND NOT (dh.huurovereenkomst_statusnaam = 'Actief' AND fh.[d_dag!huurovereenkomst_einddatum] < GETDATE())
            -- {VOORLOPIG} geen hovk met te oude einddatum om betrouwbaar te kunnen zijn
            AND NOT (fh.[d_dag!huurovereenkomst_einddatum] < '2002-01-01')
    ) AS hovk

LEFT JOIN
    (
        -- BEDOELING: 
        -- Alle relevante en betrouwbare informatie over eenheden bij een 
        -- huurovereenkomsten die beschikbaar was bij de startdatum hovk. 
        -- Dit is technisch niet mogelijk door oude snapshots te gebruiken (te weinig historie),
        -- daarom kiezen we om alleen statische variabelen mee te nemen die na start hovk niet meer wijzigen
        -- uit de meest recente snapshot.
        -- Omdat we geen informatie willen lekken die voor de predictset niet beschikbaar is, 
        -- nemen we alleen informatie mee die vaststond bij start huurovereenkomst.
        SELECT
                fe.d_eenheid
            ,   de.bk_eenheid
            ,   de.eenheidnaam
            ,   de.eenheiddetailsoortnaam
            ,   de.aantal_kamers
            ,   de.woningtype
            ,   CASE 
                    WHEN de.opleverjaar BETWEEN 1000 AND 2500 
                    THEN CONCAT(CAST(de.[opleverjaar] AS CHAR(4)), '-01-01') 
                    ELSE NULL
                END AS opleverdatum
            ,   CASE
                    WHEN de.opleverjaar = 0 THEN NULL 
                    WHEN de.opleverjaar < 1900 THEN '<1900'
                    WHEN de.opleverjaar BETWEEN 1900 AND 1919 THEN '1900-1919'
                    WHEN de.opleverjaar BETWEEN 1920 AND 1939 THEN '1920-1939'
                    WHEN de.opleverjaar BETWEEN 1940 AND 1959 THEN '1940-1959'
                    WHEN de.opleverjaar BETWEEN 1960 AND 1969 THEN '1960-1969'
                    WHEN de.opleverjaar BETWEEN 1970 AND 1979 THEN '1970-1979'
                    WHEN de.opleverjaar BETWEEN 1980 AND 1989 THEN '1980-1989'
                    WHEN de.opleverjaar BETWEEN 1990 AND 1999 THEN '1990-1999'
                    WHEN de.opleverjaar BETWEEN 2000 AND 2009 THEN '2000-2009'
                    WHEN de.opleverjaar >= 2010 THEN '>=2010'
                    ELSE NULL
                END AS opleverjaarcategorie
            ,   de.gemeentenaam
            ,   de.cbs_wijknaam
            ,   de.cbs_buurtnaam
            ,   de.etagenummer
            ,   CASE 
                    WHEN dd.daebnaam IN ('DAEB', 'Daeb') THEN 'Daeb'
                    WHEN dd.daebnaam IN ('NIETDAEB', 'Niet Daeb') THEN 'Niet Daeb'
                    ELSE NULL
                END AS daebnaam
            ,   CASE 
                    WHEN dv.vestigingsnaam IN ('(Onbekend)', '(Leeg)') THEN NULL 
                    ELSE dv.vestigingsnaam 
                END AS vestigingsnaam
            ,   CASE 
                    WHEN de.lift_aanwezig_indicator = 'Ja' THEN 1 
                    ELSE 0 
                END AS lift_aanwezig_indicator
            ,   CASE 
                    WHEN CAST(ROUND(de.gebruiksoppervlak, 0) AS INT) = 0 THEN NULL
                    ELSE CAST(ROUND(de.gebruiksoppervlak, 0) AS INT) 
                END AS gebruiksoppervlak

        FROM [dm_gold].[f_eenheid] fe

        INNER JOIN [dm_gold].[d_eenheid] de 
        on de.id = fe.[d_eenheid]

        INNER JOIN [dm_gold].[d_daeb] dd
        ON dd.id = fe.d_daeb

        INNER JOIN [dm_gold].[d_vestiging] dv 
        on dv.id = fe.d_vestiging

        WHERE
            -- alleen laatste snapshot
        fe.d_dag = (SELECT MAX(d_dag) FROM [dm_gold].[f_eenheid]) 
            -- alleen woningen (geen parkeerplekken, BOG, MOG, etc.)
            AND ((de.eenheidsoortnaam = 'Woningen') OR (de.eenheidsoortnaam='Woonruimte' AND de.woonvormnaam='Woningen'))

    ) AS eenh ON hovk.[d_eenheid] = eenh.[d_eenheid]

LEFT JOIN
    (
        -- BEDOELING: 
        -- Alle relevante en betrouwbare informatie over contractpersonen bij een 
        -- huurovereenkomsten die beschikbaar was bij de startdatum hovk. 
        -- Omdat we geen informatie willen lekken die voor de predictset niet beschikbaar 
        -- is, nemen we alleen informatie mee die vaststond bij start huurovereenkomst.

        SELECT 
                c.[d_huurovereenkomst]
            ,   MIN(c.[geboortedatum]) AS min_geboortedatum
            ,   MAX(c.[geboortedatum]) AS max_geboortedatum
            ,   AVG(c.[man_ind]) AS percentage_man
            ,   COUNT(*) AS aantal_contractant_medebewoner

        FROM 
        (
            SELECT
                    fch.d_huurovereenkomst
                ,   fh.[d_dag!huurovereenkomst_begindatum]
                ,   CASE
                        WHEN (
                            -- geen waarden die eigenlijk NULL moeten zijn
                            dr.geboortedatum NOT IN ('2999-12-31', '1900-01-01') 
                            -- geen toekomstige geboortedatum
                            AND dr.geboortedatum < GETDATE()
                            -- hovk_begindatum moet later zijn dan geboortedatum
                            AND fh.[d_dag!huurovereenkomst_begindatum] > dr.[geboortedatum]
                            )
                        THEN dr.[geboortedatum]
                    ELSE NULL END AS geboortedatum
                ,   CASE WHEN dr.geslachtsnaam IN ('Man','Mannelijk') THEN 1.0 ELSE 0.0 END AS man_ind

            FROM dm_gold.[f_contactpersoon_huurovereenkomst] fch

            INNER JOIN dm_gold.[f_huurovereenkomst] fh
            ON fch.[d_huurovereenkomst] = fh.[d_huurovereenkomst]
            AND fch.[d_dag] = fh.[d_dag]

            INNER JOIN dm_gold.d_relatie dr
            ON dr.id = fch.d_relatie

            INNER JOIN dm_gold.d_relatierol_huurovereenkomst drh
            ON drh.id = fch.d_relatierol_huurovereenkomst

            WHERE
                -- filtert lege of onbekende relatie(rol) weg
                (fch.d_relatie > 0 AND fch.d_relatierol_huurovereenkomst > 0)
                -- alleen contractant, medebewoner
                AND drh.relatierolnaam IN ('Contractant', 'Medebewoner')
                -- contractpersoon moet bekend zijn bij start huurovereenkomst
                AND fch.[d_dag!begindatum] = fh.[d_dag!huurovereenkomst_begindatum]
        ) AS c

        GROUP BY
                c.d_huurovereenkomst

    ) AS contrpers ON hovk.[d_huurovereenkomst] = contrpers.[d_huurovereenkomst]

LEFT JOIN
    (
        -- BEDOELING: Alle huurprijsinformatie die beschikbaar was bij de startdatum hovk.
        -- Omdat we geen informatie willen lekken die voor de predictset niet beschikbaar is,
        -- nemen we niet de laatste huurprijs mee.
        SELECT 
                fh.[d_huurovereenkomst]
            ,   fh.[aanvangshuurbedrag]
            ,   dha.[huurklasse_code] AS huurklasse_code_aanvang

        FROM dm_gold.f_huurovereenkomst fh

        INNER JOIN dm_gold.d_huurovereenkomst dh
        ON dh.id = fh.d_huurovereenkomst

        INNER JOIN dm_gold.d_huurklasse dha
        ON dha.id = fh.[d_huurklasse!aanvangshuur]

        WHERE
            fh.d_dag = (SELECT MAX(d_dag) FROM dm_gold.f_huurovereenkomst)

    ) AS aanvhuur ON hovk.[d_huurovereenkomst] = aanvhuur.[d_huurovereenkomst]
WHERE
    hovk.[survival_hovk_einddatum] >= '2023-01-01'
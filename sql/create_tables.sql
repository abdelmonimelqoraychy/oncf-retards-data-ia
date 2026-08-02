DROP TABLE IF EXISTS circulations;

CREATE TABLE circulations (
    id SERIAL PRIMARY KEY,
    date_trajet DATE,
    train_id VARCHAR(50),
    type_train VARCHAR(50),
    ligne VARCHAR(150),
    gare_depart VARCHAR(100),
    gare_arrivee VARCHAR(100),
    heure_depart_prevue TIME,
    heure_depart_reelle TIME,
    heure_arrivee_prevue TIME,
    heure_arrivee_reelle TIME,
    cause_retard VARCHAR(150),
    depart_prevu_dt TIMESTAMP,
    depart_reel_dt TIMESTAMP,
    arrivee_prevue_dt TIMESTAMP,
    arrivee_reelle_dt TIMESTAMP,
    retard_depart_minutes INTEGER,
    retard_arrivee_minutes INTEGER,
    train_en_retard INTEGER,
    statut VARCHAR(50),
    jour_semaine_num INTEGER,
    jour_semaine VARCHAR(50),
    heure_depart INTEGER
);
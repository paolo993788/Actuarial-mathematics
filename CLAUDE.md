# Actuarial mathematics

Repository di script e notebook di matematica attuariale.

## Organizzazione

- Leggi il README principale e la documentazione della cartella su cui lavori.
- Inserisci ogni progetto in `scripts/<nome_progetto>/` con un README dedicato; usa nomi descrittivi in snake_case.
- Conserva notebook in `notebooks/`, verifiche in `tests/` e piccoli dati sintetici o pubblici redistribuibili in `data/examples/`.
- Scrivi i risultati generati in `outputs/` e crea la cartella se manca.
- Usa percorsi relativi alla radice del repository; evita percorsi personali assoluti.
- Scrivi la documentazione in italiano e mantieni coerenti i nomi nel codice esistente.

## Codice e verifiche

- Rispetta il linguaggio del progetto. Non introdurre un framework o dipendenze senza una necessità concreta.
- Al primo script, documenta la versione del linguaggio e crea il file delle dipendenze appropriato se servono pacchetti esterni.
- Documenta scopo, input, output e comando esatto di esecuzione usando `docs/modello-script.md`.
- Rendi espliciti formule, ipotesi, unità di misura, convenzioni sui tassi e fonti utilizzate.
- Per calcoli nuovi o modificati, verifica almeno un risultato noto e i casi limite pertinenti. Usa tolleranze numeriche motivate.
- Fissa e documenta il seme degli esempi stocastici quando serve riproducibilità.
- Esegui le verifiche pertinenti disponibili e riporta i comandi e gli esiti reali. Non dichiarare test superati se non sono stati eseguiti.
- Al momento il repository contiene la struttura iniziale: non esistono ancora un comando di test generale o un workflow CI.
- Aggiorna il catalogo del README quando aggiungi uno script.

## Pubblicazione

- Segui `docs/pubblicazione.md` per branch, commit e pull request.
- Controlla `git status` e il diff prima di creare un commit; seleziona i file pertinenti alla modifica.
- Non inserire chiavi API, token, password, dati personali, dati riservati o cronologie delle sessioni.
- Usa variabili d'ambiente per le credenziali; gli eventuali file `.env.example` devono contenere solo valori fittizi.
- Mantieni la licenza MIT esistente e cita le fonti del codice riutilizzato.

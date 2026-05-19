# RiskSentinel - Hackathon Master Plan

Ultimo aggiornamento: 6 marzo 2026
Timeline hackathon: dal 24 febbraio 2026 al 15 marzo 2026.

## 1) Obiettivo

Massimizzare il punteggio finale con una submission solida, demo chiara in 2 minuti, e repo production-ready.

## 2) Cosa sappiamo sui requisiti submission (fonti ufficiali)

Link ufficiali:
- Regole: https://aka.ms/AIDevDaysHackathon/Rules
- Pagina evento: https://developer.microsoft.com/en-us/reactor/events/26647/
- Dettagli hackathon: https://aka.ms/aidevdayshackathon/details

Elementi richiesti (da verificare in checklist finale):
- Progetto funzionante e testabile
- Repository GitHub pubblico
- Video demo pubblico (max 2 minuti)
- Descrizione progetto completa
- Architettura e uso stack Microsoft/Azure chiaramente spiegati

Nota: i criteri sono strutturati su dimensioni tecniche + impatto + presentazione (impostazione bilanciata).

## 3) Stato attuale del progetto

Implementato:
- Streamlit app con simulazione contagio (build/shock/compare)
- Agentic Ops:
  - Scenario Commander
  - Autonomous Stress Test
  - Portfolio Co-Pilot
  - Auto-portfolio da network
  - Full Agentic Demo (one-click)
- Explainability:
  - policy/executor trace
  - badge Planner/Executor/Critic
- Reporting/export:
  - report, executive brief, trace bundle, action pack JSON-safe
- Refactor moduli:
  - `src/agentic_ops.py`
  - `src/reporting.py`
  - `src/ui_panels.py`
- Test unitari aggiunti:
  - `tests/test_agentic_ops.py`
  - `tests/test_reporting.py`

Bug recenti risolti:
- accesso `st.session_state` nei thread worker (Scenario Commander / Portfolio Co-Pilot)
- serializzazione JSON di oggetti non serializzabili (Graph, DataFrame, numpy)

## 4) Gap residui (alta priorita)

- Mancano prove finali end-to-end in ambiente target (deploy path)
- Mancano run test completi nel repo locale se `pytest` non disponibile nell'env corrente
- Mancano eventuali asset visuali finali (screenshot/diagrammi definitivi)
- Da finalizzare allineamento 1:1 tra criteri ufficiali e sezione README/pitch

## 5) Strategia fino alla deadline

### Fase A - Demo Reliability (immediata)

Obiettivo: zero errori live.

TODO:
- [ ] Dry-run ripetuto del percorso "golden path" (Build -> Full Agentic Demo -> Export)
- [ ] Validare fallback se GPT non disponibile
- [ ] Verificare tempi medi demo sotto i 2 minuti

### Fase B - Submission Narrative (domani)

Obiettivo: storytelling forte per giudici tecnici + business.

TODO:
- [ ] Rifinire `README.md` (inglese, chiaro, orientato ai criteri)
- [ ] Rifinire `docs/pitch.md` (problema, soluzione, novelty, impatto)
- [ ] Rifinire `docs/demo_script.md` (script 2 minuti + fallback)
- [ ] Aggiungere architecture diagram definitivo

### Fase C - Deployment Proof

Obiettivo: dimostrare "production readiness".

TODO:
- [ ] Validazione deploy Streamlit Cloud/Azure path
- [ ] Checklist configurazione secrets/env
- [ ] Smoke test finale su link pubblico

### Fase D - Packaging Finale

Obiettivo: submission pronta senza rischi.

TODO:
- [ ] Bundle finale materiale (video, repo, descrizione, link demo)
- [ ] Revisione ortografica/consistenza naming
- [ ] Verifica compliance completa con regole ufficiali

## 6) To-Do operativo (prioritizzato)

P0 (bloccanti):
- [ ] Run completo dei test con `pytest` in ambiente pronto
- [ ] Verifica full demo flow senza errori su 3 date/scenari
- [ ] Finalizzare video demo pubblico <= 2 minuti

P1 (molto importanti):
- [ ] Judge Mode guidata con copy pulito
- [ ] Diagramma architetturale finale (control plane + agent flow + evidence)
- [ ] Hardening export e naming dei file submission

P2 (nice-to-have ad alto valore):
- [ ] Upload CSV portfolio
- [ ] Toggle lingua IT/EN per output executive
- [ ] Template slide finale per pitch live

## 7) Checklist submission finale

Prima di inviare:
- [ ] Repo pubblico aggiornato e leggibile
- [ ] README completo con quickstart reale
- [ ] Link demo funzionante
- [ ] Video demo pubblico (max 2 min)
- [ ] Pitch doc + script allineati
- [ ] Architettura spiegata chiaramente
- [ ] Evidenza uso stack Microsoft/Azure
- [ ] Nessun errore critico nei flow principali

## 8) Piano di lavoro per domani (esecuzione suggerita)

1. Reliability pass (30-45 min)
2. Docs/pitch pass (45-60 min)
3. Deploy check (30 min)
4. Video capture + final QA (45 min)
5. Submission packaging (30 min)

## 9) Comandi utili per ripartenza

Esempi:

```bash
python3 -m streamlit run src/app.py
python3 -m py_compile src/app.py src/agentic_ops.py src/reporting.py src/ui_panels.py
python3 -m pytest -q tests/test_agentic_ops.py tests/test_reporting.py
```

Nota: se `pytest` non e installato nell'ambiente, installarlo prima della sessione QA finale.

## 10) Decisioni gia prese (da non perdere)

- Tab spostate in alto sotto banner (visibilita migliorata)
- Portfolio hardcoded rimosso come default: ora sample opzionale + auto-generation
- Agentic Ops con cache/timeout/fallback
- Explainability sempre visibile con badge di stato
- Action pack serializzato in modo robusto

## 11) Regola pratica per domani

Prima robustezza, poi polish.
Se una feature nuova rischia stabilita live, va posticipata dopo submission.

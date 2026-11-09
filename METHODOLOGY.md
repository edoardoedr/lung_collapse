# Collasso polmonare in VATS: metodologia e problemi trovati

Stato al 2026-11-07. Dataset: 16 pazienti (15 con risultati; karl19 si ferma su un problema del file di input).

## 1. Obiettivo

Generare i dati di training per un GNN surrogato del collasso polmonare. Il problema di partenza è inverso: dati il polmone gonfio (riferimento) e quello collassato (target) dalla TC, si cercano le pressioni pleuriche regionali che, applicate al gonfio con un modello FEM, lo portano sul collassato.

Il piano in quattro passi:

1. **Fit per regioni** sui pazienti reali: dimostra che il modello riproduce i collassi e fornisce le pressioni.
2. **Regole comuni** (fit a gradiente, legge pressione–collasso): pochi numeri trasferibili a un polmone nuovo.
3. **Collasso di un polmone di cui si ha solo la TC gonfia:** si sceglie la percentuale di collasso, si cerca la p/E media che la dà, si fa un solve in avanti.
4. **Dataset sintetico e GNN:** geometria + campo di pressione → spostamento. Il GNN sostituisce il solve FEM, non stima da solo la pressione del paziente.

## 2. Pipeline

`check_inputs` → `surface_mesh` → `hilum` → `registration` → `fem_setup` → `fem_fit`

- **`check_inputs`:** controlli sulle segmentazioni (superficie chiusa, contenimento, notch, rapporto di volume).
- **`surface_mesh`:** pulizia e remesh ACVD, circa 480 nodi.
- **`hilum`:** intersezione di vie aeree, arterie e vene con la superficie polmonare. Per ogni struttura si prende l'anello più grande; l'ilo è il centroide dei tre centri.
- **`registration`:** ora `method: "none"`. La superficie gonfia viene solo rimeshata con le stesse impostazioni, e il confronto col target avviene per punto più vicino.
- **`fem_setup`:** clamp, regioni di pressione, parete opzionale, mesh tetraedrica Gmsh (14 mm) → `problem.npz`.
- **`fem_fit`:** ottimizzazione delle pressioni.

### Configurazione attuale

| Voce | Scelta |
|---|---|
| Registrazione | nessuna |
| Clamp | solo regione dell'ilo (4 sfere), `clamp_still_mm` 0 |
| Allineamento del target | nessuno |
| Materiale | Neo-Hookean comprimibile, ν = 0.40 fisso, E_Pa = 3000 solo per convertire in Pa |
| Incognite | q = p/E, limiti [−1, 3] |
| Loss | `plane`, con termine simmetrico di peso 1 |
| Livelli | K = 1, 4, 8, 14, 25, 40 regioni |
| Core | torch, Jacobiana analitica |
| Parete (config `_wall`) | superficie gonfia di riferimento, tolleranza 5 mm, round `outer`, quadratura `nodal` |

### Perché queste scelte

- **Niente allineamento.** Con l'allineamento rigido il fit inseguiva un artefatto della registrazione: su karl04 il target ruotava di circa 16° e la distanza dalla superficie reale era 14.6 mm contro circa 3 mm senza allineamento.
- **Loss `plane`.** La corrispondenza nodo a nodo non è anatomica, perché i nodi scivolano lungo la superficie. Il residuo lungo la normale ignora questo scivolamento.
- **Niente registrazione.** Su 15 pazienti la forma è uguale con e senza (ASSD mediana 2.20 contro 2.24 mm), la rotazione è più bassa e il fit più veloce. La registrazione era anche una fonte di errore: su karl23 il volume del riferimento registrato era 1984 mL contro 2554 reali.
- **Clamp solo sull'ilo.** Senza registrazione i punti "still" clampavano il 19–58 % della superficie, contando come fermo ogni punto vicino al target anche se ci era scivolato sopra. Con il solo ilo il clamp scende dal 43 % al 5 % e la legge pressione–collasso diventa molto più pulita (r da 0.70 a 0.97). Il prezzo è una rotazione maggiore.
- **Loss simmetrica.** Misurando solo fitted → target, il polmone si rimpiccioliva su una parte del target (karl23, patient_1, patient_2: collasso del 93–98 %, ASSD 24–27 mm). Aggiungendo target → fitted l'ASSD scende a 4.0 / 2.9 / 5.0 mm e il collasso fittato è 51 / 57 / 84 % contro 51 / 59 / 86 % reali.

## 3. Codice FEM

### Modello

- Neo-Hookean comprimibile, adimensionale con E = 1, elementi P1.
- Pressione follower (formula di Nanson, area corrente), positiva verso l'interno.
- u = 0 sulle facce clampate.
- Parete come penalty sul gap linearizzato: `k/(2e) (pos(g)^2 − pos(g−e)^2)`, con g = u·n − (n·Us − φ + allow).

### Struttura (`pipeline/collapse/solvers/`)

| File | Ruolo |
|---|---|
| `base.py` | interfaccia `ForwardSolver`: `set_regions`, `solve`, `jacobian`. `fit.py` parla solo con questa |
| `getfem_solver.py` | core di riferimento (MUMPS, P1/P2), lento: 70–107 s a solve |
| `nodal.py` | tutto tranne l'assemblaggio: Newton replica di GetFEM, percorso di carico, parete, Jacobiana, solver lineari (cuDSS, pardiso, scipy) |
| `warp_solver.py` | assemblaggio con kernel Warp, derivate a mano |
| `torch_solver.py` | assemblaggio PyTorch con autodiff; `TorchFEM` è differenziabile e servirà come physics loss del GNN |
| `materials.py` | materiali come energia W(F, ν): un materiale nuovo è una funzione nuova |

### Fit

- Da grossolano a fine (K = 1 → 40), ogni livello parte dal migliore del precedente.
- `least_squares` TRF con limiti, più una penalità di regolarità tra regioni adiacenti.
- **Jacobiana analitica:** `K_t dU/dq = d(rhs)/dq` con una sola fattorizzazione. Il fit di karl04 passa da 111 min (differenze finite, GetFEM) a 7.9 min (analitica, GetFEM) e a 1–2 min con torch.
- Stop per livello (plateau) e per fit (`target_error_mm`, budget di tempo, livelli esauriti). Il migliore viene salvato a ogni miglioramento.

### Equazioni

Notazione: $\theta$ parametri del fit, $q = p/E$ pressione adimensionale, $X_i$ punti del riferimento gonfio, $x_i = X_i + u_i(\theta)$ punti fittati.

**Materiale** (E = 1, tensore di Piola $P = \partial W / \partial F$):

$$
W(F) = c_1\left(J^{-2/3} I_1 - 3\right) + d_1 (J - 1)^2,
\qquad c_1 = \frac{\mu}{2} = \frac{1}{4(1+\nu)},
\qquad d_1 = \frac{K}{2} = \frac{1}{6(1-2\nu)}
$$

È la Neo-Hookean comprimibile nella forma di GetFEM (`Compressible_Neo_Hookean`), con $I_1 = \operatorname{tr}(F^T F)$ e $J = \det F$. A ν = 0.40: $c_1 = 0.179$, $d_1 = 0.833$, $K = 1.67$.

- **Termine isocoro:** $J^{-2/3} I_1$ è il primo invariante della parte di deformazione a volume costante, quindi misura solo il cambio di forma.
- **Termine volumetrico:** $(J-1)^2$ penalizza il cambio di volume. Non è unico: esistono altre scelte, come $\ln^2 J$ o $(J^2 - 1)/2 - \ln J$.

**Compressione idrostatica.** Con $F = \lambda I$ il termine isocoro non contribuisce e, anche a grandi deformazioni,

$$
\frac{p}{E} = \frac{K}{E}(1 - J) = 1.67\,(1 - J) \quad \text{a } \nu = 0.40
$$

Il termine $(J-1)^2$ resta finito per $J \to 0$, quindi il materiale non ha barriera contro lo schiacciamento. Ne seguono tre cose:

- **Legge pressione–collasso:** con $1 - J$ pari al collasso c si ha $p/E = 1.67\,c$. La pendenza osservata (2.19) è dello stesso ordine, quindi la legge riflette in gran parte il materiale scelto, cioè ν.
- **Limite massimo:** la pressione idrostatica sostenibile è $K/E = 1.67$. Oltre, una compressione uniforme non ha equilibrio. Le regioni a p/E = 3 reggono solo perché sono locali e sostenute dal tessuto vicino.
- **Elementi schiacciati:** sotto le regioni con p/E alto $J$ scende fino a 0.086. Un volumetrico con barriera, come $\ln^2 J$ o un materiale che si irrigidisce vicino alla frazione di tessuto, lo impedirebbe, ma cambierebbe la fisica e tutte le p/E fittate.

**Equilibrio** (problema diretto): per ogni $\theta$ si cerca $u$, con $u = 0$ sul clamp, tale che per ogni $\delta u$

$$
\underbrace{\int_{\Omega_0} P : \nabla \delta u \, dV}_{\text{elasticità}}
+ \underbrace{s \sum_k q_k \int_{\Gamma_k} J F^{-T} N \cdot \delta u \, dA}_{\text{pressione follower}}
+ \underbrace{\int_{\Gamma_f} w(g) \, n_w \cdot \delta u \, dA}_{\text{parete}}
+ \underbrace{\sum_i k_s a_i \, \left[(I - n_i n_i^T) u_i\right] \cdot \delta u_i}_{\text{supporto}} = 0
$$

Il segno $s = \pm 1$ è fissato dal sign check, in modo che $q > 0$ spinga verso l'interno. I termini di parete e supporto ci sono solo se attivi ($k_s$ = `support_k`, $a_i$ area e $n_i$ normale del nodo $i$ nel riferimento).

**Parete** (penalty, con $\langle \cdot \rangle_+$ parte positiva):

$$
w(g) = \frac{k}{2e}\left(\langle g \rangle_+^2 - \langle g - e \rangle_+^2\right),
\qquad g = u \cdot n_w - G,
\qquad G = n_w \cdot U_s - \phi + a
$$

$\phi$ è la distanza con segno dalla parete allo stato di linearizzazione $U_s$, $a$ il margine ammesso, $n_w$ la normale di parete. Con `outer`, $n_w$ e $G$ restano fissi durante un solve di Newton e vengono aggiornati tra un round e l'altro. Valori: $k = 20$, $e = 0.5$.

**Campo di pressione** sul triangolo libero $t$:

$$
\text{regioni: } q_t = \theta_{k(t)}
\qquad\qquad
\text{gradiente: } q_t = \theta_0 + \sum_{a} \theta_a \, \xi_a(t),
\qquad \xi_a(t) = \frac{c_a(t) - \bar c_a}{\max c_a - \min c_a}
$$

$c_a(t)$ è la coordinata del baricentro lungo l'asse $a$ e $\bar c_a$ la sua media pesata sull'area dei triangoli liberi. Una pendenza $\theta_a$ è quindi la differenza di p/E tra i due estremi del polmone. In forma compatta $q = \Phi\,\theta$.

**Problema inverso** (`least_squares` TRF; limiti $-1 \le \theta \le 3$ su regioni e media del gradiente, $|\theta_a| \le 4$ sulle pendenze):

$$
\min_{\theta} \;
\sum_{i \in \text{liberi}} \|r_i\|^2
+ w_s \frac{n_f}{n_t} \sum_{j=1}^{n_t} \|b_j\|^2
+ \lambda^2 \sum_{(a,b)\,\text{adiacenti}} (\theta_a - \theta_b)^2
$$

Con $n_f$ punti liberi, $n_t$ punti del target, $w_s = 1$ e $\lambda = 1$. La regolarizzazione esiste solo per il modello a regioni: col gradiente non ci sono coppie adiacenti.

- **Fitted → target**, con $c_i$ punto più vicino a $x_i$ sul target e $n_i$ la normale lì (ricalcolati a ogni valutazione), $w = 0$ nelle config:

$$
r_i = \left[n_i n_i^T + w\,(I - n_i n_i^T)\right](x_i - c_i)
$$

- **Target → fitted**, con $z_j = \sum_m \beta_{jm} x_{v_m}$ punto più vicino a $y_j$ sulla superficie fittata (coordinate baricentriche $\beta$):

$$
b_j = z_j - y_j, \qquad b_j = 0 \text{ se } z_j \text{ cade su un triangolo clampato}
$$

**Errore riportato** (non è la radice del costo), con la seconda media solo sui punti non azzerati:

$$
e = \frac{\operatorname{media}_i \|r_i\| + w_s \operatorname{media}_j \|b_j\|}{1 + w_s}
$$

**Jacobiana** (Gauss-Newton: $c_i$, $n_i$, triangolo e $\beta$ tenuti fissi). Con $K_t$ tangente all'equilibrio, sui gradi di libertà liberi:

$$
K_t \frac{\partial U}{\partial q} = -\frac{\partial R}{\partial q},
\qquad \frac{\partial U}{\partial \theta} = \frac{\partial U}{\partial q}\,\Phi,
\qquad \frac{\partial r_i}{\partial \theta} = \left[n_i n_i^T + w\,(I - n_i n_i^T)\right]\frac{\partial u_i}{\partial \theta},
\qquad \frac{\partial b_j}{\partial \theta} = \sum_m \beta_{jm} \frac{\partial u_{v_m}}{\partial \theta}
$$

Un solve fallito non ha residuo: l'ottimizzatore riceve il doppio del miglior residuo del livello, cioè un trial rifiutato.

**Conversione:** $p = q\,E$, con $E$ = `E_Pa` = 3000 Pa. Il fit identifica solo il rapporto $p/E$, quindi il valore di E non cambia la forma.

### Validazione

- Senza parete, torch e Warp coincidono con GetFEM a 1e-14 o meglio.
- Con parete (`outer`) coincidono a ≤ 1e-11, con le stesse penetrazioni.
- Jacobiana contro differenze finite: circa 1e-8.
- Tempi di un solve: GetFEM 70–107 s, Warp + cuDSS 3.2 s, torch + cuDSS 5.7 s.

## 4. Pressioni

### Modello per regioni

Le regioni sono cluster di Ward, vincolati alla connettività, dello spostamento senza la parte rigida. Sono ricavate dal target del singolo paziente, quindi **non esistono per un polmone nuovo**.

Risultati (`fit_sym`, 15 pazienti, senza parete):

- Nessun solve fallito, 0.2–1.0 min a fit, K = 40 in 12 casi su 15.
- ASSD mediana circa 2.9 mm.
- **Legge pressione–collasso:** p/E medio = −0.18 + 2.19 × collasso, r 0.98, residuo massimo 0.17.
- Anteriore > posteriore in 15 su 15 (patient_1 solo di poco).
- Pendenza relativa apice–base negativa in 13 su 15.
- Il gradiente lineare spiega solo il 4–47 % della varianza delle p/E regionali.

### Modello a gradiente

q_tri = media + Σ pendenza_a · ξ_a, con ξ coordinate normalizzate sull'estensione del polmone (assi ap, si; paziente supino, anteriore in alto). Ogni triangolo libero è una regione del core e la Jacobiana è J_tri · Φ, quindi i core non sono cambiati.

Risultati (clamp sull'ilo, 11 pazienti karl, **loss ancora a una direzione**, da rifare con quella simmetrica):

- ASSD mediana 7.9 mm (ap) e 7.1 mm (ap + si), contro 2.7 mm per le regioni: buona parte della deformazione è locale e specifica del paziente.
- ap > 0 in 9 su 10, si < 0 in 8 su 10.
- Il collasso fittato supera quello reale in molti pazienti, quindi p/E e pendenze sono sovrastimate.

### Significato fisico

p/E è un carico equivalente che riunisce ritorno elastico, peso e sostegno degli organi vicini.

- **Livello medio:** è soprattutto la risposta volumetrica del materiale. In compressione idrostatica p/E = c / (3(1 − 2ν)) = 1.67 c a ν = 0.4, anche a grandi deformazioni (sezione 3, Equazioni), quindi il valore dipende da ν.
- **Gradiente anteroposteriore:** probabilmente è la gravità. Il polmone gonfio regge il proprio peso col gradiente verticale di pressione pleurica, che lo pneumotorace azzera. Questo dà una pendenza ρ g per unità di altezza, da cui E ≈ ρ g L / pendenza.
- **Stima di E** (ρ = 250 kg/m³): mediana 4.4 kPa, IQR 3.1–6.6, range 0.5–9.1 kPa. L'interpretazione è plausibile per segno e ordine di grandezza, ma non è dimostrata: E varia di un fattore 18 tra i pazienti.

## 5. Problemi trovati con la parete

Senza parete e senza `support_k` i fit convergono sempre. Con la parete e k = 0.001, 10 pazienti su 14 hanno 11–22 solve falliti e si fermano presto; solo karl06, 08, 18 e 24 convergono in meno di un minuto.

**Attenzione:** tutti i fit con k in `results/` (28–30/10) sono precedenti al fix della linearizzazione di equilibrio (30/10, 13:23). Quel fix è stato verificato solo con `wall_diagnose.py` su karl04, quindi l'effetto sui fit completi non è ancora misurato.

### Risolti

| Problema | Causa | Soluzione |
|---|---|---|
| Newton non converge sulla parete | il gradiente di distanza VTK salta tra spigoli e vertici, e ogni nodo parte su un vertice | normale di vertice interpolata (aiuto modesto) |
| Tangente di contatto indefinita | peso negativo (−27/48) nella regola di quadratura sulle facce | `wall_quadrature: "nodal"`: penetrazione da 0.4–2.3 a 0.04–0.25 mm |
| Stati non assestati accettati | il loop accettava lo stato dopo `wall_max_updates`, con spostamenti fino a 203 mm | non assestato = fallimento |
| Falsi "non assestato" | la misura si muoveva con la normale lontano dalla parete | errore di gap solo sui punti vicini alla parete |
| Salti di carico nel warm start | `set_regions` azzerava il carico precedente | percorso di carico sulle pressioni di faccia, ripartenza dall'iterata dell'ottimizzatore |
| Fallback lento | se il primo solve falliva, la Jacobiana passava alle differenze finite | il livello si ferma e il fit prosegue |
| Punti target irraggiungibili | l'allineamento rigido li portava fin 20 mm oltre la parete | niente allineamento |
| Uno stato assestato non era un punto fisso | la linearizzazione veniva rifatta a ogni ripartenza, fuori equilibrio: fallivano anche passi di 1e-6 | si conserva la linearizzazione di equilibrio (ultimi 256 stati) |

### Ipotesi verificate e scartate

- **Tempo di solve:** con `max_solve_s` 300 il risultato è identico a 120 s. I solve riusciti durano al massimo 2.6 s, gli altri non convergono mai.
- **Pressioni troppo alte:** con il limite q = 2 il fit peggiora (8.04 contro 6.07 mm) e fallisce comunque.
- **Elementi schiacciati:** nelle zone di forte collasso J scende a 0.086, ma i fallimenti sono quasi tutti `unsettled`, senza elementi invertiti.
- **Rallentamento da run in parallelo:** rifatti uno alla volta, i fallimenti restano.

### Cause ancora aperte

1. **Quasi-meccanismo.** Clamp piccolo e contatto senza attrito lasciano il polmone libero di ruotare e scivolare lungo la parete quasi senza costo. Le pressioni fittate vanno da p/E 0.1 a 3, quindi non si possono limitare.
2. **Punti negli angoli della parete.** Sono pochi punti, sempre gli stessi, a 90–160 mm dal clamp: la doccia paravertebrale e la base, dove la normale salta di 76°. Con un solo piano tangente per round, il punto attraversa l'altra faccia e nel round dopo torna indietro, quindi i round alternano e l'errore cresce.

### Rimedi

- **Supporto tangenziale `support_k` (implementato, non attivo nelle config).** Molle k·a·(I − n nᵀ) sui nodi di superficie.
  - Nel diagnose a p/E 0.5, con k ≥ 0.001 si arriva al 100 % senza fallimenti.
  - Senza parete riduce la rotazione mediana (11.0 → 8.4 → 6.1° con k 0 / 0.001 / 0.003), ma assorbe il 5–12 % del carico e introduce fallimenti Newton di causa ignota.
  - Con parete migliora l'ASSD (karl04 8.3 → 5.6 mm) ma resta lontano dai fit senza parete (2.3–4.6 mm).
- **In attesa (decisione del 2026-11-02):**
  - smoothing della sola copia della parete in `fem_setup`;
  - contatto a due piani negli angoli.
- **Clamp anatomico:** ilo + legamento polmonare; serve una definizione automatica per paziente.
- **Attrito di Coulomb: scartato.** La pleura è lubrificata e la soluzione dipenderebbe dal percorso di carico, rendendo rumorosa la funzione obiettivo.

Senza parete i fit escono comunque dalla cavità: 218 punti oltre 5 mm sui 14 pazienti, con massimi di 12–14 mm su karl23. La parete quindi ha un ruolo, ma la decisione su quanto serva è ancora aperta.

## 6. Altri problemi aperti

- **Rotazione elevata** dove il collasso è grande (karl23 73°, patient_2 44°, patient_1 34°, karl04 32°; mediana 11°). Da controllare nelle sequenze.
- **Regioni al limite di p/E:** −1 su karl10 e karl20, 3 su 6 pazienti.
- **Pressioni di K = 40** poco identificabili.
- **Neo-Hookean** non limita il rapporto di volume locale vicino a quello del tessuto senza aria (circa 0.1–0.2).
- **karl19:** il remesh si ferma su `lung_right.vtk` ("non-triangle cells").
- **Posizione del paziente** nella TC collassata: fissa la direzione del gradiente (dovrebbe essere supina).

## 7. Prossimi passi

Elenco aggiornato in `NEXT.md`, in sintesi:

1. Rifare i fit a gradiente con la loss simmetrica e ricontrollare E = ρ g L / pendenza.
2. Rifare i fit con parete dopo il fix del 30/10, poi decidere: serve davvero, con quale k, e se provare lo smoothing.
3. Setup senza target e modalità solo-forward.
4. Predizione leave-one-out sui pazienti reali.
5. Generatore del dataset e GNN.

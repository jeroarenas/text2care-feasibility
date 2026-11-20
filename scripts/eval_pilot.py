"""Evaluación del mini-piloto. Criterio "lenient": correcto si coincide con el valor esperado o es equivalente
según la codificación pedida en el esquema (p. ej., RP = RESPUESTA PARCIAL; T2b = cT2b; ABIERTO = TORACOTOMÍA),
o clínicamente equivalente (SUBCARINAL = estación 7). Las equivalencias están listadas en ACCEPT.
SCHEMA: campos cuyo valor esperado depende de una regla que el esquema no comunicaba al modelo."""
import sys, json, statistics
sys.path.insert(0, __import__('os').path.dirname(__file__))
from eval_common import norm, load

def clean(v):
    s = str(v).strip().lstrip(':').strip().strip('"').strip()
    return norm(s)

ACCEPT = {
 'abordaje_final': {'ABIERTO'}, 'abordaje_inicial': {'VATS'},
 'cT': {'T2B'}, 'cN': {'N2'}, 'cM': {'M0'}, 'ycT': {'T1C'}, 'ycN': {'N0'}, 'ypT': {'T1B'}, 'ypN': {'N0'},
 'tipo_neoadyuvancia': {'QT+IO'}, 'respuesta_neoadyuvancia': {'RP'}, 'respuesta_primera_recaida': {'RP'},
 'respuesta_sotorasib': {'RP'}, 'estado_oncologico_final': {'EE'},
 'tipo_primera_recaida': {'GANGLIONAR'}, 'tipo_segunda_progresion': {'DISTANCIA'},
 'localizacion_primera_recaida': {'SUBCARINAL', 'ESTACION 7', 'MEDIASTINO SUBCARINAL'},
 'organo_segunda_progresion': {'HEPATICO'},
 'histologia_preoperatoria': {'ADENOCARCINOMA', 'ADENOCARCINOMA PULMONAR MODERADAMENTE DIFERENCIADO'},
 'histologia_recaida': {'ADENOCARCINOMA PULMONAR'},
 'medio_biopsia': {'TC GUIADA PERCUTANEA', 'BIOPSIA PERCUTANEA GUIADA POR TC', 'TC GUIADA',
                   'PERCUTANEA GUIADA POR TC', 'BIOPSIA PERCUTANEA GUIADA POR TAC'},
 'primer_sintoma': {'TOS SECA', 'TOS SECA INTERMITENTE'},
 'tratamiento_primera_recaida': {'QUIMIORADIOTERAPIA', 'QUIMIORRADIOTERAPIA'},
 'fecha_primer_sintoma': {'2025-11-01/2025-12-31', 'FINALES DEL OTONO DE 2025', '2025-10-01/2025-12-31', '2025-OTONO'},
 'complicacion_principal': {'ARRITMIA FIBRILACION AURICULAR'},
}
# Correct content with an annotation added to the requested format (counted as correct; reported as format deviations)
FORMAT = {'icat_pa': {'55-56 (APROXIMADO)'},
          'fecha_primer_sintoma': {'2025-11/2025-12 (FINALES DE OTONO DE 2025, APROXIMADA)'},
          'kras_preop': {'MUTACION G12C'},
          'localizacion_primera_recaida': {'GANGLIONAR MEDIASTINICA SUBCARINAL (ESTACION 7)'}}
SCHEMA = {'kras_preop': {'MUTACION'},  # codificación pedida: mutación / no mutado / no realizado
          'fecha_primera_recaida': {'2027-03-22', '2027-03-14', '2027-04-04'}}  # regla (fecha de confirmación histológica) no comunicada

def judge(k, v, g):
    c = clean(v); gv = norm(g['value'])
    if c == gv: return 'ok'
    if c in FORMAT.get(k, ()): return 'ok'
    if k == 'icat_pa':
        try: return 'ok' if abs(float(c) - float(gv)) / float(gv) <= 0.05 else 'err'
        except ValueError: return 'err'
    if c in ACCEPT.get(k, ()) or c in FORMAT.get(k, ()): return 'ok'
    if c in SCHEMA.get(k, ()): return 'schema'
    return 'err'

gold, runs = load()
ORDER = ['gpt_oss_120b', 'qwen3_32b', 'medgemma_27b', 'claude-opus-5-5', 'claude_opus_5_5']
models = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER and 'jwang580' not in m]
out = {}
for m in models:
    res = {}
    for lvl in ['N1', 'N2']:
        rr = runs[m].get(lvl, {}); g0 = rr.get('greedy')
        fields = [k for k, g in gold.items() if g['level'] == lvl]
        if not g0: res[lvl] = None; continue
        samples = [rr[s] for s in sorted(rr) if s != 'greedy' and rr[s]]
        rows = []
        for k in fields:
            item = g0.get(k) or {}
            j = judge(k, item.get('value', ''), gold[k])
            # agreement of samples with the greedy answer (same verdict class and, if wrong, same value)
            agree = None
            if samples:
                ref = clean(item.get('value', ''))
                same = 0
                for s in samples:
                    sv = (s.get(k) or {}).get('value', '')
                    js = judge(k, sv, gold[k])
                    same += (js == j == 'ok') or (clean(sv) == ref)
                agree = same / len(samples)
            rows.append({'field': k, 'verdict': j, 'value': item.get('value', ''), 'conf': item.get('confidence'),
                         'agree': agree, 'docs': item.get('docs')})
        res[lvl] = {'rows': rows, 'n_samples': len(samples)}
    out[m] = res

print('ACCURACY (greedy run, lenient)')
for m, res in out.items():
    line = [m]
    for lvl in ['N1', 'N2']:
        r = res.get(lvl)
        if not r: line.append(f'{lvl}: n/a'); continue
        n = len(r['rows']); ok = sum(x['verdict'] == 'ok' for x in r['rows']); sc = sum(x['verdict'] == 'schema' for x in r['rows'])
        line.append(f"{lvl}: {ok}/{n} ok, {sc} schema, {n-ok-sc} err, samples={r['n_samples']}")
    print(' | '.join(line))
print('\nSTRICT exact match (greedy)')
for m, res in out.items():
    print(m, {lvl: (sum(norm(x['value']) == norm(gold[x['field']]['value']) for x in r['rows']), len(r['rows'])) for lvl, r in res.items() if r})
print('\nERRORS and their confidence / sample agreement')
for m, res in out.items():
    for lvl, r in res.items():
        if not r: continue
        for x in r['rows']:
            if x['verdict'] != 'ok':
                print(f"  {m} {lvl} {x['field']}: {x['verdict']} value={x['value']!r} gold={gold[x['field']]['value']!r} conf={x['conf']} agree={x['agree']}")
print('\nCONFIDENCE SIGNALS')
for m, res in out.items():
    allrows = [x for lvl, r in res.items() if r for x in r['rows'] if x['agree'] is not None]
    if not allrows: continue
    allrows.sort(key=lambda x: (x['agree'], x['conf'] or 0))
    t = len(allrows) // 3
    low = allrows[:t]
    bad = [x for x in allrows if x['verdict'] != 'ok']
    print(f"  {m}: {len(bad)} wrong of {len(allrows)}; wrong in lowest third by sample agreement: {sum(x['verdict']!='ok' for x in low)}/{len(bad)}")
    print(f"    mean agreement ok={statistics.mean(x['agree'] for x in allrows if x['verdict']=='ok'):.2f} wrong={statistics.mean(x['agree'] for x in bad) if bad else float('nan'):.2f}")
    print(f"    mean verbal conf ok={statistics.mean((x['conf'] or 0) for x in allrows if x['verdict']=='ok'):.1f} wrong={statistics.mean((x['conf'] or 0) for x in bad) if bad else float('nan'):.1f}")
    print(f"    fields with agreement < 1: {sum(x['agree']<1 for x in allrows)}; of which wrong: {sum(x['agree']<1 and x['verdict']!='ok' for x in allrows)}")
json.dump(out, open(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'results_eval.json'), 'w'), ensure_ascii=False, indent=1, default=str)

#!/usr/bin/env python3
"""Mini-piloto TEXT2CARE: extracción de variables a nivel de paciente sobre el caso sintético del HUVR.

Solo usa la biblioteca estándar de Python. Llama a Ollama (API /api/chat) o, para la referencia en la
nube, a cualquier endpoint compatible con OpenAI (/v1/chat/completions).

Por cada modelo y nivel (N1, N2) hace 1 ejecución determinista (temperatura 0) y K ejecuciones
muestreadas (temperatura 0.7, semillas distintas). La concordancia entre muestras da una señal de
confianza independiente de la confianza que declara el propio modelo.

Ejemplos:
  python3 scripts/run_pilot.py --model gpt-oss:120b --host http://localhost:11435
  python3 scripts/run_pilot.py --model qwen3:32b --host http://localhost:11435 --think off
  python3 scripts/run_pilot.py --backend anthropic --model claude-sonnet-5-5 --api-key-env ANTHROPIC_API_KEY
  python3 scripts/run_pilot.py --backend openai --model <modelo> --host <url base, p. ej. https://api.anthropic.com/v1> --api-key-env <VAR>
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RES = os.path.join(HERE, '..', 'results')

SYSTEM = (
    "Eres un sistema de extracción de información clínica. Recibes TODOS los documentos de la historia de un "
    "paciente con cáncer de pulmón, ordenados cronológicamente, y una lista de variables solicitadas por un "
    "investigador. Para cada variable devuelve:\n"
    "- value: el valor, siguiendo la codificación indicada (binarias como 0 o 1; fechas como AAAA-MM-DD; "
    "categorías en MAYÚSCULAS; números sin unidades). Si el dato es un intervalo o una aproximación, "
    "respétalo sin forzar un valor único. Si la información no consta en los documentos, devuelve \"NO CONSTA\".\n"
    "- docs: números de los documentos en los que te basas.\n"
    "- quote: cita literal breve (máximo 25 palabras) del fragmento que lo justifica; vacía si NO CONSTA.\n"
    "- confidence: tu confianza en que el valor es correcto, de 0 a 100.\n"
    "Usa solo la información de los documentos. Integra la información de varios documentos cuando sea "
    "necesario y, si hay contradicciones, da prioridad al documento más reciente o más específico. "
    "Responde únicamente con el objeto JSON."
)

def build_prompt(case, schema):
    lines = [f"- {v['field']}: {v['question']} (tipo: {v['type']}; codificación: {v['coding']})" for v in schema]
    return ("DOCUMENTOS DEL PACIENTE\n\n" + case + "\n\nVARIABLES SOLICITADAS\n" + "\n".join(lines) +
            "\n\nDevuelve un objeto JSON con una clave por variable.")

def json_schema(schema):
    item = {"type": "object",
            "properties": {"value": {"type": "string"},
                           "docs": {"type": "array", "items": {"type": "integer"}},
                           "quote": {"type": "string"},
                           "confidence": {"type": "integer"}},
            "required": ["value", "docs", "quote", "confidence"]}
    return {"type": "object", "properties": {v['field']: item for v in schema},
            "required": [v['field'] for v in schema]}

def post(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode(errors="replace")[:800]}')

def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None

def call(args, prompt, schema, temperature, seed):
    t0 = time.time()
    if args.backend == 'ollama':
        payload = {"model": args.model, "stream": False,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                   "format": json_schema(schema),
                   "options": {"temperature": temperature, "seed": seed, "num_ctx": args.num_ctx,
                               "num_predict": args.num_predict}}
        if args.think != 'default':
            payload["think"] = {"on": True, "off": False}.get(args.think, args.think)
        r = post(args.host.rstrip('/') + '/api/chat', payload, {"Content-Type": "application/json"}, args.timeout)
        msg = r.get("message", {})
        content, thinking = msg.get("content", ""), msg.get("thinking", "")
        usage = {k: r.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration")}
    elif args.backend == 'anthropic':
        key = os.environ.get(args.api_key_env, "")
        # Recent Claude models do not accept `temperature`: every call uses the default sampling.
        payload = {"model": args.model, "max_tokens": args.num_predict,
                   "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]}
        r = post('https://api.anthropic.com/v1/messages', payload,
                 {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
                 args.timeout)
        content = ''.join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        thinking, usage = "", r.get("usage", {})
    else:
        key = os.environ.get(args.api_key_env, "")
        payload = {"model": args.model, "temperature": temperature, "seed": seed,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                   "response_format": {"type": "json_object"}}
        r = post(args.host.rstrip('/') + '/chat/completions', payload,
                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, args.timeout)
        content, thinking = r["choices"][0]["message"]["content"], ""
        usage = r.get("usage", {})
    return {"content": content, "thinking_chars": len(thinking or ""), "seconds": round(time.time() - t0, 1),
            "usage": usage, "parsed": parse_json(content)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--backend', choices=['ollama', 'openai', 'anthropic'], default='ollama')
    ap.add_argument('--host', default='http://localhost:11435')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--levels', default='N1,N2')
    ap.add_argument('--k', type=int, default=5, help='ejecuciones muestreadas para la concordancia')
    ap.add_argument('--think', default='default', help="default | on | off | low | medium | high")
    ap.add_argument('--num-ctx', type=int, default=16384)
    ap.add_argument('--num-predict', type=int, default=12000)
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    case = open(os.path.join(DATA, 'case.txt')).read()
    os.makedirs(RES, exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9.]+', '_', args.model) + (f'_{args.tag}' if args.tag else '')
    for lvl in args.levels.split(','):
        schema = json.load(open(os.path.join(DATA, f'schema_{lvl}.json')))
        prompt = build_prompt(case, schema)
        runs = [('greedy', 0.0, 1)] + [(f's{i}', 0.7, 100 + i) for i in range(1, args.k + 1)]
        for rid, temp, seed in runs:
            out = os.path.join(RES, f'{name}__{lvl}__{rid}.json')
            if os.path.exists(out):
                print('ya existe, salto:', os.path.basename(out)); continue
            print(f'[{time.strftime("%H:%M:%S")}] {args.model} {lvl} {rid} ...', flush=True)
            try:
                res = call(args, prompt, schema, temp, seed)
            except Exception as e:
                print('  ERROR:', e, flush=True); continue
            ok = isinstance(res['parsed'], dict)
            n = sum(1 for v in schema if ok and v['field'] in res['parsed'])
            print(f'  {res["seconds"]} s | JSON válido: {ok} | campos: {n}/{len(schema)}', flush=True)
            res.update({"model": args.model, "backend": args.backend, "level": lvl, "run": rid,
                        "temperature": temp, "seed": seed, "think": args.think,
                        "date": time.strftime('%Y-%m-%d %H:%M')})
            json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main()

import json, glob, re, unicodedata, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
def norm(x):
    s = str(x).strip().upper()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('_', ' ')
    s = re.sub(r'\s+', ' ', s)
    try:
        f = float(s.replace(',', '.'))
        return ('%g' % f)
    except ValueError:
        return s
def load():
    gold = json.load(open(os.path.join(ROOT, 'data', 'gold.json')))
    runs = {}
    for f in glob.glob(os.path.join(ROOT, 'results', '*.json')):
        b = os.path.basename(f)[:-5]
        if 'prueba' in b: continue
        m, lvl, rid = b.split('__')
        r = json.load(open(f))
        runs.setdefault(m, {}).setdefault(lvl, {})[rid] = r.get('parsed') or None
    return gold, runs

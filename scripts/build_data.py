"""Construye los ficheros de entrada del piloto a partir del caso sintético del HUVR."""
import zipfile, re, glob, json, os, openpyxl
SRC = os.path.join(os.path.dirname(__file__), '..', 'data', 'original')
OUT = os.path.join(os.path.dirname(__file__), '..', 'data')

def docx_text(f):
    x = zipfile.ZipFile(f).read('word/document.xml').decode()
    paras = re.findall(r'<w:p[ >].*?</w:p>', x, flags=re.S)
    lines = [''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)) for p in paras]
    return '\n'.join(l for l in lines if l.strip())

parts = []
for f in sorted(glob.glob(os.path.join(SRC, '[0-1][0-9]_*.docx'))):
    name = os.path.basename(f)
    num = int(name[:2])
    if num == 0:
        continue
    title = name[3:-5].replace('_', ' ')
    t = docx_text(f)
    open(os.path.join(OUT, 'docs', name[:-5] + '.txt'), 'w').write(t)
    parts.append(f'=== DOCUMENTO {num:02d}: {title} ===\n{t}\n')
open(os.path.join(OUT, 'case.txt'), 'w').write('\n'.join(parts))

wb = openpyxl.load_workbook(os.path.join(SRC, 'SALIDA_N1_N2.xlsx'))
gold = {}
for sheet, lvl in [('SALIDA_N1', 'N1'), ('SALIDA_N2', 'N2')]:
    schema = []
    for r in wb[sheet].iter_rows(min_row=2, values_only=True):
        if not r[0]:
            continue
        q, field, val, typ, coding, src, how, cert, note = r[:9]
        schema.append({'field': field, 'question': q, 'type': typ, 'coding': coding})
        gold[field] = {'level': lvl, 'value': val, 'type': typ, 'coding': coding,
                       'source': src, 'how': how, 'certainty': cert, 'note': note}
    json.dump(schema, open(os.path.join(OUT, f'schema_{lvl}.json'), 'w'), ensure_ascii=False, indent=1)
json.dump(gold, open(os.path.join(OUT, 'gold.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print('docs:', len(parts), '| N1/N2 fields:', sum(g['level']=='N1' for g in gold.values()), sum(g['level']=='N2' for g in gold.values()),
      '| case chars:', sum(len(p) for p in parts))

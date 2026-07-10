# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Matheus Henrique

"""Merge COCO/YOLO/VOC/LabelMe aquatic datasets into one YOLO detection dataset.

Install:  py -m pip install pillow
Inspect:  py aquavision_builder.py inventory --root ./sources
Build:    py aquavision_builder.py build --root ./sources --out ./aquavision_v0 --zip
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, shutil, sys, zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:
    raise SystemExit("Instale Pillow: py -m pip install pillow")

IMG_EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
SKIP={'annotated','.extracted','preview','previews','visualization','visualizations','rendered','__macosx','.git'}

@dataclass
class Box:
    x:float; y:float; w:float; h:float; label:str='unknown'
@dataclass
class Sample:
    image:Path; boxes:list[Box]; source:str; fmt:str; ann:Path|None=None

def skip(p:Path, root:Path)->bool:
    try: parts=p.relative_to(root).parts[:-1]
    except ValueError: parts=p.parts[:-1]
    return any(x.lower() in SKIP for x in parts)

def source_of(p:Path, root:Path)->str:
    try: return p.relative_to(root).parts[0]
    except Exception: return p.parent.name

def scan(root:Path):
    imgs=[]; js=[]; xml=[]; txt=[]; size=0
    for p in root.rglob('*'):
        if not p.is_file() or skip(p,root): continue
        try: size+=p.stat().st_size
        except OSError: pass
        e=p.suffix.lower()
        if e in IMG_EXTS: imgs.append(p)
        elif e=='.json': js.append(p)
        elif e=='.xml': xml.append(p)
        elif e=='.txt': txt.append(p)
    return imgs,js,xml,txt,size

def load_json(p):
    try: return json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception: return None

def is_coco(d): return isinstance(d,dict) and all(isinstance(d.get(k),list) for k in ('images','annotations','categories'))
def is_labelme(d): return isinstance(d,dict) and isinstance(d.get('shapes'),list)
def risid_ok(p:Path,g:int):
    n=p.name.lower(); variants=('annotations_2cat','annotations_5cat','annotations_7cat')
    return not any(v in n for v in variants) or f'annotations_{g}cat' in n

def indices(imgs):
    byname=defaultdict(list); bystem=defaultdict(list)
    for p in imgs: byname[p.name.lower()].append(p); bystem[p.stem.lower()].append(p)
    return byname,bystem

def score(a:Path,b:Path):
    s=0
    for x,y in zip(a.parts,b.parts):
        if x.lower()!=y.lower(): break
        s+=1
    return s

def candidate_paths(ann:Path, name:str, root:Path):
    name=name.replace('\\','/').strip(); bn=Path(name).name; stem=Path(name).stem
    candidates=[]
    if name:
        candidates.append(Path(name))
        candidates.append(Path(bn))
    for base in (root, ann.parent, ann.parent.parent, ann.parent.parent.parent):
        for rel in (
            Path(name), Path(bn),
            Path('images')/bn, Path('JPEGImages')/bn, Path('Images')/bn,
            Path('images')/Path(name), Path('JPEGImages')/Path(name), Path('Images')/Path(name),
            Path('images')/stem, Path('JPEGImages')/stem, Path('Images')/stem,
        ):
            candidates.append(base/rel)
    if ann.parent.name.lower() in {'annotations','annotation','ann','anns'}:
        for base in (ann.parent.parent, ann.parent.parent.parent):
            for rel in (Path('images')/bn, Path('JPEGImages')/bn, Path('Images')/bn, Path('images')/Path(name), Path('JPEGImages')/Path(name), Path('Images')/Path(name)):
                candidates.append(base/rel)
    seen=set(); out=[]
    for c in candidates:
        try:
            key=str(c.resolve())
        except Exception:
            key=str(c)
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def resolve(name:str, ann:Path, root:Path, byname, bystem):
    name=name.replace('\\','/').strip(); bn=Path(name).name
    for c in candidate_paths(ann,name,root):
        if c.is_file() and c.suffix.lower() in IMG_EXTS and not skip(c,root): return c.resolve()
    m=byname.get(bn.lower(),[]) or bystem.get(Path(bn).stem.lower(),[])
    return max(m,key=lambda p:score(p.parent,ann.parent)).resolve() if m else None

def segbox(seg):
    xs=[]; ys=[]
    if not isinstance(seg,list): return None
    for poly in seg:
        if not isinstance(poly,list): continue
        for i in range(0,len(poly)-1,2):
            try: xs.append(float(poly[i])); ys.append(float(poly[i+1]))
            except Exception: pass
    return (min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)) if xs else None

def parse_coco(p,d,root,byname,bystem):
    cats={int(c.get('id',-1)):str(c.get('name',c.get('id','unknown'))) for c in d['categories']}
    infos={int(i['id']):i for i in d['images'] if 'id' in i}; grouped=defaultdict(list)
    for a in d['annotations']:
        try: iid=int(a['image_id'])
        except Exception: continue
        bb=a.get('bbox') if isinstance(a.get('bbox'),list) and len(a['bbox'])>=4 else segbox(a.get('segmentation'))
        if not bb: continue
        try: x,y,w,h=map(float,bb[:4])
        except Exception: continue
        if w>0 and h>0: grouped[iid].append(Box(x,y,w,h,cats.get(int(a.get('category_id',-1)),'unknown')))
    out=[]
    for iid,info in infos.items():
        ip=resolve(str(info.get('file_name','')),p,root,byname,bystem)
        if ip and grouped.get(iid): out.append(Sample(ip,grouped[iid],source_of(p,root),'COCO',p))
    return out

def paired(p, byname, bystem, root):
    try:
        r=ET.parse(p).getroot(); filename=(r.findtext('filename') or '').strip()
    except Exception:
        filename=''
    for name in [filename, p.stem, Path(filename).stem if filename else '']:
        if not name: continue
        for c in candidate_paths(p,name,root):
            if c.is_file() and c.suffix.lower() in IMG_EXTS and not skip(c,root): return c.resolve()
    for key in [p.stem.lower(), Path(filename).stem.lower() if filename else '']:
        if not key: continue
        m=byname.get(key,[]) or bystem.get(key,[])
        if m: return max(m,key=lambda x:score(x.parent,p.parent)).resolve()
    return None

def parse_yolo(p,ip):
    try:
        with Image.open(ip) as im: W,H=im.size
    except Exception: return None
    boxes=[]
    for line in p.read_text(encoding='utf-8-sig',errors='ignore').splitlines():
        q=line.split()
        if len(q)<5: continue
        try: c,xc,yc,w,h=q[:5]; xc,yc,w,h=map(float,(xc,yc,w,h))
        except Exception: continue
        boxes.append(Box((xc-w/2)*W,(yc-h/2)*H,w*W,h*H,c))
    return boxes

def parse_voc(p):
    try: r=ET.parse(p).getroot()
    except Exception: return []
    out=[]
    for o in r.findall('.//object'):
        b=o.find('bndbox')
        if b is None: continue
        try:
            x1=float(b.findtext('xmin')); y1=float(b.findtext('ymin')); x2=float(b.findtext('xmax')); y2=float(b.findtext('ymax'))
        except Exception: continue
        if x2>x1 and y2>y1: out.append(Box(x1,y1,x2-x1,y2-y1,o.findtext('name','unknown')))
    return out

def parse_labelme(p,d,root,byname,bystem):
    ip=resolve(str(d.get('imagePath','')),p,root,byname,bystem)
    if not ip: return None
    boxes=[]
    for s in d['shapes']:
        pts=s.get('points',[])
        if not pts: continue
        try: xs=[float(x[0]) for x in pts]; ys=[float(x[1]) for x in pts]
        except Exception: continue
        x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys)
        if x2>x1 and y2>y1: boxes.append(Box(x1,y1,x2-x1,y2-y1,str(s.get('label','unknown'))))
    return Sample(ip,boxes,source_of(p,root),'LabelMe',p) if boxes else None

def collect(root,g):
    imgs,js,xmls,txts,_=scan(root); byname,bystem=indices(imgs); out=[]; used=set(); fmts=Counter(); ignored=[]
    for p in js:
        d=load_json(p)
        if is_coco(d):
            if not risid_ok(p,g): ignored.append(p); continue
            x=parse_coco(p,d,root,byname,bystem); out+=x; used|={s.image for s in x}; fmts['COCO']+=len(x)
        elif is_labelme(d):
            s=parse_labelme(p,d,root,byname,bystem)
            if s: out.append(s); used.add(s.image); fmts['LabelMe']+=1
    for p in xmls:
        ip=paired(p,byname,bystem,root)
        if not ip or ip.resolve() in used: continue
        b=parse_voc(p)
        if b: out.append(Sample(ip.resolve(),b,source_of(p,root),'VOC',p)); used.add(ip.resolve()); fmts['VOC']+=1
    for p in txts:
        if p.name.lower() in {'readme.txt','classes.txt','train.txt','val.txt','test.txt'}: continue
        ip=paired(p,byname,bystem,root)
        if not ip or ip.resolve() in used: continue
        b=parse_yolo(p,ip)
        if b: out.append(Sample(ip.resolve(),b,source_of(p,root),'YOLO',p)); used.add(ip.resolve()); fmts['YOLO']+=1
    return out,fmts,ignored

def dhash(im):
    x=ImageOps.grayscale(im).resize((9,8),Image.Resampling.LANCZOS); px=list(x.getdata()); v=0
    for r in range(8):
        for c in range(8): v=(v<<1)|int(px[r*9+c]>px[r*9+c+1])
    return f'{v:016x}'

def split_for(key,tr,va):
    r=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)/(16**12)
    return 'train' if r<tr else ('val' if r<tr+va else 'test')

def safe_box(b,W,H):
    x1=max(0,min(W,b.x)); y1=max(0,min(H,b.y)); x2=max(0,min(W,b.x+b.w)); y2=max(0,min(H,b.y+b.h))
    return Box(x1,y1,x2-x1,y2-y1,b.label) if x2-x1>=1 and y2-y1>=1 else None

def inventory(root,g):
    imgs,js,xmls,txts,size=scan(root)
    print(f'Raiz: {root.resolve()}\nTamanho analisado: {size/1024**3:.2f} GB\nImagens: {len(imgs):,} | JSON: {len(js):,} | XML: {len(xmls):,} | TXT: {len(txts):,}')
    for p in js:
        d=load_json(p)
        if is_coco(d):
            st='USAR' if risid_ok(p,g) else 'IGNORAR: taxonomia alternativa'
            print(f'[{st}] {p.relative_to(root)} -> {len(d["images"]):,} imagens, {len(d["annotations"]):,} objetos, {len(d["categories"])} classes')
    print('\nPastas ignoradas:',', '.join(sorted(SKIP)))

def build(a):
    root=a.root.resolve(); out=a.out.resolve()
    if out.exists():
        if not a.force: raise SystemExit(f'{out} já existe; use --force.')
        shutil.rmtree(out)
    for s in ('train','val','test'):
        (out/'images'/s).mkdir(parents=True,exist_ok=True); (out/'labels'/s).mkdir(parents=True,exist_ok=True)
    samples,fmts,ignored=collect(root,a.risid_granularity)
    print('Amostras encontradas:',len(samples),'| formatos:',dict(fmts))
    seen={}; rows=[]; discarded=Counter()
    for i,s in enumerate(samples,1):
        try:
            with Image.open(s.image) as raw: im=ImageOps.exif_transpose(raw).convert('RGB')
        except Exception: discarded['imagem_invalida']+=1; continue
        ow,oh=im.size; h=dhash(im)
        if h in seen: discarded['duplicata_visual']+=1; continue
        seen[h]=str(s.image)
        scale=min(1.0,a.max_side/max(ow,oh)) if a.max_side>0 else 1.0
        nw,nh=(max(1,round(ow*scale)),max(1,round(oh*scale)))
        if scale<1: im=im.resize((nw,nh),Image.Resampling.LANCZOS)
        boxes=[]
        for b in s.boxes:
            z=safe_box(Box(b.x*scale,b.y*scale,b.w*scale,b.h*scale,b.label),nw,nh)
            if z: boxes.append(z)
        if not boxes: discarded['sem_boxes_validas']+=1; continue
        sp=split_for(s.source+'|'+h,a.train,a.val); ident=hashlib.sha256((s.source+'|'+h).encode()).hexdigest()[:16]
        base=s.source.lower().replace(' ','_')+'_'+ident; ip=out/'images'/sp/(base+'.jpg'); lp=out/'labels'/sp/(base+'.txt')
        im.save(
            ip,
            'JPEG',
            quality=a.jpeg_quality,
            optimize=False,
            progressive=False,
            subsampling='4:4:4',
        )
        lines=[]
        for b in boxes:
            xc=(b.x+b.w/2)/nw; yc=(b.y+b.h/2)/nh; bw=b.w/nw; bh=b.h/nh
            if all(math.isfinite(v) for v in (xc,yc,bw,bh)): lines.append(f'0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}')
        lp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        rows.append({'split':sp,'image':ip.relative_to(out).as_posix(),'label':lp.relative_to(out).as_posix(),'source':s.source,'original':str(s.image),'annotation':str(s.ann or ''),'format':s.fmt,'width':nw,'height':nh,'objects':len(boxes),'original_classes':'|'.join(sorted({b.label for b in boxes}))})
        if i%250==0 or i==len(samples): print(f'\rProcessados {i:,}/{len(samples):,}; aceitos {len(rows):,}',end='')
    print()
    if not rows: raise SystemExit('Nada foi gerado. Rode inventory e envie a saída.')
    with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out/'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: floating_object\n',encoding='utf-8')
    sc=Counter(r['split'] for r in rows); src=Counter(r['source'] for r in rows)
    report=['# AquaVision v0','',f'- Imagens: {len(rows):,}',f'- Train: {sc["train"]:,}',f'- Val: {sc["val"]:,}',f'- Test: {sc["test"]:,}',f'- Descartadas: {sum(discarded.values()):,} — {dict(discarded)}','', '## Fontes']+[f'- {k}: {v:,}' for k,v in src.most_common()]+['','## Treino','```bash','pip install ultralytics',f'yolo detect train data="{out / "data.yaml"}" model=yolo26n.pt imgsz=960 epochs=100 batch=-1','```']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print('Dataset:',out,'\nImagens:',len(rows),'\nDescartadas:',dict(discarded))
    if a.zip:
        zp=out.with_suffix('.zip')
        with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in out.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(out.parent))
        print('ZIP:',zp)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    i=sub.add_parser('inventory'); i.add_argument('--root',type=Path,required=True); i.add_argument('--risid-granularity',type=int,choices=(2,5,7),default=2)
    b=sub.add_parser('build'); b.add_argument('--root',type=Path,required=True); b.add_argument('--out',type=Path,required=True); b.add_argument('--risid-granularity',type=int,choices=(2,5,7),default=2); b.add_argument('--max-side',type=int,default=1920); b.add_argument('--jpeg-quality',type=int,default=90); b.add_argument('--train',type=float,default=.8); b.add_argument('--val',type=float,default=.1); b.add_argument('--zip',action='store_true'); b.add_argument('--force',action='store_true')
    a=p.parse_args()
    if a.cmd=='inventory': inventory(a.root.resolve(),a.risid_granularity)
    else: build(a)
if __name__=='__main__': main()
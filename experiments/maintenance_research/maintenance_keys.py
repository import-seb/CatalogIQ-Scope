"""Conservative exact-match keys; never guess missing/scientific-notation identifiers."""
import re
import unicodedata
import numpy as np
from core import MISSING,metrics
from data import fingerprint


def gtin(value):
    value=str(value or '').strip()
    if len(value) not in (8,12,13,14) or not value.isascii() or not value.isdigit() or set(value)=={'0'}:return None
    expected=(-sum(int(d)*(3 if i%2==0 else 1) for i,d in enumerate(reversed(value[:-1]))))%10
    return value.zfill(14) if expected==int(value[-1]) else None


def retailer_sku(row):
    retailer=re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(row.get('Retailer','')))).strip().casefold()
    sku=unicodedata.normalize('NFKC',str(row.get('Sku',''))).strip()
    if retailer in MISSING or sku.casefold() in MISSING:return None
    return (retailer,sku)


def keys(row):return {'gtin14':gtin(row.get('Upc')),'retailer_sku':retailer_sku(row),'full_features':fingerprint(row)}


def evaluate_key_memory(incoming,validation,y,baseline,names):
    results={}
    for scheme in ('gtin14','retailer_sku','full_features'):
        for guarded in ([False,True] if scheme!='full_features' else [False]):
            grouped={}
            for row in incoming:
                key=row['keys'][scheme]
                if key is not None:
                    grouped.setdefault(key,[]).append((row['y'],row['keys']['full_features']))
            valid={key:values for key,values in grouped.items() if len({value[0] for value in values})==1}
            pred=baseline.copy();hits=0
            for i,row in enumerate(validation):
                values=valid.get(row['keys'][scheme])
                if values and (not guarded or row['keys']['full_features'] in {v[1] for v in values}):
                    pred[i]=values[0][0];hits+=1
            results[scheme+('_with_feature_guard' if guarded else '')]={
                'incoming_identifier_coverage':sum(r['keys'][scheme] is not None for r in incoming)/len(incoming),
                'validation_identifier_coverage':sum(r['keys'][scheme] is not None for r in validation)/len(validation),
                'conflicting_incoming_identifiers':len(grouped)-len(valid),'validation_hits':hits,
                'validation_suggestion_coverage':hits/len(validation),'accuracy':float((pred==y).mean()),
                'previously_correct_became_wrong':int(((baseline==y)&(pred!=y)).sum()),
                'previously_wrong_became_correct':int(((baseline!=y)&(pred==y)).sum()),
                'mandatory_human_review':True,'genuinely_new_confirmation_data':False}
    return results

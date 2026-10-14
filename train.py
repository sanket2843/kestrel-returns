"""Usage: python train.py [data_dir]  (default ./data). Needs train.csv, test_unlabelled.csv, customers.csv, products.csv. Data is NOT in the repo (policy s10)."""
import sys,os,joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from kestrel import *
D=sys.argv[1] if len(sys.argv)>1 else 'data'
tr=pd.read_csv(f'{D}/train.csv');te=pd.read_csv(f'{D}/test_unlabelled.csv')
cu=pd.read_csv(f'{D}/customers.csv');pr=pd.read_csv(f'{D}/products.csv')
tr=tr.sort_values('source').drop_duplicates('order_id')      # partner_feed re-imports: keep crm copy
if te.order_id.duplicated().any(): te=te.sort_values('source').drop_duplicates('order_id')
X=build(tr,pr,cu);y=tr.returned.values;ts=pd.to_datetime(tr.order_placed_at).values
cut=np.datetime64('2026-04-01');a=ts<cut
m=make_model().fit(X[a],y[a]);pv=m.predict_proba(X[~a])[:,1]
print('VALIDATION (train<Apr-2026, test Apr-Jun 2026) n=%d returns=%d AUC=%.3f AP=%.3f base=%.3f'%((~a).sum(),y[~a].sum(),roc_auc_score(y[~a],pv),average_precision_score(y[~a],pv),y[~a].mean()))
thr=float(np.quantile(pv,.70)); print('call threshold (top 30%% of validation scores): %.4f'%thr)
m=make_model().fit(X,y)                                          # final: all 15 months
rates={c:{k:(float(v.mean()),int(len(v))) for k,v in pd.Series(y,index=X.index).groupby(X[c].astype(str)) } for c in LABEL}
bundle={'model':m,'threshold':thr,'base':float(y.mean()),'rates':rates,'products':pr}
joblib.dump(bundle,'model.joblib')
s=m.predict_proba(build(te,pr,cu))[:,1]
pd.DataFrame({'order_id':te.order_id.values,'score':s}).to_csv('predictions.csv',index=False)
print('wrote predictions.csv',len(te),'rows; flagged for call at threshold:',(s>=thr).mean().round(3))

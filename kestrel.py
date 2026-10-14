import numpy as np, pandas as pd
CAT=['sales_channel','payment_mode','is_gift','sku','family','state','shield_member']
NUM=['discount_pct','qty','ratio','promised_delivery_days','no_addr','customer_prior_orders','customer_prior_returns',
     'prior_rate','hour','dow','tenure','age_sku','list_price_inr','warranty_months','note_len']
RAW=['order_placed_at','customer_id','sku','sales_channel','payment_mode','discount_pct','qty','order_value_inr',
     'promised_delivery_days','delivery_pincode','is_gift','customer_prior_orders','customer_prior_returns','delivery_note',
     'state','shield_member','signup_date']
# NOTE: service-event and pickup columns are deliberately NOT used: they are written after a return happens (leakage).
def build(d, products, customers=None):
    d=d.copy()
    for c in RAW:
        if c not in d: d[c]=np.nan
    if customers is not None:
        d=d.drop(columns=['state','shield_member','signup_date']).merge(customers,on='customer_id',how='left')
    d=d.merge(products,on='sku',how='left')
    ts=pd.to_datetime(d['order_placed_at'],errors='coerce')
    for c in ['discount_pct','qty','order_value_inr','promised_delivery_days','customer_prior_orders','customer_prior_returns','list_price_inr','warranty_months']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['ratio']=d.order_value_inr/(d.list_price_inr*d.qty)
    d.loc[d.ratio>10,'ratio']/=100          # Oct-2025 gateway stored values 100x too high
    d['no_addr']=(pd.to_numeric(d.delivery_pincode,errors='coerce')==0).astype(int)
    d['prior_rate']=d.customer_prior_returns/(d.customer_prior_orders+1)
    d['hour']=ts.dt.hour; d['dow']=ts.dt.dayofweek
    d['tenure']=(ts-pd.to_datetime(d.signup_date,errors='coerce')).dt.days
    d['age_sku']=(ts-pd.to_datetime(d.launch_date,errors='coerce')).dt.days
    d['note_len']=d.delivery_note.fillna('').astype(str).str.len()
    for c in CAT: d[c]=d[c].astype(object).where(d[c].notna(),np.nan)
    return d[CAT+NUM]
def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    ct=ColumnTransformer([('c',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan),CAT)],remainder='passthrough')
    return Pipeline([('ct',ct),('m',H(categorical_features=[True]*len(CAT)+[False]*len(NUM),max_iter=200,learning_rate=0.05,random_state=0))])
LABEL={'payment_mode':'Payment mode','sales_channel':'Sales channel','family':'Product','shield_member':'Shield member','is_gift':'Gift order','sku':'This model'}
def reasons(bundle, rec):
    base=bundle['base']; out=[]
    for col,tab in bundle['rates'].items():
        v=rec.get(col)
        if v in tab and tab[v][1]>=60:
            r,n=tab[v]
            if r>=1.3*base: out.append((r/base,f"{LABEL[col]} = {v}: {r:.0%} of such orders were returned (average {base:.0%})"))
            elif r<=0.7*base: out.append((base/max(r,.01)*0.8,f"{LABEL[col]} = {v}: only {r:.0%} returned (average {base:.0%}) - lowers risk"))
    po,pr=rec.get('customer_prior_orders') or 0,rec.get('customer_prior_returns') or 0
    if pr and pr>0: out.append((3,f"Customer has returned {pr} of {po} earlier orders"))
    if str(rec.get('delivery_pincode','')).strip('0')=='' and rec.get('delivery_pincode') is not None: out.append((1.5,"No delivery address captured (walk-in order)"))
    out.sort(key=lambda x:-x[0]); r=[t for _,t in out[:4]]
    return r or ["No strong single factor; score reflects a combination of small effects"]

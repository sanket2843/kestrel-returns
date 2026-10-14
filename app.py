"""python app.py  -> http://localhost:8000   (no API key, no internet needed)"""
import json,joblib,pandas as pd,os
from http.server import BaseHTTPRequestHandler,HTTPServer
from kestrel import build,reasons
B=joblib.load(os.path.join(os.path.dirname(__file__) or '.','model.joblib'))
EXAMPLE={"order_placed_at":"2026-08-12 14:05:00","customer_id":"C000123","sku":B['products'].sku.iloc[0],"sales_channel":"web","payment_mode":"cod",
 "discount_pct":10,"qty":1,"order_value_inr":4500,"promised_delivery_days":5,"delivery_pincode":"411001","is_gift":"N",
 "customer_prior_orders":3,"customer_prior_returns":1,"delivery_note":"","shield_member":"N","state":"Maharashtra","signup_date":"2024-03-01"}
def predict(rec):
    if not isinstance(rec,dict) or not rec: raise ValueError("Send one order as a JSON object.")
    X=build(pd.DataFrame([rec]),B['products']); s=float(B['model'].predict_proba(X)[:,1][0])
    call=s>=B['threshold']
    return {"score":round(s,4),"decision":"CALL customer before dispatch" if call else "Dispatch as normal",
            "why":reasons(B,rec),
            "note":"Score is a ranking (higher = more likely returned), not an exact probability. Average return rate is %.1f%%. Do not hold the order; call."%(100*B['base'])}
PAGE="""<!doctype html><meta charset=utf-8><title>Kestrel returns risk</title><body style="font-family:Arial;max-width:760px;margin:24px auto">
<h2>Kestrel - returns risk check</h2><p>Paste one order as JSON and press Check.</p>
<textarea id=t rows=17 style="width:100%;font-family:monospace">__EX__</textarea><br><button onclick=go() style="padding:8px 18px;margin:8px 0">Check</button>
<div id=o></div><script>async function go(){let o=document.getElementById('o');try{let r=await fetch('/predict',{method:'POST',body:document.getElementById('t').value});
let j=await r.json();if(j.error){o.innerHTML='<p style=color:#b00>'+j.error+'</p>';return}
o.innerHTML='<h3>'+j.decision+'</h3><p>Risk score: <b>'+j.score+'</b></p><ul>'+j.why.map(x=>'<li>'+x+'</li>').join('')+'</ul><small>'+j.note+'</small>'}
catch(e){o.innerHTML='<p style=color:#b00>Could not read the order. Check the JSON.</p>'}}</script>"""
class H(BaseHTTPRequestHandler):
    def _send(self,code,body,ct):
        b=body.encode();self.send_response(code);self.send_header('Content-Type',ct);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self): self._send(200,PAGE.replace('__EX__',json.dumps(EXAMPLE,indent=1)),'text/html; charset=utf-8')
    def do_POST(self):
        try:
            rec=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}')
            self._send(200,json.dumps(predict(rec)),'application/json')
        except Exception as e: self._send(400,json.dumps({"error":"Could not score this order: %s"%e}),'application/json')
    def log_message(self,*a): pass
if __name__=='__main__':
    print('Open http://localhost:8000'); HTTPServer(('0.0.0.0',8000),H).serve_forever()

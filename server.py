"""Local demonstration server. No accounts, external services, or hardware credentials."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import threading
import uuid
from energy.scenarios import make_scenario
from energy.dispatch import solve
from energy.quantum import build_problem, from_dispatch, benchmark
from energy.integration import evaluate_city_candidates

# Native solver backends share process-wide thread pools. Serialize computations
# on one stable worker rather than executing HiGHS/Aer in transient HTTP threads.
COMPUTE=ThreadPoolExecutor(max_workers=1,thread_name_prefix='energy-compute')
ROOT=Path(__file__).parent
JOBS={}; LOCK=threading.Lock(); QUANTUM_LOCK=threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self, data, status=200):
        body=json.dumps(data,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u=urlparse(self.path); params=parse_qs(u.query)
        if u.path=='/api/scenario':
            try: self.reply(make_scenario(params.get('mode',['weekday'])[0],params.get('stress',['normal'])[0]))
            except ValueError as e: self.reply({'error':str(e)},400)
        elif u.path.startswith('/api/jobs/'):
            with LOCK: job=JOBS.get(u.path.rsplit('/',1)[-1])
            self.reply(job or {'error':'Job not found'},200 if job else 404)
        elif u.path in ('/','/index.html'):
            body=(ROOT/'web/index.html').read_bytes()
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        else: self.reply({'error':'Not found'},404)
    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<2_000_000: raise ValueError('Request too large or empty')
            req=json.loads(self.rfile.read(length))
            if self.path=='/api/dispatch':
                self.reply(COMPUTE.submit(solve,req['config']).result())
            elif self.path=='/api/quantum':
                if not QUANTUM_LOCK.acquire(blocking=False):
                    self.reply({'error':'A quantum benchmark is already running'},409); return
                key=str(uuid.uuid4())
                with LOCK:
                    if len(JOBS)>30: JOBS.pop(next(iter(JOBS)))
                    JOBS[key]={'status':'running'}
                def work():
                    try:
                        p=build_problem() if req.get('scope','demo')=='demo' else from_dispatch(req['dispatch'],int(req.get('hour',13)))
                        result=benchmark(p,alpha=float(req.get('alpha',.25)),restarts=int(req.get('restarts',2)),
                                         maxiter=int(req.get('maxiter',70)),seed=int(req.get('seed',7)))
                        if req.get('scope')=='slot':
                            result['city_recourse']=evaluate_city_candidates(req['dispatch'],result,int(req.get('hour',13)))
                        outcome={'status':'done','result':result}
                    except Exception as e: outcome={'status':'error','error':str(e)}
                    finally: QUANTUM_LOCK.release()
                    with LOCK: JOBS[key]=outcome
                COMPUTE.submit(work)
                self.reply({'job':key},202)
            else: self.reply({'error':'Not found'},404)
        except (ValueError,KeyError,TypeError,RuntimeError) as e: self.reply({'error':str(e)},400)
        except Exception as e: self.reply({'error':f'{type(e).__name__}: {e}'},500)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8000); args=parser.parse_args()
    print(f'Quantum City running on port {args.port}',flush=True)
    ThreadingHTTPServer(('0.0.0.0',args.port),Handler).serve_forever()

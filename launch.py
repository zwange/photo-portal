"""Start or reuse the local portal and open the default browser."""
import argparse,json,subprocess,sys,time,urllib.request,webbrowser
from pathlib import Path
APP=Path(__file__).resolve().parent
def current():
    try:
        info=json.loads((APP/'server.json').read_text())
        with urllib.request.urlopen(info['url']+'api/health',timeout=1) as response:
            if json.load(response).get('app')=='photo-portal-v1':return info['url']
    except (OSError,ValueError):pass
    return None
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    url=current()
    if not url:
        out=(APP/'server-out.log').open('w');err=(APP/'server-err.log').open('w')
        process=subprocess.Popen([sys.executable,'-u',str(APP/'server.py')],stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        for _ in range(60):
            time.sleep(.2);url=current()
            if url:break
            if process.poll() is not None:raise RuntimeError('Portal exited. Check server-err.log.')
        out.close();err.close()
    if not url:raise RuntimeError('Portal did not become ready. Check server-err.log.')
    print(url)
    if not args.no_browser:webbrowser.open(url)
if __name__=='__main__':main()

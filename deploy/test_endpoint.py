import socket, ssl, time, urllib.request, sys

HOST = "face.broadmesse.net"
PORT = 443
PATHS = ["/modelAi/dh/health", "/modelAi/dh/", "/modelAi/dh/api/monet/robot/ask"]

def dns():
    t = time.time()
    try:
        info = socket.gethostbyname_ex(HOST)
        return True, info[2], (time.time() - t) * 1000
    except Exception as e:
        return False, str(e), (time.time() - t) * 1000

def cert_raw():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((HOST, PORT), timeout=10) as s:
            with ctx.wrap_socket(s, server_hostname=HOST) as ss:
                return True, ss.getpeercert()
    except Exception as e:
        return False, str(e)

def cert_verify():
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((HOST, PORT), timeout=10) as s:
            with ctx.wrap_socket(s, server_hostname=HOST) as ss:
                ss.getpeercert()
        return True, "trusted"
    except ssl.SSLError as e:
        return False, f"SSL error: {e}"
    except Exception as e:
        return False, str(e)

def req(path):
    url = f"https://{HOST}{path}"
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=15)
        rt = (time.time() - t) * 1000
        body = r.read(2000).decode('utf-8', 'replace')
        return r.status, dict(r.headers), body, rt
    except urllib.error.HTTPError as e:
        rt = (time.time() - t) * 1000
        return e.code, dict(e.headers), e.read(2000).decode('utf-8', 'replace'), rt
    except Exception as e:
        rt = (time.time() - t) * 1000
        return None, {}, f"{e!r}", rt

print("=== DNS ===")
ok, val, ms = dns()
print(f"  resolve_ok={ok} value={val} ({ms:.0f}ms)")

print("=== CERT raw ===")
cok, c = cert_raw()
if cok:
    print("  subject:", c.get('subject'))
    print("  issuer:", c.get('issuer'))
    print("  notBefore:", c.get('notBefore'))
    print("  notAfter:", c.get('notAfter'))
    print("  SAN:", c.get('subjectAltName'))
else:
    print("  ", c)

print("=== CERT verify ===")
vok, vmsg = cert_verify()
print(f"  trusted={vok} {vmsg}")

for p in PATHS:
    print(f"=== REQ GET {p} ===")
    st, hd, body, rt = req(p)
    print(f"  status={st} time={rt:.0f}ms")
    print(f"  headers={ {k: hd[k] for k in list(hd)[:10]} }")
    print(f"  body={body[:500]!r}")

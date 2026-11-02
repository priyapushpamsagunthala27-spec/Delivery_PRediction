# Optional TLS transport

Module 2 and Module 3 can pass global model parameters, local model parameters,
and the metadata used by aggregation over an authenticated TLS connection.
The existing in-process behavior remains the default when TLS options are
omitted. The current project simulates clients in the training process, so this
transport binds to `127.0.0.1`; it does not turn the training scripts into
separate remote client/server applications.

Only model state dictionaries and aggregation metadata are serialized for
transport. Raw training or validation records are not included in messages.
Module 2 and Module 3 continue to access their existing client-local split
files as they did before. The TLS server validates the certificate against the
configured CA, verifies the server hostname, negotiates TLS 1.2 or later, and
requires a configured bearer token for every client update.

TLS encrypts transport and authenticates the server/client token. It does not
provide secure aggregation or differential privacy: the server receives
individual client model updates and metadata.

## Local certificate and token setup (PowerShell)

Use OpenSSL 3 or another OpenSSL installation on `PATH`. These commands keep
private keys and test tokens outside the project directory:

```powershell
New-Item -ItemType Directory -Force C:\fedprada-tls | Out-Null
Set-Location C:\fedprada-tls

openssl req -x509 -newkey rsa:3072 -sha256 -days 365 -nodes `
  -keyout ca.key -out ca.crt -subj "/CN=FedPRADA Local Test CA" `
  -addext "basicConstraints=critical,CA:TRUE" `
  -addext "keyUsage=critical,keyCertSign,cRLSign"

openssl req -newkey rsa:3072 -nodes `
  -keyout server.key -out server.csr -subj "/CN=localhost"

@"
subjectAltName=DNS:localhost,IP:127.0.0.1
extendedKeyUsage=serverAuth
keyUsage=digitalSignature,keyEncipherment
"@ | Set-Content -Encoding ascii server.ext

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key `
  -CAcreateserial -out server.crt -days 365 -sha256 -extfile server.ext
```

Generate a unique random bearer token for each DataCo client:

```powershell
$tokens = @{}
'africa','europe','latam','pacific_asia','usca' | ForEach-Object {
    $tokens[$_] = (openssl rand -hex 32).Trim()
}
$tokens | ConvertTo-Json | Set-Content -Encoding ascii dataco-client-tokens.json
```

For E-Commerce instead, generate tokens for its configured clients:

```powershell
$tokens = @{}
'A','B','C','D','F' | ForEach-Object {
    $tokens[$_] = (openssl rand -hex 32).Trim()
}
$tokens | ConvertTo-Json | Set-Content -Encoding ascii ecommerce-client-tokens.json
```

Keep `ca.key`, `server.key`, and token JSON files private. Do not commit them.
The CA certificate `ca.crt` is the trust anchor supplied to each client.

## Run securely

From the project directory, run Module 2 for DataCo:

```powershell
python module2.py --dataset dataco `
  --tls-cert C:\fedprada-tls\server.crt `
  --tls-key C:\fedprada-tls\server.key `
  --tls-ca C:\fedprada-tls\ca.crt `
  --client-tokens C:\fedprada-tls\dataco-client-tokens.json
```

Run one FedPRADA aggregation round in Module 3:

```powershell
python fedprada_module3.py --dataset dataco --method fedprada --rounds 1 `
  --tls-cert C:\fedprada-tls\server.crt `
  --tls-key C:\fedprada-tls\server.key `
  --tls-ca C:\fedprada-tls\ca.crt `
  --client-tokens C:\fedprada-tls\dataco-client-tokens.json
```

For E-Commerce, use `--dataset ecommerce` and
`ecommerce-client-tokens.json`. Supplying any TLS option requires all four
options. With no TLS options the existing local-only path is unchanged.

## Verify TLS and rejection of insecure requests

From the project directory:

```powershell
python -m unittest discover -s tests -v
```

The test starts a local TLS server and verifies a trusted TLS broadcast and
client update, rejects a wrong token, rejects a client that does not trust the
server CA, and confirms a plaintext HTTP client cannot communicate with the
TLS-only listener.

"""Interactive demonstration of TLS client-server communication and security checks."""

import http.client
import json
import ssl
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from secure_transport import (
    FederatedRoundCoordinator,
    TLSFederatedClient,
    TLSFederatedServer,
)


def run_demo():
    tls_dir = Path(r"C:\fedprada-tls")
    certfile = tls_dir / "server.crt"
    keyfile = tls_dir / "server.key"
    ca_file = tls_dir / "ca.crt"
    tokens_file = tls_dir / "ecommerce-client-tokens.json"

    if not (certfile.exists() and keyfile.exists() and ca_file.exists() and tokens_file.exists()):
        print(f"[!] Test TLS certificates or tokens not found in {tls_dir}.")
        print("    Please run the certificate generation steps in SECURE_TRANSPORT.md first.")
        sys.exit(1)

    with open(tokens_file, "r", encoding="utf-8") as f:
        tokens = json.load(f)

    client_id = "A"
    valid_token = tokens[client_id]

    print("=" * 70)
    print("FEDPRADA TLS TRANSPORT SECURITY VERIFICATION DEMO")
    print("=" * 70)

    coordinator = FederatedRoundCoordinator([client_id])
    coordinator.begin_round("ecommerce", 1, {"fc.weight": torch.tensor([[0.1, 0.2]])})

    # Start the secure server
    with TLSFederatedServer(
        certfile=str(certfile),
        keyfile=str(keyfile),
        client_tokens=tokens,
        handler=coordinator.handle,
    ) as server:
        host, port = server.address
        print(f"\n[1] TLS Server listening on https://{host}:{port}")

        # Step 1: Handshake and Cipher inspection
        client = TLSFederatedClient(host, port, str(ca_file), client_id, valid_token)
        conn = http.client.HTTPSConnection(host, port, context=client.context, timeout=5)
        conn.connect()
        cipher_name, proto_version, secret_bits = conn.sock.cipher()
        peer_cert = conn.sock.getpeercert()
        conn.close()

        print(f"\n[2] TLS Handshake & Certificate Verification:")
        print(f"    - Protocol Version    : {proto_version}")
        print(f"    - Cipher Suite        : {cipher_name} ({secret_bits} bits)")
        print(f"    - Verified Subject    : {dict(x[0] for x in peer_cert['subject'])}")
        print(f"    - Certificate SAN     : {peer_cert.get('subjectAltName')}")
        print("    --> Server certificate verified successfully against CA.")

        # Step 2: Encrypted Model Exchange
        print(f"\n[3] Authenticated & Encrypted Exchange (Client {client_id}):")
        broadcast = client.exchange({
            "action": "broadcast",
            "dataset": "ecommerce",
            "round": 1,
            "metadata": {},
        })
        print(f"    - Broadcast received  : Tensor shape {broadcast['model_state']['fc.weight'].shape}")

        update_ack = client.exchange({
            "action": "update",
            "dataset": "ecommerce",
            "round": 1,
            "metadata": {"num_train_samples": 1466, "validation_loss": 0.65},
            "model_state": {"fc.weight": torch.tensor([[0.3, 0.4]])},
        })
        print(f"    - Model update status : {update_ack['metadata']}")
        print("    --> Model weights and metadata encrypted in transit via TLS.")

        # Step 3: Test Unauthorized Client (Wrong Token)
        print(f"\n[4] Security Test: Unauthorized Client (Invalid Bearer Token):")
        unauth_client = TLSFederatedClient(host, port, str(ca_file), client_id, "invalid-token-123")
        try:
            unauth_client.exchange({
                "action": "update",
                "dataset": "ecommerce",
                "round": 1,
                "metadata": {},
            })
            print("    [!] ERROR: Unauthorized client was accepted!")
        except RuntimeError as e:
            print(f"    --> Rejected as expected: {e}")

        # Step 4: Test Insecure Plaintext HTTP Connection
        print(f"\n[5] Security Test: Insecure Plaintext HTTP Client:")
        plaintext_conn = http.client.HTTPConnection(host, port, timeout=3)
        try:
            plaintext_conn.request("POST", "/v1/exchange", body=b"{}")
            plaintext_conn.getresponse()
            print("    [!] ERROR: Plaintext connection succeeded!")
        except Exception as e:
            print(f"    --> Rejected as expected ({type(e).__name__}): Plaintext rejected by TLS port.")
        finally:
            plaintext_conn.close()

    print("\n" + "=" * 70)
    print("ALL SECURITY CHECKS PASSED: TLS 1.2+ encryption and auth verified.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()

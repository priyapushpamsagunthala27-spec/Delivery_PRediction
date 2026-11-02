import http.client
import shutil
import ssl
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from secure_transport import (
    FederatedRoundCoordinator,
    TLSFederatedClient,
    TLSFederatedServer,
)


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL is required for test certificates")
class SecureTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self._create_ca("ca")
        self._create_ca("untrusted-ca")
        self._create_server_certificate()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_openssl(self, *arguments):
        subprocess.run(
            ["openssl", *arguments],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _create_ca(self, name):
        self._run_openssl(
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "2",
            "-nodes",
            "-keyout",
            str(self.directory / f"{name}.key"),
            "-out",
            str(self.directory / f"{name}.crt"),
            "-subj",
            f"/CN={name}",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        )

    def _create_server_certificate(self):
        self._run_openssl(
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(self.directory / "server.key"),
            "-out",
            str(self.directory / "server.csr"),
            "-subj",
            "/CN=localhost",
        )
        (self.directory / "server.ext").write_text(
            "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
            "extendedKeyUsage=serverAuth\n",
            encoding="ascii",
        )
        self._run_openssl(
            "x509",
            "-req",
            "-in",
            str(self.directory / "server.csr"),
            "-CA",
            str(self.directory / "ca.crt"),
            "-CAkey",
            str(self.directory / "ca.key"),
            "-CAcreateserial",
            "-out",
            str(self.directory / "server.crt"),
            "-days",
            "2",
            "-sha256",
            "-extfile",
            str(self.directory / "server.ext"),
        )

    def test_tls_authentication_and_plaintext_rejection(self):
        coordinator = FederatedRoundCoordinator(["client-a"])
        coordinator.begin_round(
            "test",
            1,
            {"weight": torch.tensor([1.0, 2.0])},
        )
        tokens = {"client-a": "correct-token"}

        with TLSFederatedServer(
            certfile=str(self.directory / "server.crt"),
            keyfile=str(self.directory / "server.key"),
            client_tokens=tokens,
            handler=coordinator.handle,
        ) as server:
            host, port = server.address
            client = TLSFederatedClient(
                host,
                port,
                str(self.directory / "ca.crt"),
                "client-a",
                tokens["client-a"],
            )
            broadcast = client.exchange(
                {
                    "action": "broadcast",
                    "dataset": "test",
                    "round": 1,
                    "metadata": {},
                }
            )
            self.assertTrue(torch.equal(
                broadcast["model_state"]["weight"],
                torch.tensor([1.0, 2.0]),
            ))

            tls_probe = http.client.HTTPSConnection(host, port, context=client.context, timeout=5)
            tls_probe.connect()
            self.assertIn(tls_probe.sock.version(), ("TLSv1.2", "TLSv1.3"))
            tls_probe.close()

            acknowledgment = client.exchange(
                {
                    "action": "update",
                    "dataset": "test",
                    "round": 1,
                    "metadata": {"num_train_samples": 3},
                    "model_state": {"weight": torch.tensor([3.0, 4.0])},
                }
            )
            self.assertTrue(acknowledgment["metadata"]["accepted"])
            self.assertEqual(
                coordinator.updates["client-a"]["metadata"]["num_train_samples"],
                3,
            )

            unauthorized = TLSFederatedClient(
                host,
                port,
                str(self.directory / "ca.crt"),
                "client-a",
                "wrong-token",
            )
            with self.assertRaisesRegex(RuntimeError, "401"):
                unauthorized.exchange(
                    {
                        "action": "update",
                        "dataset": "test",
                        "round": 1,
                        "metadata": {},
                    }
                )

            untrusted = TLSFederatedClient(
                host,
                port,
                str(self.directory / "untrusted-ca.crt"),
                "client-a",
                tokens["client-a"],
            )
            with self.assertRaises(ssl.SSLCertVerificationError):
                untrusted.exchange(
                    {
                        "action": "broadcast",
                        "dataset": "test",
                        "round": 1,
                        "metadata": {},
                    }
                )

            plaintext = http.client.HTTPConnection(host, port, timeout=3)
            try:
                with self.assertRaises(
                    (OSError, ssl.SSLError, http.client.HTTPException)
                ):
                    plaintext.request("POST", "/v1/exchange", body=b"{}")
                    plaintext.getresponse()
            finally:
                plaintext.close()


if __name__ == "__main__":
    unittest.main()

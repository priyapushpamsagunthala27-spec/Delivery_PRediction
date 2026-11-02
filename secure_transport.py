"""TLS transport for federated model states and client metadata."""

import base64
import hmac
import http.client
import _ssl
import io
import json
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch


MAX_MESSAGE_BYTES = 512 * 1024 * 1024
TLS_MINIMUM = ssl.TLSVersion.TLSv1_2


def _set_minimum_tls_version(context):
    # Use the OpenSSL context descriptor directly so this also works when
    # another package wraps SSLContext and its Python property setter.
    native_context = getattr(context, "_ctx", context)
    _ssl._SSLContext.minimum_version.__set__(native_context, TLS_MINIMUM)


def load_client_tokens(path):
    with open(path, "r", encoding="utf-8") as token_file:
        tokens = json.load(token_file)
    if not isinstance(tokens, dict) or not tokens:
        raise ValueError("Client token file must contain a non-empty JSON object.")
    if any(
        not isinstance(client, str)
        or not client
        or not isinstance(token, str)
        or not token
        for client, token in tokens.items()
    ):
        raise ValueError("Every client ID and token must be a non-empty string.")
    return tokens


def add_tls_arguments(parser):
    parser.add_argument(
        "--tls-cert",
        help="TLS server certificate PEM; enables secure client-server transport.",
    )
    parser.add_argument("--tls-key", help="TLS server private key PEM.")
    parser.add_argument("--tls-ca", help="CA PEM used to verify the TLS server.")
    parser.add_argument(
        "--client-tokens",
        help="JSON file mapping authorized client IDs to bearer tokens.",
    )


def tls_config_from_args(args):
    values = {
        "certfile": args.tls_cert,
        "keyfile": args.tls_key,
        "ca_file": args.tls_ca,
        "tokens_file": args.client_tokens,
    }
    configured = [value is not None for value in values.values()]
    if not any(configured):
        return None
    if not all(configured):
        raise ValueError(
            "TLS requires all of --tls-cert, --tls-key, --tls-ca, "
            "and --client-tokens."
        )
    for option, path in values.items():
        if option != "tokens_file" and not Path(path).is_file():
            raise FileNotFoundError(f"TLS file not found: {path}")
    values["client_tokens"] = load_client_tokens(values.pop("tokens_file"))
    return values


def _encode_state(state):
    if state is None:
        return None
    buffer = io.BytesIO()
    torch.save(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in state.items()
        },
        buffer,
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_state(encoded):
    if encoded is None:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid model-state encoding.") from error
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("Model state must be a mapping of tensor names to tensors.")
    return state


def _encode_message(message):
    encoded = dict(message)
    encoded["model_state"] = _encode_state(message.get("model_state"))
    return json.dumps(encoded, separators=(",", ":")).encode("utf-8")


def _decode_message(body):
    message = json.loads(body.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("Request must be a JSON object.")
    message["model_state"] = _decode_state(message.get("model_state"))
    return message


class TLSFederatedClient:
    """TLS client with CA and hostname verification enabled."""

    def __init__(self, host, port, ca_file, client_id, token, timeout=30):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.token = token
        self.timeout = timeout
        self.context = ssl.create_default_context(cafile=ca_file)
        _set_minimum_tls_version(self.context)

    def exchange(self, message):
        body = _encode_message(message)
        if len(body) > MAX_MESSAGE_BYTES:
            raise ValueError("Federated message exceeds the configured size limit.")
        connection = http.client.HTTPSConnection(
            self.host,
            self.port,
            context=self.context,
            timeout=self.timeout,
        )
        try:
            connection.request(
                "POST",
                "/v1/exchange",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                    "X-Client-ID": self.client_id,
                    "Authorization": f"Bearer {self.token}",
                },
            )
            response = connection.getresponse()
            response_body = response.read(MAX_MESSAGE_BYTES + 1)
            if response.status != 200:
                detail = response_body.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Federated server rejected request "
                    f"({response.status}): {detail}"
                )
            if len(response_body) > MAX_MESSAGE_BYTES:
                raise ValueError("Federated response exceeds the configured size limit.")
            return _decode_message(response_body)
        finally:
            connection.close()


class TLSFederatedServer:
    """Threaded local HTTPS server that authenticates every client request."""

    def __init__(
        self,
        certfile,
        keyfile,
        client_tokens,
        handler,
        host="127.0.0.1",
        port=0,
    ):
        if not isinstance(client_tokens, dict) or not client_tokens:
            raise ValueError("At least one authorized client token is required.")
        self.client_tokens = dict(client_tokens)
        self.handler = handler

        server_context_type = next(
            (
                context_type
                for context_type in ssl.SSLContext.__mro__
                if context_type.__module__ == "ssl"
            ),
            ssl.SSLContext,
        )
        server_context = server_context_type(ssl.PROTOCOL_TLS_SERVER)
        _set_minimum_tls_version(server_context)
        server_context.load_cert_chain(certfile=certfile, keyfile=keyfile)

        outer = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/v1/exchange":
                    self.send_error(404)
                    return
                client_id = self.headers.get("X-Client-ID", "")
                authorization = self.headers.get("Authorization", "")
                supplied_token = authorization.removeprefix("Bearer ")
                expected_token = outer.client_tokens.get(client_id, "")
                if not expected_token or not hmac.compare_digest(
                    supplied_token, expected_token
                ):
                    self.send_error(401, "Client authentication failed.")
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self.send_error(400, "Invalid Content-Length.")
                    return
                if content_length < 0 or content_length > MAX_MESSAGE_BYTES:
                    self.send_error(413, "Federated message is too large.")
                    return
                try:
                    request = _decode_message(self.rfile.read(content_length))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    self.send_error(400, str(error))
                    return
                response = outer.handler(client_id, request)
                response_body = _encode_message(response)
                if len(response_body) > MAX_MESSAGE_BYTES:
                    self.send_error(413, "Federated response is too large.")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format_string, *args):
                return

        class FederatedHTTPServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                error = sys.exc_info()[1]
                if isinstance(error, ConnectionResetError):
                    return
                super().handle_error(request, client_address)

        self._httpd = FederatedHTTPServer((host, port), RequestHandler)
        self._httpd.daemon_threads = True
        self._httpd.socket = server_context.wrap_socket(
            self._httpd.socket,
            server_side=True,
        )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="fedprada-tls-server",
            daemon=True,
        )

    @property
    def address(self):
        return self._httpd.server_address

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join()


class FederatedRoundCoordinator:
    """Validates and stores round updates after the TLS server receives them."""

    def __init__(self, clients):
        self.clients = set(clients)
        self.begin_round(None, None, None)

    def begin_round(self, dataset, round_number, global_state):
        self.dataset = dataset
        self.round_number = round_number
        self.global_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in (global_state or {}).items()
        }
        self.updates = {}

    def handle(self, client_id, request):
        if client_id not in self.clients:
            raise ValueError("Client is not a participant in this run.")
        if request.get("dataset") != self.dataset:
            raise ValueError("Dataset does not match the active federated round.")
        if request.get("round") != self.round_number:
            raise ValueError("Round does not match the active federated round.")
        action = request.get("action")
        if action == "broadcast":
            return {
                "model_state": self.global_state,
                "metadata": {"dataset": self.dataset, "round": self.round_number},
            }
        if action != "update":
            raise ValueError("Unsupported federated action.")
        if client_id in self.updates:
            raise ValueError("Client has already submitted an update this round.")
        state = request.get("model_state")
        metadata = request.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Client metadata must be a JSON object.")
        if set(state or {}) != set(self.global_state):
            raise ValueError("Client model state keys do not match the global model.")
        for name, reference in self.global_state.items():
            value = state[name]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"Client model tensor is incompatible: {name}")
        self.updates[client_id] = {"model_state": state, "metadata": metadata}
        return {"metadata": {"accepted": True, "client_id": client_id}}

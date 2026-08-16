#!/usr/bin/env bash
# Generates throwaway EC P-256 CA + node cert + client cert for test suites.
# @author Olumuyiwa Oluwasanmi
#
# Usage: gen_test_certs.sh <out_dir>

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <out_dir>" >&2
    exit 1
fi

OUT_DIR="$1"

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl not found" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# Generate throwaway CA + node cert + client cert (EC P-256).
# Node cert MUST include subjectAltName for loopback dialing ([::1], 127.0.0.1, localhost).
openssl ecparam -name prime256v1 -genkey -noout -out "$OUT_DIR/ca.key" 2>/dev/null
openssl req -new -x509 -sha256 -key "$OUT_DIR/ca.key" -subj "/CN=SGEE-Test-CA" -days 1 -out "$OUT_DIR/ca.crt" 2>/dev/null

openssl ecparam -name prime256v1 -genkey -noout -out "$OUT_DIR/node.key" 2>/dev/null
openssl req -new -sha256 -key "$OUT_DIR/node.key" -subj "/CN=localhost" -out "$OUT_DIR/node.csr" 2>/dev/null
cat > "$OUT_DIR/node_ext.cnf" << 'EOF'
[v3_req]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = DNS:localhost,IP:127.0.0.1,IP:0:0:0:0:0:0:0:1
EOF
openssl x509 -req -in "$OUT_DIR/node.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" -CAcreateserial -out "$OUT_DIR/node.crt" -days 1 -sha256 -extfile "$OUT_DIR/node_ext.cnf" -extensions v3_req 2>/dev/null

openssl ecparam -name prime256v1 -genkey -noout -out "$OUT_DIR/client.key" 2>/dev/null
openssl req -new -sha256 -key "$OUT_DIR/client.key" -subj "/CN=sgee-client" -out "$OUT_DIR/client.csr" 2>/dev/null
cat > "$OUT_DIR/client_ext.cnf" << 'EOF'
[v3_req]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
EOF
openssl x509 -req -in "$OUT_DIR/client.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" -CAcreateserial -out "$OUT_DIR/client.crt" -days 1 -sha256 -extfile "$OUT_DIR/client_ext.cnf" -extensions v3_req 2>/dev/null

rm -f "$OUT_DIR/node.csr" "$OUT_DIR/client.csr" "$OUT_DIR/node_ext.cnf" "$OUT_DIR/client_ext.cnf" "$OUT_DIR/ca.srl"
exit 0

"""Use IPv4 for pip's HTTP connections when a server's IPv6 DNS stalls.

No TLS changes, no system resolver changes. Arguments are the normal pip args.
"""
import socket
from pip._vendor.urllib3.util import connection
connection.allowed_gai_family=lambda:socket.AF_INET
from pip._internal.cli.main import main
raise SystemExit(main())

#!/usr/bin/env bash
# Linux 安装为 systemd 服务。请在项目根目录执行，或改下面两个变量。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$(command -v python3)}"
UNIT=/etc/systemd/system/monet-guide.service

sed -e "s|^WorkingDirectory=.*|WorkingDirectory=${ROOT}|" \
    -e "s|^ExecStart=.*|ExecStart=${PY} ${ROOT}/server8.py|" \
    "${ROOT}/deploy/monet-guide.service" | sudo tee "${UNIT}" >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable monet-guide
sudo systemctl restart monet-guide
sudo systemctl status monet-guide --no-pager
echo
echo "常用命令:"
echo "  sudo systemctl status monet-guide"
echo "  sudo systemctl restart monet-guide"
echo "  journalctl -u monet-guide -f"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_FILE="${PROJECT_ROOT}/artifacts/automation/loto_crontab.generated"
BACKUP_DIR="${PROJECT_ROOT}/artifacts/automation/crontab_backups"
mkdir -p "$(dirname "${CRON_FILE}")" "${BACKUP_DIR}"

INSTALL=0
INSTALL_DASHBOARD=0
INSTALL_FEATURE=0

for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    --dashboard) INSTALL_DASHBOARD=1 ;;
    --feature-job) INSTALL_FEATURE=1 ;;
    --all) INSTALL_DASHBOARD=1; INSTALL_FEATURE=1 ;;
    --help|-h)
      cat <<EOF
Usage:
  ./scripts/install_wsl_automation.sh --all
  LOTO_ALLOW_AUTOMATION_INSTALL=1 ./scripts/install_wsl_automation.sh --install --all

Default is dry-run. It writes a generated crontab file but does not install it.
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "${INSTALL_DASHBOARD}" != "1" && "${INSTALL_FEATURE}" != "1" ]]; then
  INSTALL_DASHBOARD=1
  INSTALL_FEATURE=1
fi

{
  echo "# loto_forecast_project automation generated at $(date -Is)"
  echo "SHELL=/bin/bash"
  echo "LOTO_PROJECT_ROOT=${PROJECT_ROOT}"
  echo "UV_LINK_MODE=copy"
  if [[ "${INSTALL_DASHBOARD}" == "1" ]]; then
    echo "@reboot cd ${PROJECT_ROOT} && ./scripts/wsl_start_loto_app.sh"
  fi
  if [[ "${INSTALL_FEATURE}" == "1" ]]; then
    echo "15 3 * * * cd ${PROJECT_ROOT} && ./scripts/cron_run_feature_pipeline.sh"
  fi
} >"${CRON_FILE}"

echo "Generated crontab preview:"
cat "${CRON_FILE}"

if [[ "${INSTALL}" != "1" ]]; then
  echo
  echo "Dry-run only. To install:"
  echo "  LOTO_ALLOW_AUTOMATION_INSTALL=1 ./scripts/install_wsl_automation.sh --install --all"
  exit 0
fi

if [[ "${LOTO_ALLOW_AUTOMATION_INSTALL:-0}" != "1" ]]; then
  echo "Refusing to install crontab without LOTO_ALLOW_AUTOMATION_INSTALL=1" >&2
  exit 3
fi

backup="${BACKUP_DIR}/crontab_$(date +%Y%m%d_%H%M%S).txt"
crontab -l >"${backup}" 2>/dev/null || true
{
  cat "${backup}" 2>/dev/null || true
  echo
  cat "${CRON_FILE}"
} | crontab -

echo "Installed crontab. Backup: ${backup}"

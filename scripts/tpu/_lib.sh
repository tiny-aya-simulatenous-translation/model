# Shared bash helpers for the TPU launch scripts.
# Sourced (not executed) by setup_gcp.sh, launch_qr.sh, launch_canary.sh, ops.sh.

# Load KEY=VALUE pairs from a dotenv-style file WITHOUT overwriting variables
# that are already set in the environment. Skips blank / comment lines.
# Strips optional single or double quotes around the value.
#
# Precedence: shell env  >  .env  >  script defaults.
load_env_file() {
    local file=$1
    [ -f "$file" ] || return 0
    local line key value
    while IFS='' read -r line || [ -n "$line" ]; do
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            value="${value%\"}"; value="${value#\"}"
            value="${value%\'}"; value="${value#\'}"
            if [ -z "${!key:-}" ]; then
                export "$key=$value"
            fi
        fi
    done < "$file"
}

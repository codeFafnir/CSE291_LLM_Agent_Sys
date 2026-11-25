#!/usr/bin/env bash
# Control the EC2 experiment box. Usage: box.sh <start|stop|ip|ssh|sync|status>
# Credentials are sourced from $AWS_CREDS_FILE (required) and never printed.
set -euo pipefail

INSTANCE_ID=i-035e789cad43ca158
REGION=us-west-2
KEY_FILE="${KEY_FILE:-/Users/raghu/coding/CSE291_SystemsForLLMs&AI/raghu.pem}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR=/data/CSE291_LLM_Agent_Sys/agentic_pipeline/

usage() { echo "usage: $0 <start|stop|ip|ssh|sync|status>" >&2; exit 2; }

[ $# -ge 1 ] || usage
cmd="$1"
shift

: "${AWS_CREDS_FILE:?AWS_CREDS_FILE must point to a credentials file}"
set -a
# shellcheck disable=SC1090
source "$AWS_CREDS_FILE"
set +a

ec2() { aws ec2 "$@" --region "$REGION" --output text; }

state() {
  ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name'
}

public_ip() {
  local ip
  ip="$(ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress')"
  if [ -z "$ip" ] || [ "$ip" = "None" ]; then echo stopped; else echo "$ip"; fi
}

require_ip() {
  local ip
  ip="$(public_ip)"
  if [ "$ip" = "stopped" ]; then echo "instance $INSTANCE_ID has no public IP (not running)" >&2; exit 3; fi
  echo "$ip"
}

case "$cmd" in
  start)
    ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
    ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    public_ip
    ;;
  stop)
    ec2 stop-instances --instance-ids "$INSTANCE_ID" \
      --query 'StopInstances[0].CurrentState.Name'
    ;;
  ip)
    public_ip
    ;;
  status)
    echo "state: $(state)"
    echo "ip: $(public_ip)"
    my_ip="$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')"
    sg_id="$(ec2 describe-instances --instance-ids "$INSTANCE_ID" \
      --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId')"
    cidrs="$(ec2 describe-security-groups --group-ids "$sg_id" \
      --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp')"
    if echo "$cidrs" | tr '\t' '\n' | grep -qxF -e "$my_ip/32" -e "0.0.0.0/0"; then
      echo "ssh_allowed: yes ($my_ip)"
    else
      echo "ssh_allowed: no"
      echo "security_group: $sg_id"
      echo "current_ip: $my_ip"
      exit 4
    fi
    ;;
  ssh)
    ip="$(require_ip)"
    exec ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "ec2-user@$ip" "$@"
    ;;
  sync)
    ip="$(require_ip)"
    rsync -az --exclude __pycache__ --exclude .pytest_cache \
      -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
      "$REPO/" "ec2-user@$ip:$REMOTE_DIR"
    ;;
  *)
    usage
    ;;
esac

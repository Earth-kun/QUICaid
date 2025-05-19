#!/bin/bash

FIFO_PATH="/tmp/tshark_fifo"
FLAG_PATH="/tmp/label_flag.txt"
IP_SRC="10.10.3.10"
TIMEOUT=1

# STEP 1: rm tmp files and create new ones
echo "Cleaning temp files..."
[ -p "$FIFO_PATH" ] && rm "$FIFO_PATH"
[ -f "$FLAG_PATH" ] && rm "$FLAG_PATH"
mkfifo "$FIFO_PATH"
touch "$FLAG_PATH"

# STEP 2: PRE-TRAIN
python3 /home/user2/QUIC/live_program/parser.py --timeout $TIMEOUT --ipsrc "$IP_SRC" &
PARSER_PID=$!

echo "[*] Starting tshark..."
tshark -Y quic -i ens18 -T fields \
  -e frame.time_relative -e ip.src -e ip.dst -e udp.dstport \
  -e quic.version -e quic.packet_length > "$FIFO_PATH" &
TSHARK_PID=$!

echo 2 > /tmp/label_flag.txt
sleep 180

echo 3 > /tmp/label_flag.txt
# Run the attack
# /home/user2/QUIC-attacks/aioquic/quic-loris-CVE-2022-30591.sh &
sudo hping3 -p 443 --udp --flood 10.10.3.10 &
ATTACK_PID=$!
sleep 180

kill $ATTACK_PID
sleep 2
wait $ATTACK_PID 2>/dev/null

# STEP 3: ATTACK PHASE
echo 1 > /tmp/label_flag.txt
# Run the attack
# /home/user2/QUIC-attacks/aioquic/quic-loris-CVE-2022-30591.sh &
sudo hping3 -p 443 --udp --flood 10.10.3.10 &
ATTACK_PID=$!
sleep 180

kill $ATTACK_PID
sleep 2
wait $ATTACK_PID 2>/dev/null

echo 0 > "$FLAG_PATH"
# Optional: trigger benign traffic
# ./run_benign.sh
sleep 180


# Cleanup
echo "[*] Stopping attack parser and tshark..."
kill $PARSER_PID
kill $TSHARK_PID
sleep 1  
wait $PARSER_PID 2>/dev/null
wait $TSHARK_PID 2>/dev/null

echo "program finished..."

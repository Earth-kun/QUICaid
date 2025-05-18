#!/bin/bash

FIFO_PATH="/tmp/tshark_fifo"
IP_SRC="10.10.3.10"
TIMEOUT=1

# Helper function to create FIFO cleanly
function recreate_fifo {
    [ -p "$FIFO_PATH" ] && rm "$FIFO_PATH"
    mkfifo "$FIFO_PATH"
}

# STEP 1: Clean start
recreate_fifo

# STEP 2: BENIGN PHASE
echo "[*] Starting benign parser..."
python3 parser.py --timeout $TIMEOUT --label 0 --ipsrc "$IP_SRC" --output benign.csv &

PARSER_PID=$!
echo "[*] Starting tshark..."
tshark -f "udp port 443" -i ens18 -T fields \
  -e frame.time_relative -e ip.src -e ip.dst -e udp.dstport \
  -e quic.version -e quic.packet_length -e _ws.col.Protocol > "$FIFO_PATH" &

TSHARK_PID=$!

# Optional: trigger benign traffic
# ./run_benign.sh

sleep 180  # allow time for benign capture

# STEP 3: Clean shutdown of benign capture
echo "[*] Stopping benign parser..."
kill $PARSER_PID
wait $PARSER_PID 2>/dev/null

# STEP 4: Prepare for attack phase
echo "[*] Resetting FIFO to clear unread packets..."
kill $TSHARK_PID
wait $TSHARK_PID 2>/dev/null
recreate_fifo  # ensures no leftover packets in pipe

# STEP 5: ATTACK PHASE
echo "[*] Starting attack parser..."
python3 parser.py --timeout $TIMEOUT --label 1 --ipsrc "$IP_SRC" --output attack.csv &

PARSER_PID=$!

echo "[*] Restarting tshark for attack capture..."
tshark -f "udp port 443" -i ens18 -T fields \
  -e frame.time_relative -e ip.src -e ip.dst -e udp.dstport \
  -e quic.version -e quic.packet_length -e _ws.col.Protocol > "$FIFO_PATH" &

TSHARK_PID=$!

# Run the attack
./attack_script.sh

sleep 180

# Cleanup
echo "[*] Stopping attack parser and tshark..."
kill $PARSER_PID
kill $TSHARK_PID
sleep 1  
wait $PARSER_PID 2>/dev/null
wait $TSHARK_PID 2>/dev/null

echo "[+] Capture complete. Benign: benign.csv | Attack: attack.csv"

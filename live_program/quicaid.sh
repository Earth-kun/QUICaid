#!/bin/bash

FIFO_FILE="/tmp/tshark_fifo"

# Create FIFO if it does not exist
if [[ ! -p "$FIFO_FILE" ]]; then
    mkfifo "$FIFO_FILE"
fi

# Run tshark for UDP QUIC traffic and write output to FIFO in the background
tshark -Y quic -i ens18 -T fields \
    -e frame.time_relative -e ip.src -e ip.dst -e udp.dstport \
    -e quic.version -e quic.packet_length > "$FIFO_FILE" &

# Run process_packets.py with user-defined parameters
python3 process_packets.py --timeout "$1" --label "$2" --ipsrc "$3"

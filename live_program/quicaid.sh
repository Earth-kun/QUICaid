#!/bin/bash

FIFO_FILE="/tmp/tshark_fifo"

# Remove old FIFO file if it exists
if [[ -p "$FIFO_FILE" ]]; then
    rm "$FIFO_FILE"
fi

# Create a fresh FIFO
mkfifo "$FIFO_FILE"


# Run tshark for UDP QUIC traffic and write output to FIFO in the background
cap = "tshark -Y quic -i ens18 -T fields -e frame.time_relative -e ip.src -e ip.dst -e udp.dstport -e quic.version -e quic.packet_length> '$FIFO_FILE' &"

# Run process_packets.py with user-defined parameters
parser = "python3 parser.py --timeout '$1' --label '$2' --ipsrc '$3' --output '$4'"

parallel ::: "$cap" "$parser"
#!/bin/bash

FIFO_FILE="/tmp/tshark_fifo"

# Create FIFO if it does not exist
if [[ ! -p "$FIFO_FILE" ]]; then
    mkfifo "$FIFO_FILE"
fi

# Run tshark and write output to FIFO in the background
# tshark -T fields -e frame.number -e frame.time -e ip.src -e ip.dst -e frame.protocols -e frame.len -e _ws.col.Info > "$FIFO_FILE" &
tshark -f 'udp port 443' -i ens18 -a duration:300 -t r -T fields \
	-e frame.time_relative -e ip.src -e ip.dst -e udp.srcport -e udp.dstport -e quic.version \
	-e quic.packet_length -e _ws.col.Protocol

# Run process_packets.py with user-defined parameters
python3 process_packets.py --timeout "$1" --label "$2" --ipsrc "$3"

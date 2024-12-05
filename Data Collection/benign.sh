#!/bin/bash
#capture UPD packets (inlcuding QUIC) at port 443 for 5 minutes
#fields: DURATION, SRC_IP, DST_IP, SRC_PORT, DST_PORT, QUIC_VERSION, BYTES, PROTOCOL
tshark -f "udp port 443" -i ens18 -a duration:300 -t r -T fields \
-e frame.time -e ip.src -e ip.dst -e udp.srcport -e udp.dstport -e quic.version -e udp.payload -e _ws.col.Protocol \
-E separator=, -E quote=d > ./benign_flow/test1.csv

python3 benign_yt_traffic.py

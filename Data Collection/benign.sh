#!/bin/bash
#capture UPD packets (inlcuding QUIC) at port 443 for 5 minutes
#fields: SRC_IP, DST_IP, SRC_PORT, DST_PORT, QUIC_VERSION, BYTES, PROTOCOL
tshark -f "udp port 443" -i ens18 -a duration:300 -t r -T fields \
-e ip.src -e ip.dst -e ip.geoip.asnum -e udp.srcport -e udp.dstport -e quic.version -e frame.len -e _ws.col.Protocol \
-E separator=, -E quote=d > ./benign_flow/test1.csv

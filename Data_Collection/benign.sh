#!/bin/bash

command1="python3 fbvids.py"

for i in {1..1}; do
	parallel --delay 3 ::: "$command1" "tshark -f 'udp port 443' -i ens18 -a duration:300 -t r -T fields \
	-e frame.time_relative -e ip.src -e ip.dst -e udp.srcport -e udp.dstport -e quic.version \
	-e quic.packet_length -e _ws.col.Protocol \
	-E separator=, -E quote=d > ./benign_flow/out_test$i.csv"
done



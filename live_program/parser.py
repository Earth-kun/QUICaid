import os
import time
import pandas as pd
import argparse
import numpy as np
import statistics as stats
import pyasn
from threading import Thread, Event

# Argument Parsing
parser = argparse.ArgumentParser(description="Process packets from FIFO and batch them for analysis.")
parser.add_argument("--timeout", type=int, default=1, help="Timeout in seconds between batch processing.")
parser.add_argument("--label", type=str, default="2", help="Label for the flow analysis.")
parser.add_argument("--ipsrc", type=str, required=True, help="Source IP for flow processing.")
args = parser.parse_args()

FIFO_FILE = "/tmp/tshark_fifo"
BATCH_SIZE = 30
TIMEOUT = args.timeout
LABEL = args.label
IPSRC = args.ipsrc

# Initialize ASN lookup
asndb = pyasn.pyasn("ipasn_20140513.dat")

columns = ["No.", "Time", "Source", "Destination", "Protocol", "Length", "Info"]
packet_batch = []
stop_event = Event()

def get_asn(ip):
    """Lookup ASN for given IP."""
    try:
        return asndb.lookup(ip)[0]
    except:
        return 0  # Default ASN if lookup fails

def process_packet(row):
    """Extract features from a packet."""
    is_forward = (row['Source'] == IPSRC)
    
    if is_forward:
        port = row['Destination']
        asn = get_asn(row['Destination'])
    else:
        port = row['Source']
        asn = get_asn(row['Source'])
        
    return {
        'is_forward': is_forward,
        'bytes': int(row['Length']),
        'port': port,
        'asn': asn
    }

def process_batch():
    """Processes the collected batch using flow analysis."""
    global packet_batch
    if not packet_batch:
        return

    df = pd.DataFrame(packet_batch, columns=columns)
    
    # Extract flow features
    fwd_packets = [process_packet(row) for _, row in df.iterrows() if process_packet(row)['is_forward']]
    rev_packets = [process_packet(row) for _, row in df.iterrows() if not process_packet(row)['is_forward']]
    
    fwd_bytes = [p['bytes'] for p in fwd_packets]
    rev_bytes = [p['bytes'] for p in rev_packets]

    flow_features = {
        "fwd_bytes_max": max(fwd_bytes, default=0),
        "fwd_bytes_min": min(fwd_bytes, default=0),
        "fwd_bytes_avg": np.mean(fwd_bytes) if fwd_bytes else 0,
        "rev_bytes_max": max(rev_bytes, default=0),
        "rev_bytes_min": min(rev_bytes, default=0),
        "rev_bytes_avg": np.mean(rev_bytes) if rev_bytes else 0,
        "label": LABEL
    }

    # Print processed features (could also save to CSV or DB)
    print(flow_features)

    # Reset batch
    packet_batch = []

def read_fifo():
    """Reads packets from FIFO and batches them for processing."""
    global packet_batch
    last_process_time = time.time()

    with open(FIFO_FILE, "r") as fifo:
        for line in fifo:
            fields = line.strip().split("\t")
            if len(fields) == len(columns):
                packet_batch.append(dict(zip(columns, fields)))

            # Process batch if 30 packets are reached or timeout occurs
            if len(packet_batch) >= BATCH_SIZE or (time.time() - last_process_time) >= TIMEOUT:
                process_batch()
                last_process_time = time.time()

            if stop_event.is_set():
                break

if __name__ == "__main__":
    try:
        reader_thread = Thread(target=read_fifo, daemon=True)
        reader_thread.start()

        print("Processing packets... Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
        reader_thread.join()

# #!/bin/bash

# FIFO_FILE="/tmp/tshark_fifo"

# # Create FIFO if it does not exist
# if [[ ! -p "$FIFO_FILE" ]]; then
#     mkfifo "$FIFO_FILE"
# fi

# # Run tshark and write output to FIFO in the background
# tshark -T fields -e frame.number -e frame.time -e ip.src -e ip.dst -e frame.protocols -e frame.len -e _ws.col.Info > "$FIFO_FILE" &

# # Run process_packets.py with user-defined parameters
# python3 process_packets.py --timeout "$1" --label "$2" --ipsrc "$3"
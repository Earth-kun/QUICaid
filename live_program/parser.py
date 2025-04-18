import os
import time
import pandas as pd
import argparse
import numpy as np
import statistics as stats
import pyasn
from threading import Thread, Event
from collections import Counter

# Argument Parsing
parser = argparse.ArgumentParser(description="Process packets from FIFO and batch them for analysis.")
parser.add_argument("--timeout", type=int, default=1, help="Timeout in seconds between batch processing.")
parser.add_argument("--label", type=str, default="2", help="Label for the flow analysis.")
parser.add_argument("--ipsrc", type=str, required=True, help="Source IP for flow processing.")
parser.add_argument("--output", type=str, default=None, help="Optional output CSV file for raw packet data")
args = parser.parse_args()

FIFO_FILE = "/tmp/tshark_fifo"
BATCH_SIZE = 30
TIMEOUT = args.timeout
LABEL = args.label
IPSRC = args.ipsrc
OUTPUT_CSV=args.output

# Initialize ASN lookup
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
asndb = pyasn.pyasn("ipasn_20140513.dat")

columns = ["Duration", "SourceIP", "DestinationIP", "DestinationPort", "QUICVersion", "PacketLength"]
flow = []
stop_event = Event()

def process_flow_stats(arr_values, prefix=''):
    """Calculate statistics for an array of values"""
    if not arr_values:
        return {
            f"{prefix}max": 0,
            f"{prefix}min": 0,
            f"{prefix}ave": 0,
            f"{prefix}std": 0,
            f"{prefix}var": 0
        }
    
    arr = np.array(arr_values)
    return {
        f"{prefix}max": np.max(arr),
        f"{prefix}min": np.min(arr),
        f"{prefix}ave": np.mean(arr),
        f"{prefix}std": np.std(arr),
        f"{prefix}var": np.var(arr)
    }

def get_mode(arr):
    """Get mode from array, handling errors."""
    if not arr:
        return None
    try:
        return stats.mode(arr)
    except:
        filtered_arr = [x for x in arr if isinstance(x, (int, float))]
        return Counter(filtered_arr).most_common(1)[0][0]

def get_version_mode(arr):
    """Get QUIC version mode, handling different formats"""
    ver_mode = get_mode(arr)
    
    if isinstance(ver_mode, tuple):
        return max([int(h) for h in ver_mode])
    elif isinstance(ver_mode, int):
        return int(ver_mode)
    return 1  # Default value

def get_asn(ip):
    """Lookup ASN for given IP."""
    try:
        return asndb.lookup(ip)[0]
    except:
        try:
            false_ip = ip.split(",")[0]
            return asndb.lookup(false_ip)[0]
        except:
            return 0  # Default ASN if lookup fails


def process_packet(row, init_dur):
    """Extract features from a packet."""
    is_forward = (row['SourceIP'] == IPSRC)
    
    if is_forward:
        port = row['DestinationPort']
        asn = get_asn(row['DestinationIP'])
    else:
        port = row['DestinationPort']
        asn = get_asn(row['SourceIP'])
        
    return {
        'is_forward': is_forward,
        'bytes': row['PacketLength'],
        'iat': float(row['Duration']) - init_dur,
        'port': port,
        'asn': asn,
        'quic_version': row['QUICVersion']
    }

def calculate_flow_statistics(fwd_packets, rev_packets, ports, asns, versions, label):
    """Calculate statistical features for a complete flow"""
    # Get basic flow metrics
    dst_port = get_mode(ports)
    dst_asn = get_mode(asns)
    quic_ver = get_version_mode(versions)
    
    # Extract arrays for statistical calculations
    fwd_bytes = [int(p['bytes']) for p in fwd_packets]
    rev_bytes = [int(p['bytes']) for p in rev_packets]
    fwd_iat = [float(p['iat']) for p in fwd_packets if p['iat'] > 0]
    rev_iat = [float(p['iat']) for p in rev_packets if p['iat'] > 0]
    
    # Calculate duration and packet counts
    fwd_pkt = len(fwd_packets)
    rev_pkt = len(rev_packets)
    ratio = 1 if rev_pkt > fwd_pkt else 0
        
    # Calculate total values
    fwd_bytes_sum = sum(fwd_bytes)
    rev_bytes_sum = sum(rev_bytes)
    tot_pkt = fwd_pkt + rev_pkt
    tot_bytes = fwd_bytes_sum + rev_bytes_sum
    
    # Calculate durations
    fwd_dur = max([p['iat'] for p in fwd_packets]) if fwd_packets else 0
    rev_dur = max([p['iat'] for p in rev_packets]) if rev_packets else 0
    dur = max(fwd_dur, rev_dur)
    
    if dur > 0:
        flow_pkt = tot_pkt / dur
        flow_bytes = tot_bytes / dur
    else:
        flow_pkt = 0
        flow_bytes = 0
    
    # Calculate statistics
    combined_bytes_stats = process_flow_stats(fwd_bytes + rev_bytes)
    fwd_bytes_stats = process_flow_stats(fwd_bytes, 'fwd_')
    rev_bytes_stats = process_flow_stats(rev_bytes, 'rev_')
    
    combined_iat_stats = process_flow_stats(fwd_iat + rev_iat)
    fwd_iat_stats = process_flow_stats(fwd_iat, 'fwd_')
    rev_iat_stats = process_flow_stats(rev_iat, 'rev_')
    
    # Combine all features into a single list
    return [
        dst_port, dst_asn, quic_ver, dur, ratio, flow_pkt, flow_bytes, tot_pkt, tot_bytes,
        combined_bytes_stats['max'], combined_bytes_stats['min'], combined_bytes_stats['ave'], 
        combined_bytes_stats['std'], combined_bytes_stats['var'],
        fwd_pkt, fwd_bytes_sum,
        fwd_bytes_stats['fwd_max'], fwd_bytes_stats['fwd_min'], fwd_bytes_stats['fwd_ave'], 
        fwd_bytes_stats['fwd_std'], fwd_bytes_stats['fwd_var'],
        rev_pkt, rev_bytes_sum,
        rev_bytes_stats['rev_max'], rev_bytes_stats['rev_min'], rev_bytes_stats['rev_ave'], 
        rev_bytes_stats['rev_std'], rev_bytes_stats['rev_var'],
        combined_iat_stats['max'], combined_iat_stats['min'], combined_iat_stats['ave'], 
        combined_iat_stats['std'], combined_iat_stats['var'],
        fwd_dur,
        fwd_iat_stats['fwd_max'], fwd_iat_stats['fwd_min'], fwd_iat_stats['fwd_ave'], 
        fwd_iat_stats['fwd_std'], fwd_iat_stats['fwd_var'],
        rev_dur,
        rev_iat_stats['rev_max'], rev_iat_stats['rev_min'], rev_iat_stats['rev_ave'], 
        rev_iat_stats['rev_std'], rev_iat_stats['rev_var'],
        label
    ]

def process_flow():
    """Processes the collected batch using flow analysis."""
    global flow
    if not flow:
        return

    df = pd.DataFrame(flow, columns=columns)
    init_dur = float(df.at[0, 'Duration'])
    
    df["PacketLength"] = pd.to_numeric(df["PacketLength"], errors='coerce').fillna(0)

    # Extract flow features
    fwd_packets = [process_packet(row, init_dur) for _, row in df.iterrows() if process_packet(row, init_dur)['is_forward']]
    rev_packets = [process_packet(row, init_dur) for _, row in df.iterrows() if not process_packet(row, init_dur)['is_forward']]
    
    fwd_bytes = [p['bytes'] for p in fwd_packets]
    rev_bytes = [p['bytes'] for p in rev_packets]
    
    ports = [p['port'] for p in fwd_packets + rev_packets]
    asns = [p['asn'] for p in fwd_packets + rev_packets]
    versions = [p['quic_version'] for p in fwd_packets + rev_packets]

    # Calculate flow statistics
    flow_stats = calculate_flow_statistics(fwd_packets, rev_packets, ports, asns, versions, LABEL)
            
    flow_df = pd.DataFrame([flow_stats])
    flow_df.to_csv(OUTPUT_CSV, sep=',', header=False, index=False, mode='a')

    # Print processed features (could also save to CSV or DB)
    #print(f"Processed Flow: {flow_stats}\n")

    # Reset batch
    flow = []

def read_fifo():
    """Reads packets from FIFO and batches them for processing."""
    global flow
    last_process_time = time.time()

    with open(FIFO_FILE, "r") as fifo:
        for line in fifo:
            fields = line.strip().split("\t")
            if len(fields) == len(columns):
                flow.append(dict(zip(columns, fields)))

            # Process batch if 30 packets are reached or timeout occurs
            if len(flow) >= BATCH_SIZE or (time.time() - last_process_time) >= TIMEOUT:
                process_flow()
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

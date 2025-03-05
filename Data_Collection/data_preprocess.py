import os
# For data transformation
import pandas as pd            
# For statistical analysis
import numpy as np
import statistics as stats
# For ASN lookup
import pyasn
os.chdir(os.path.dirname(os.path.abspath(__file__)))
asndb = pyasn.pyasn('ipasn_20140513.dat')

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
    """Get mode from array, handling potential errors"""
    if not arr:
        return None
    try:
        return stats.mode(arr)
    except:
        # If stats.mode fails, return most common value
        from collections import Counter
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
    
def is_flow_boundary(index, df, ctr):
    """Determine if current packet represents a flow boundary"""
    if ctr == 29 or index == len(df) - 1:
        return True
    
    # Check for time gap greater than 1.0 second
    if index > 0 and abs(df.at[index, 'DURATION'] - df.at[index - 1, 'DURATION']) > 1.0:
        return True
    
    return False

def get_asn(ip):
    try:
        return asndb.lookup(ip)[0]
    except:
        false_ip = ip.split(",")[0]
        return asndb.lookup(false_ip)[0]

def process_packet(row, ipsrc, init_dur):
    """Process a packet and return relevant features"""
    is_forward = (row['SRC_IP'] == ipsrc)
    
    if is_forward:
        port = row['DST_PORT']
        asn = get_asn(row['DST_IP'])
    else:
        port = row['SRC_PORT']
        asn = get_asn(row['SRC_IP'])
        
    return {
        'is_forward': is_forward,
        'bytes': row['BYTES'],
        'iat': row['DURATION'] - init_dur,
        'port': port,
        'asn': asn,
        'ver': row['QUIC_VERSION']
    }

def process_flows(df, ipsrc, category="Streaming", label="0"):
    """Process network packets into flows with statistical features"""
    flows = []
    
    # Initialize variables for the first flow
    fwd_packets = []
    rev_packets = []
    ports = []
    asns = []
    versions = []
    ctr = 0
    init_dur = df.at[0, 'DURATION'] if len(df) > 0 else 0
    
    for index, row in df.iterrows():
        # Show progress
        if index % 1000 == 0:
            print(f"{index / len(df) * 100:.2f}%")
        
        # Check if we've reached the end of a flow
        if is_flow_boundary(index, df, ctr):
            # Process this last packet too
            packet = process_packet(row, ipsrc, init_dur)
            
            if packet['is_forward']:
                fwd_packets.append(packet)
            else:
                rev_packets.append(packet)
                
            ports.append(packet['port'])
            asns.append(packet['asn'])
            versions.append(packet['ver'])
            
            # Calculate flow statistics
            flow_stats = calculate_flow_statistics(fwd_packets, rev_packets, ports, asns, versions, label)
            flows.append(flow_stats)
            
            # Reset for next flow
            fwd_packets = []
            rev_packets = []
            ports = []
            asns = []
            versions = []
            ctr = 0
        else:
            # Process regular packet in the middle of a flow
            if ctr == 0:
                init_dur = row['DURATION']
                
            packet = process_packet(row, ipsrc, init_dur)
            
            if packet['is_forward']:
                fwd_packets.append(packet)
            else:
                rev_packets.append(packet)
                
            ports.append(packet['port'])
            asns.append(packet['asn'])
            versions.append(packet['ver'])
            ctr += 1
    
    return flows

def calculate_flow_statistics(fwd_packets, rev_packets, ports, asns, versions, label):
    """Calculate statistical features for a complete flow"""
    # Get basic flow metrics
    dst_port = get_mode(ports)
    dst_asn = get_mode(asns)
    quic_ver = get_version_mode(versions)
    
    # Extract arrays for statistical calculations
    fwd_bytes = [p['bytes'] for p in fwd_packets]
    rev_bytes = [p['bytes'] for p in rev_packets]
    fwd_iat = [p['iat'] for p in fwd_packets if p['iat'] > 0]
    rev_iat = [p['iat'] for p in rev_packets if p['iat'] > 0]
    
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

# input csv
input_file = "./benign_flow/yt_test/test10.csv"
df = pd.read_csv(input_file)

column_names = ["DURATION", "SRC_IP", "DST_IP", "SRC_PORT", "DST_PORT", "QUIC_VERSION", "BYTES", "PROTOCOL"]
df.columns = column_names

# delete protocol column
df = df.drop("PROTOCOL", axis=1)

df["BYTES"] = pd.to_numeric(df["BYTES"], errors='coerce').fillna(0)

# Input parameters
ipsrc = "10.10.3.10"
label = "0"

# TODO: Fix dst_asn, flow_pkt_rate, flow_byte_rate, min_bytes, ave_bytes, std_bytes, var_bytes, max_fwd_bytes, min_fwd_bytes, avg_fwd_bytes, std_fwd_bytes, var_fwd_bytes, max_iat, min_iat, avg_iat, std_iat, var_iat, fwd_duration, max_fwd_iat, min_fwd_iat, avg_fwd_iat, std_fwd_iat, var_fwd_iat, rev_duration, max_rev_iat, min_rev_iat, avg_rev_iat, std_rev_iat, var_rev_iat

# Process the flows
flows = process_flows(df, ipsrc, label)

# Save the flows to CSV
flow_df = pd.DataFrame(flows)
column_names = [
    "dst_port", "dst_asn", "quic_ver", "dur", "ratio", 
    "flow_pkt_rate", "flow_byte_rate", "total_pkts", "total_bytes",
    "max_bytes", "min_bytes", "ave_bytes", "std_bytes", "var_bytes",
    "fwd_pkts", "fwd_bytes", 
    "max_fwd_bytes", "min_fwd_bytes", "ave_fwd_bytes", "std_fwd_bytes", "var_fwd_bytes",
    "rev_pkts", "rev_bytes",
    "max_rev_bytes", "min_rev_bytes", "ave_rev_bytes", "std_rev_bytes", "var_rev_bytes",
    "max_iat", "min_iat", "ave_iat", "std_iat", "var_iat",
    "fwd_dur",
    "max_fwd_iat", "min_fwd_iat", "ave_fwd_iat", "std_fwd_iat", "var_fwd_iat",
    "rev_dur",
    "max_rev_iat", "min_rev_iat", "ave_rev_iat", "std_rev_iat", "var_rev_iat",
    "label"
]

flow_df.columns = column_names


# Append to existing file or create new one
output_file = "./benign_flow/benign1.csv"
try:
    # existing_df = pd.read_csv(output_file)
    # combined_df = pd.concat([existing_df, flow_df], ignore_index=True)
    # combined_df.to_csv(output_file, index=False)
    flow_df.to_csv(output_file, index = False)
except FileNotFoundError:
    flow_df.to_csv(output_file, index=False)

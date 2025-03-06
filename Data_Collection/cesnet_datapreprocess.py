import os
import time
import json
import psutil
import pandas as pd
import numpy as np
import ast
from functools import wraps
from datetime import datetime

# Initialize profiling variables
start_time = time.time()
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

# Profiling decorator
def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Time profiling
        func_start_time = time.time()
        
        # Memory profiling
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Calculate metrics
        func_duration = time.time() - func_start_time
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        print(f"Function {func.__name__}:")
        print(f"  Time: {func_duration:.4f} seconds")
        print(f"  Memory: {mem_used:.2f} MB ({mem_before:.2f} MB -> {mem_after:.2f} MB)")
        
        return result
    return wrapper

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

def parse_ppi_data(ppi_str):
    """
    Parse the PPI string containing inter-packet times, directions, and sizes.
    Returns a dictionary with parsed data.
    """
    try:
        # Convert string representation of list to actual list
        if isinstance(ppi_str, str):
            ppi_data = ast.literal_eval(ppi_str)
        else:
            ppi_data = ppi_str
            
        # Extract components - PPI contains [ipt_list, direction_list, size_list]
        if len(ppi_data) >= 3:
            ipt_list = ppi_data[0]
            direction_list = ppi_data[1]
            size_list = ppi_data[2]
            
            return {
                'ipt': ipt_list,
                'direction': direction_list,
                'size': size_list
            }
        else:
            print(f"Warning: PPI data has invalid format: {ppi_str[:100]}")
            return {'ipt': [], 'direction': [], 'size': []}
    except Exception as e:
        print(f"Error parsing PPI data: {e}")
        print(f"Data sample: {str(ppi_str)[:100]}")
        return {'ipt': [], 'direction': [], 'size': []}

def extract_stats_from_histogram(hist, bin_ranges):
    """
    Extract statistics from a histogram
    hist: List of counts for each bin
    bin_ranges: List of tuples (min, max) for each bin
    """
    if not hist or sum(hist) == 0:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "std": 0,
            "var": 0,
            "dur": 0
        }
    
    # Calculate the center of each bin for mean estimation
    bin_centers = [(min_val + max_val) / 2 for min_val, max_val in bin_ranges]
    
    # Special case for the last bin (>1024)
    bin_centers[-1] = 1.5 * bin_ranges[-1][0]  # 1.5 * 1024 as an approximation
    
    # Total packet count
    total_packets = sum(hist)
    
    # Min: the lower bound of the first non-zero bin
    for i, count in enumerate(hist):
        if count > 0:
            min_val = bin_ranges[i][0]
            break
    else:
        min_val = 0
    
    # Max: the upper bound of the last non-zero bin
    for i in range(len(hist)-1, -1, -1):
        if hist[i] > 0:
            if i == len(bin_ranges) - 1:  # Last bin (>1024)
                max_val = 2 * bin_ranges[i][0]  # Estimate
            else:
                max_val = bin_ranges[i][1]
            break
    else:
        max_val = 0
    
    # Mean: weighted average of bin centers
    mean = sum(center * count for center, count in zip(bin_centers, hist)) / total_packets if total_packets > 0 else 0
    
    # Variance and std: approximation using bin centers
    var = sum((center - mean) ** 2 * count for center, count in zip(bin_centers, hist)) / total_packets if total_packets > 0 else 0
    std = np.sqrt(var)
    
    # Duration: rough estimate based on packet count and mean IAT
    dur = mean * total_packets if mean > 0 else 0
    
    return {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "std": std,
        "var": var,
        "dur": max_val  # Use max as a conservative estimate for duration
    }

def calculate_flow_statistics(flow_row, label="0"):
    """
    Calculate all flow statistics from a preprocessed flow row.
    """
    # Parse PPI data
    ppi_data = parse_ppi_data(flow_row['PPI'])
    
    # Extract packet information
    packet_sizes = ppi_data['size']
    packet_directions = ppi_data['direction']
    packet_ipt = ppi_data['ipt']
    
    # Separate forward and reverse packets
    fwd_indices = [i for i, direction in enumerate(packet_directions) if direction == 1]
    rev_indices = [i for i, direction in enumerate(packet_directions) if direction == -1]
    
    # Extract sizes
    fwd_bytes = [packet_sizes[i] for i in fwd_indices]
    rev_bytes = [packet_sizes[i] for i in rev_indices]
    
    # Extract inter-packet times (skip first packet as it has no previous packet)
    if len(packet_ipt) > 0:
        packet_ipt = packet_ipt[1:]  # First packet typically has 0 IPT
    
    # Use existing IPT or calculate from packet directions and times
    if packet_ipt and len(packet_ipt) > 0:
        # Match IPT with packet directions (IPT list might be one element shorter)
        fwd_iat = []
        rev_iat = []
        
        for i, ipt in enumerate(packet_ipt):
            if i < len(packet_directions) - 1:  # Ensure we don't go out of bounds
                if packet_directions[i+1] == 1:  # Forward packet
                    fwd_iat.append(ipt)
                elif packet_directions[i+1] == -1:  # Reverse packet
                    rev_iat.append(ipt)
    else:
        # Use provided duration metrics if IPT not available
        fwd_iat = []
        rev_iat = []
    
    # Calculate basic flow metrics
    dst_port = int(flow_row['DST_PORT'])
    dst_asn = int(flow_row['DST_ASN']) if pd.notna(flow_row['DST_ASN']) else 0
    quic_ver = int(flow_row['QUIC_VERSION']) if pd.notna(flow_row['QUIC_VERSION']) and flow_row['QUIC_VERSION'] != '' else 1
    
    # Calculate duration and packet counts
    fwd_pkt = int(flow_row['PACKETS']) if pd.notna(flow_row['PACKETS']) else len(fwd_bytes)
    rev_pkt = int(flow_row['PACKETS_REV']) if pd.notna(flow_row['PACKETS_REV']) else len(rev_bytes)
    ratio = 1 if rev_pkt > fwd_pkt else 0
    
    # Calculate total values
    fwd_bytes_sum = int(flow_row['BYTES']) if pd.notna(flow_row['BYTES']) else sum(fwd_bytes)
    rev_bytes_sum = int(flow_row['BYTES_REV']) if pd.notna(flow_row['BYTES_REV']) else sum(rev_bytes)
    tot_pkt = fwd_pkt + rev_pkt
    tot_bytes = fwd_bytes_sum + rev_bytes_sum
    
    # Get duration values
    dur = float(flow_row['DURATION']) if pd.notna(flow_row['DURATION']) else 0
    
    ipt_bin_ranges = [
    (0, 15),
    (16, 31),
    (32, 63),
    (64, 127),
    (128, 255),
    (256, 511),
    (512, 1023),
    (1024, float('inf'))  # >1024
    ]

    try:
      if pd.notna(flow_row.get('PHIST_SRC_IPT')) and flow_row['PHIST_SRC_IPT']:
          # Convert histogram string to actual list
          src_ipt_hist = ast.literal_eval(flow_row['PHIST_SRC_IPT'])
          
          # Check if it's a histogram (8 values) or raw IAT list
          if len(src_ipt_hist) == 8:
              # It's a histogram
              src_stats = extract_stats_from_histogram(src_ipt_hist, ipt_bin_ranges)
              # Convert from ms to seconds for consistency with other values
              fwd_dur = src_stats["dur"] / 1000
              # Generate synthetic IAT values for statistics if needed
              if len(fwd_iat) == 0:
                  # Create representative IATs based on histogram
                  fwd_iat = []
                  for bin_idx, count in enumerate(src_ipt_hist):
                      min_val, max_val = ipt_bin_ranges[bin_idx]
                      bin_center = (min_val + max_val) / 2
                      fwd_iat.extend([bin_center] * count)
                  # Convert from ms to seconds
                  fwd_iat = [x/1000 for x in fwd_iat]
          else:
              # It's a raw list of IATs
              fwd_iat = src_ipt_hist if len(fwd_iat) == 0 else fwd_iat
              fwd_dur = max(src_ipt_hist) if src_ipt_hist else 0
      else:
          fwd_dur = dur * (fwd_pkt / tot_pkt) if tot_pkt > 0 else 0
          
      # Same approach for DST_IPT
      if pd.notna(flow_row.get('PHIST_DST_IPT')) and flow_row['PHIST_DST_IPT']:
          dst_ipt_hist = ast.literal_eval(flow_row['PHIST_DST_IPT'])
          
          if len(dst_ipt_hist) == 8:
              dst_stats = extract_stats_from_histogram(dst_ipt_hist, ipt_bin_ranges)
              rev_dur = dst_stats["dur"] / 1000
              if len(rev_iat) == 0:
                  rev_iat = []
                  for bin_idx, count in enumerate(dst_ipt_hist):
                      min_val, max_val = ipt_bin_ranges[bin_idx]
                      bin_center = (min_val + max_val) / 2
                      rev_iat.extend([bin_center] * count)
                  rev_iat = [x/1000 for x in rev_iat]
          else:
              rev_iat = dst_ipt_hist if len(rev_iat) == 0 else rev_iat
              rev_dur = max(dst_ipt_hist) if dst_ipt_hist else 0
      else:
          rev_dur = dur * (rev_pkt / tot_pkt) if tot_pkt > 0 else 0
    except Exception as e:
      print(f"Error parsing PHIST data: {e}")
      fwd_dur = dur * (fwd_pkt / tot_pkt) if tot_pkt > 0 else 0
      rev_dur = dur * (rev_pkt / tot_pkt) if tot_pkt > 0 else 0


    # Calculate rates
    if dur > 0:
        flow_pkt = tot_pkt / dur
        flow_bytes = tot_bytes / dur
    else:
        flow_pkt = 0
        flow_bytes = 0
    
    # Calculate size statistics
    combined_bytes_stats = process_flow_stats(fwd_bytes + rev_bytes)
    fwd_bytes_stats = process_flow_stats(fwd_bytes, 'fwd_')
    rev_bytes_stats = process_flow_stats(rev_bytes, 'rev_')
    
    # Calculate inter-arrival time statistics
    combined_iat_stats = process_flow_stats(fwd_iat + rev_iat)
    fwd_iat_stats = process_flow_stats(fwd_iat, 'fwd_')
    rev_iat_stats = process_flow_stats(rev_iat, 'rev_')
    
    # Create and return feature dictionary
    flow_features = {
        "dst_port": dst_port,
        "dst_asn": dst_asn,
        "quic_ver": quic_ver,
        "dur": dur,
        "ratio": ratio,
        "flow_pkt_rate": flow_pkt,
        "flow_byte_rate": flow_bytes,
        "total_pkts": tot_pkt,
        "total_bytes": tot_bytes,
        "max_bytes": combined_bytes_stats['max'],
        "min_bytes": combined_bytes_stats['min'],
        "ave_bytes": combined_bytes_stats['ave'],
        "std_bytes": combined_bytes_stats['std'],
        "var_bytes": combined_bytes_stats['var'],
        "fwd_pkts": fwd_pkt,
        "fwd_bytes": fwd_bytes_sum,
        "max_fwd_bytes": fwd_bytes_stats['fwd_max'],
        "min_fwd_bytes": fwd_bytes_stats['fwd_min'],
        "ave_fwd_bytes": fwd_bytes_stats['fwd_ave'],
        "std_fwd_bytes": fwd_bytes_stats['fwd_std'],
        "var_fwd_bytes": fwd_bytes_stats['fwd_var'],
        "rev_pkts": rev_pkt,
        "rev_bytes": rev_bytes_sum,
        "max_rev_bytes": rev_bytes_stats['rev_max'],
        "min_rev_bytes": rev_bytes_stats['rev_min'],
        "ave_rev_bytes": rev_bytes_stats['rev_ave'],
        "std_rev_bytes": rev_bytes_stats['rev_std'],
        "var_rev_bytes": rev_bytes_stats['rev_var'],
        "max_iat": combined_iat_stats['max'],
        "min_iat": combined_iat_stats['min'],
        "ave_iat": combined_iat_stats['ave'],
        "std_iat": combined_iat_stats['std'],
        "var_iat": combined_iat_stats['var'],
        "fwd_dur": fwd_dur,
        "max_fwd_iat": fwd_iat_stats['fwd_max'],
        "min_fwd_iat": fwd_iat_stats['fwd_min'], 
        "ave_fwd_iat": fwd_iat_stats['fwd_ave'],
        "std_fwd_iat": fwd_iat_stats['fwd_std'],
        "var_fwd_iat": fwd_iat_stats['fwd_var'],
        "rev_dur": rev_dur,
        "max_rev_iat": rev_iat_stats['rev_max'],
        "min_rev_iat": rev_iat_stats['rev_min'],
        "ave_rev_iat": rev_iat_stats['rev_ave'],
        "std_rev_iat": rev_iat_stats['rev_std'],
        "var_rev_iat": rev_iat_stats['rev_var'],
        "label": label
    }
    
    return flow_features

@profile_function
def process_flows_df(df, label="0"):
    """Process all flows in the DataFrame"""
    all_flow_features = []
    
    for idx, flow_row in df.iterrows():
        # Show progress
        if idx % 100 == 0:
            print(f"Processing flow {idx}/{len(df)} ({idx / len(df) * 100:.2f}%)")
        
        try:
            # Determine label from category or APP if available
            flow_label = label
            if pd.notna(flow_row.get('CATEGORY')):
                category = str(flow_row['CATEGORY']).lower()
                if 'malicious' in category:
                    flow_label = "1"  # Malicious
            
            # Calculate statistics for this flow
            flow_features = calculate_flow_statistics(flow_row, flow_label)
            all_flow_features.append(flow_features)
        except Exception as e:
            print(f"Error processing flow {idx}: {e}")
    
    return all_flow_features

@profile_function
def main():
    """Main function to process flows CSV"""
    global start_time
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process flows CSV into statistical features')
    parser.add_argument('--input', '-i', default='./input_flows.csv', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='./output_features.csv', help='Output CSV file path')
    parser.add_argument('--label', '-l', default='0', help='Default label for flows (0=benign, 1=malicious)')
    parser.add_argument('--batch', '-b', type=int, default=10000, help='Batch size for processing')
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    default_label = args.label
    batch_size = args.batch
    
    print(f"Reading CSV file: {input_file}")
    read_start = time.time()
    
    # Check CSV file size
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Read in batches if file is large
    if file_size_mb > 100:  # For files larger than 100MB
        print(f"Large file detected, reading in batches of {batch_size} rows")
        all_flow_features = []
        
        for chunk in pd.read_csv(input_file, chunksize=batch_size):
            # Process this batch
            print(f"Processing batch of {len(chunk)} rows")
            batch_features = process_flows_df(chunk, default_label)
            all_flow_features.extend(batch_features)
            
            # Create intermediate DataFrame and memory cleanup
            print("Saving intermediate results...")
            batch_df = pd.DataFrame(batch_features)
            batch_df.to_csv(f"{output_file}.part", mode='a', header=not os.path.exists(f"{output_file}.part"), index=False)
            del batch_features
            del batch_df
        
        # Load the partial file
        print("Loading all processed results...")
        flow_df = pd.read_csv(f"{output_file}.part")
        
        # Remove partial file
        os.remove(f"{output_file}.part")
    else:
        # For smaller files, process all at once
        df = pd.read_csv(input_file)
        read_time = time.time() - read_start
        print(f"CSV read time: {read_time:.2f} seconds")
        print(f"DataFrame shape: {df.shape}")
        
        # Process flows
        print("Processing flows...")
        all_flow_features = process_flows_df(df, default_label)
        flow_df = pd.DataFrame(all_flow_features)
    
    # Ensure column order matches original
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
    
    # Reorder columns if they exist
    existing_columns = flow_df.columns.tolist()
    ordered_columns = [col for col in column_names if col in existing_columns]
    flow_df = flow_df[ordered_columns]
    
    # Save the flows to CSV
    print(f"Saving to {output_file}")
    write_start = time.time()
    flow_df.to_csv(output_file, index=False)
    write_time = time.time() - write_start
    print(f"CSV write time: {write_time:.2f} seconds")
    
    # Final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024
    
    # Print profiling summary
    total_time = time.time() - start_time
    
    print("\n===== PROFILING SUMMARY =====")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
    print(f"Input flows: {len(df) if 'df' in locals() else 'N/A (batch mode)'}")
    print(f"Output features: {len(flow_df)}")
    print(f"Flows processed per second: {len(flow_df)/total_time:.2f}")
    print(f"CSV writing: {write_time:.2f} seconds ({write_time/total_time*100:.1f}%)")
    print("============================")
    
    return flow_df

if __name__ == "__main__":
    import cProfile
    import pstats
    
    # Use cProfile for detailed profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the main function
    main()
    
    # Disable profiler and print stats
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
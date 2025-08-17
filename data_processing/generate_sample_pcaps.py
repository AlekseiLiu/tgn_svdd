#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Sample PCAP Files for CIC-IDS2017 Dataset Testing

This script creates sample PCAP files that mimic the structure of the CIC-IDS2017 dataset
for testing the TGN-SVDD data processing pipeline.

The generated files will include:
- Monday-WorkingHours.pcap (normal traffic baseline)
- Tuesday-WorkingHours.pcap (SSH-Patator/FTP-Patator attacks)
- Wednesday-WorkingHours.pcap (DoS attacks, Heartbleed)
- Thursday-WorkingHours.pcap (Web attacks, Infiltration)
- Friday-WorkingHours.pcap (Botnet, PortScan, DDoS)
"""

import os
import sys
from datetime import datetime, timedelta
import random
import time

try:
    from scapy.all import *
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.l2 import Ether
    from scapy.layers.dns import DNS, DNSQR
    from scapy.layers.http import HTTP
except ImportError:
    print("ERROR: Scapy is required to generate PCAP files.")
    print("Please install it with: pip install scapy")
    sys.exit(1)


def generate_normal_traffic(count=1000, start_time=None):
    """Generate normal network traffic packets."""
    if start_time is None:
        start_time = time.time()
    
    packets = []
    
    # Common IP ranges for internal network
    internal_ips = [
        "192.168.10.5", "192.168.10.8", "192.168.10.9", 
        "192.168.10.14", "192.168.10.15", "192.168.10.50"
    ]
    external_ips = [
        "8.8.8.8", "1.1.1.1", "172.217.12.142", "151.101.193.140"
    ]
    
    for i in range(count):
        # Random timing
        packet_time = start_time + random.uniform(0, 3600)  # Within 1 hour
        
        # Generate different types of normal traffic
        traffic_type = random.choice(['http', 'dns', 'ssh', 'icmp'])
        
        src_ip = random.choice(internal_ips)
        
        if traffic_type == 'http':
            dst_ip = random.choice(external_ips)
            packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=random.randint(1024, 65535), dport=80)
        elif traffic_type == 'dns':
            dst_ip = random.choice(external_ips)
            packet = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=random.randint(1024, 65535), dport=53) / DNS(qd=DNSQR(qname="example.com"))
        elif traffic_type == 'ssh':
            dst_ip = random.choice(external_ips)
            packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=random.randint(1024, 65535), dport=22)
        else:  # icmp
            dst_ip = random.choice(external_ips)
            packet = Ether() / IP(src=src_ip, dst=dst_ip) / ICMP()
        
        packet.time = packet_time
        packets.append(packet)
    
    return sorted(packets, key=lambda x: x.time)


def generate_attack_traffic(attack_type, count=100, start_time=None):
    """Generate attack traffic based on CIC-IDS2017 attack patterns."""
    if start_time is None:
        start_time = time.time()
    
    packets = []
    
    if attack_type == "SSH-Patator":
        # SSH brute force attack from 172.16.0.1 to 192.168.10.50
        attacker_ip = "172.16.0.1"
        target_ip = "192.168.10.50"
        
        for i in range(count):
            packet_time = start_time + i * 0.1  # Rapid attacks
            packet = Ether() / IP(src=attacker_ip, dst=target_ip) / TCP(sport=random.randint(1024, 65535), dport=22, flags="S")
            packet.time = packet_time
            packets.append(packet)
    
    elif attack_type == "DoS_slowloris":
        # Slowloris DoS attack
        attacker_ip = "172.16.0.1"
        target_ip = "192.168.10.50"
        
        for i in range(count):
            packet_time = start_time + i * 0.05
            packet = Ether() / IP(src=attacker_ip, dst=target_ip) / TCP(sport=random.randint(1024, 65535), dport=80, flags="S")
            packet.time = packet_time
            packets.append(packet)
    
    elif attack_type == "Web_Attack_XSS":
        # Web-based XSS attack
        attacker_ip = "172.16.0.1"
        target_ip = "192.168.10.50"
        
        for i in range(count):
            packet_time = start_time + i * 0.2
            packet = Ether() / IP(src=attacker_ip, dst=target_ip) / TCP(sport=random.randint(1024, 65535), dport=80) / Raw(load=b"GET /?q=<script>alert('XSS')</script> HTTP/1.1\r\n\r\n")
            packet.time = packet_time
            packets.append(packet)
    
    elif attack_type == "Bot":
        # Botnet traffic
        bot_ips = ["192.168.10.5", "192.168.10.8", "192.168.10.9", "192.168.10.14", "192.168.10.15"]
        c2_ip = "205.174.165.73"
        
        for i in range(count):
            packet_time = start_time + i * 0.3
            src_ip = random.choice(bot_ips)
            packet = Ether() / IP(src=src_ip, dst=c2_ip) / TCP(sport=random.randint(1024, 65535), dport=8080)
            packet.time = packet_time
            packets.append(packet)
    
    elif attack_type == "PortScan":
        # Port scanning attack
        attacker_ip = "172.16.0.1"
        target_ip = "192.168.10.50"
        
        for port in range(1, count + 1):
            if port > 65535:
                break
            packet_time = start_time + port * 0.01
            packet = Ether() / IP(src=attacker_ip, dst=target_ip) / TCP(sport=random.randint(1024, 65535), dport=port, flags="S")
            packet.time = packet_time
            packets.append(packet)
    
    return packets


def generate_cic_2017_day(day_name, output_dir, packet_count=5000):
    """Generate a complete day's worth of traffic for CIC-2017 dataset."""
    print(f"Generating {day_name}...")
    
    # Base time for each day (different days)
    base_times = {
        "Monday": datetime(2017, 7, 3, 8, 0, 0),
        "Tuesday": datetime(2017, 7, 4, 8, 0, 0),
        "Wednesday": datetime(2017, 7, 5, 8, 0, 0),
        "Thursday": datetime(2017, 7, 6, 8, 0, 0),
        "Friday": datetime(2017, 7, 7, 8, 0, 0)
    }
    
    day_key = day_name.split('-')[0]
    start_time = base_times[day_key].timestamp()
    
    all_packets = []
    
    if day_name == "Monday-WorkingHours.pcap":
        # Only normal traffic for Monday (baseline)
        all_packets.extend(generate_normal_traffic(packet_count, start_time))
        
    elif day_name == "Tuesday-WorkingHours.pcap":
        # Normal traffic + SSH-Patator attack
        all_packets.extend(generate_normal_traffic(int(packet_count * 0.8), start_time))
        attack_start = start_time + 3600 * 5.5  # 13:53 attack time
        all_packets.extend(generate_attack_traffic("SSH-Patator", int(packet_count * 0.2), attack_start))
        
    elif day_name == "Wednesday-WorkingHours.pcap":
        # Normal traffic + DoS attacks
        all_packets.extend(generate_normal_traffic(int(packet_count * 0.7), start_time))
        attack_start = start_time + 3600 * 6  # 14:00 attack time
        all_packets.extend(generate_attack_traffic("DoS_slowloris", int(packet_count * 0.3), attack_start))
        
    elif day_name == "Thursday-WorkingHours.pcap":
        # Normal traffic + Web attacks
        all_packets.extend(generate_normal_traffic(int(packet_count * 0.8), start_time))
        attack_start = start_time + 3600 * 7.2  # 15:14 attack time
        all_packets.extend(generate_attack_traffic("Web_Attack_XSS", int(packet_count * 0.2), attack_start))
        
    elif day_name == "Friday-WorkingHours.pcap":
        # Normal traffic + Bot + PortScan
        all_packets.extend(generate_normal_traffic(int(packet_count * 0.6), start_time))
        attack_start1 = start_time + 3600 * 6.5  # 14:30 bot traffic
        all_packets.extend(generate_attack_traffic("Bot", int(packet_count * 0.2), attack_start1))
        attack_start2 = start_time + 3600 * 10  # 18:04 port scan
        all_packets.extend(generate_attack_traffic("PortScan", int(packet_count * 0.2), attack_start2))
    
    # Sort all packets by time
    all_packets.sort(key=lambda x: x.time)
    
    # Write to PCAP file
    output_file = os.path.join(output_dir, day_name)
    print(f"  Writing {len(all_packets)} packets to {output_file}")
    wrpcap(output_file, all_packets)
    
    return len(all_packets)


def main():
    """Main function to generate all CIC-2017 sample PCAP files."""
    print("=== CIC-IDS2017 Sample PCAP Generator ===")
    
    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "raw", "cic_2017")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # List of PCAP files to generate
    pcap_files = [
        "Monday-WorkingHours.pcap",
        "Tuesday-WorkingHours.pcap", 
        "Wednesday-WorkingHours.pcap",
        "Thursday-WorkingHours.pcap",
        "Friday-WorkingHours.pcap"
    ]
    
    total_packets = 0
    
    # Generate each day's traffic
    for pcap_file in pcap_files:
        packet_count = generate_cic_2017_day(pcap_file, output_dir, packet_count=2000)
        total_packets += packet_count
    
    print(f"\n=== Generation Complete ===")
    print(f"Generated {len(pcap_files)} PCAP files with {total_packets} total packets")
    print(f"Files saved to: {output_dir}")
    print("\nYou can now run the data processing pipeline:")
    print("cd data_processing && python cic_2017_preprocess.py --verbose")


if __name__ == "__main__":
    main()

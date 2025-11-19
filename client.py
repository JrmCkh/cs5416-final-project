#!/usr/bin/env python3
"""
Client script for testing the ML inference pipeline.
Sends one request every 10 seconds for 1 minute (6 requests total).
Requests are sent at fixed intervals regardless of response time.
"""

import os
import time
import requests
import json
import threading
from datetime import datetime
from typing import Dict, Optional

# Read NODE_0_IP from environment variable
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
SERVER_URL = f"http://{NODE_0_IP}/query"

EXTRA_QUERIES = [
    "Do you offer international shipping?",
    "How can I contact customer support?",
    "Where can I find product manuals?",
    "Are there any discounts for bulk orders?",
    "What should I do if I received the wrong item?",
    "How do I track my order?",
    "Can I cancel my order after it's been placed?",
    "What are the shipping options available?",
    "How do I apply a promo code to my order?",
    "What is your privacy policy?",
    "How do I create an account?",
    "What are the terms and conditions of sale?",
    "How do I reset my password?"
]

# Test queries
TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
    "What payment methods do you accept?",
    "How long does shipping typically take?",
    *EXTRA_QUERIES
]

# Shared data structures
results = {}
results_lock = threading.Lock()
requests_sent = []
requests_lock = threading.Lock()


def send_request_async(request_id: str, query: str, send_time: float):
    """Send a single request to the server asynchronously"""
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending request {request_id}")
        print(f"Query: {query}")
        
        payload = {
            'request_id': request_id,
            'query': query
        }
        
        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload, timeout=300)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Response received for {request_id} in {elapsed_time:.2f}s")
            print(f"  Generated Response: {result.get('generated_response', '')[:100]}...")
            print(f"  Sentiment: {result.get('sentiment')}")
            print(f"  Is Toxic: {result.get('is_toxic')}")
            
            with results_lock:
                results[request_id] = {
                    'result': result,
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'success': True
                }
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            
            with results_lock:
                results[request_id] = {
                    'error': f"HTTP {response.status_code}",
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'success': False
                }
            
    except requests.exceptions.Timeout:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} timed out after 300s")
        with results_lock:
            results[request_id] = {
                'error': 'Timeout',
                'send_time': send_time,
                'success': False
            }
    except requests.exceptions.ConnectionError:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Failed to connect to server for {request_id}")
        with results_lock:
            results[request_id] = {
                'error': 'Connection error',
                'send_time': send_time,
                'success': False
            }
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: {str(e)}")
        with results_lock:
            results[request_id] = {
                'error': str(e),
                'send_time': send_time,
                'success': False
            }


def main():
    """
    Main function: sends requests every 10 seconds for 1 minute
    Requests are sent at fixed intervals regardless of response time
    """
    queries_to_run = 18
    print("="*70)
    print("ML INFERENCE PIPELINE CLIENT")
    print("="*70)
    print(f"Server URL: {SERVER_URL}")
    print(f"Sending {queries_to_run} requests")
    print("="*70)
    
    # Check if server is healthy
    try:
        health_response = requests.get(f"http://{NODE_0_IP}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"Server is healthy: {health_response.json()}")
        else:
            print(f"Server health check returned status {health_response.status_code}")
    except:
        print(f"Could not reach server health endpoint")
    
    start_time = time.time()
    threads = []
    
    # Send 6 requests at 10-second intervals
    for i in range(queries_to_run):
        # Calculate when this request should be sent
        target_send_time = start_time + (i * 10)
        
        # Wait until the target send time
        current_time = time.time()
        if current_time < target_send_time:
            wait_time = target_send_time - current_time
            if i > 0:
                print(f"\nWaiting {wait_time:.2f}s before next request...")
            time.sleep(wait_time)
        
        # Send request in a separate thread
        request_id = f"req_{int(time.time())}_{i}"
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        with requests_lock:
            requests_sent.append({
                'request_id': request_id,
                'query': query,
                'send_time': time.time()
            })
        
        thread = threading.Thread(
            target=send_request_async,
            args=(request_id, query, time.time())
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete (with a reasonable timeout)
    print(f"\n\nWaiting for all responses (up to 5 minutes)...")
    for thread in threads:
        thread.join(timeout=320)  # 5 min 20 sec to allow for some buffer
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total requests sent: {queries_to_run}")
    
    with results_lock:
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"Successful responses: {successful}")
        print(f"Failed requests: {queries_to_run - successful}")
    
    print(f"Total elapsed time: {total_time:.2f}s")
    
    # Maximum throughput in requests/minute that your system can handle
    throughput_rps = successful / total_time
    throughput_rpm = throughput_rps * 60
    print(f"Maximum throughput (requests/minute): {throughput_rpm:.2f}request/minute")
    
    with results_lock:
        if results:
            print("\nResults:")
            with requests_lock:
                for i, req_info in enumerate(requests_sent, 1):
                    req_id = req_info['request_id']
                    if req_id in results:
                        res_info = results[req_id]
                        print(f"\n{i}. Request ID: {req_id}")
                        print(f"   Query: {req_info['query'][:60]}...")
                        
                        if res_info.get('success'):
                            result = res_info['result']
                            print(f"   Success (took {res_info['elapsed_time']:.2f}s)")
                            print(f"   Sentiment: {result.get('sentiment')}")
                            print(f"   Is Toxic: {result.get('is_toxic')}")
                            print(f"   Response: {result.get('generated_response', '')[:80]}...")
                        else:
                            print(f"   Failed: {res_info.get('error', 'Unknown error')}")
                    else:
                        print(f"\n{i}. Request ID: {req_id}")
                        print(f"   ⏳ Still pending or not received")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
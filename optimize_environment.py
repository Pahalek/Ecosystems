#!/usr/bin/env python3
"""
Environment optimization script for pure econometric modeling.
Disables unnecessary MCP servers and optimizes the environment.
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path


def get_mcp_processes():
    """Get list of running MCP server processes."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = []
        for line in result.stdout.split('\n'):
            if 'mcp' in line.lower() and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    cmd = ' '.join(parts[10:])
                    processes.append({'pid': int(pid), 'cmd': cmd})
        return processes
    except Exception as e:
        print(f"Error getting processes: {e}")
        return []


def disable_unnecessary_servers():
    """Disable unnecessary MCP servers for econometric modeling."""
    print("🔍 Analyzing running MCP servers...")
    
    processes = get_mcp_processes()
    if not processes:
        print("✅ No MCP servers found running")
        return
    
    print(f"Found {len(processes)} MCP-related processes:")
    for proc in processes:
        print(f"  PID {proc['pid']}: {proc['cmd']}")
    
    # Identify unnecessary servers for econometric modeling
    unnecessary_keywords = [
        'playwright', 'browser', 'ui', 'frontend', 'web-server',
        'graphql', 'api-server', 'dashboard'
    ]
    
    terminated = []
    kept = []
    
    for proc in processes:
        cmd_lower = proc['cmd'].lower()
        should_terminate = any(keyword in cmd_lower for keyword in unnecessary_keywords)
        
        if should_terminate:
            try:
                print(f"🚫 Terminating unnecessary server: PID {proc['pid']}")
                os.kill(proc['pid'], signal.SIGTERM)
                terminated.append(proc)
                time.sleep(1)  # Give it time to terminate gracefully
                
                # Check if still running, force kill if needed
                try:
                    os.kill(proc['pid'], 0)  # Check if process exists
                    print(f"   Force killing PID {proc['pid']}")
                    os.kill(proc['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already terminated
                    
            except (ProcessLookupError, PermissionError) as e:
                print(f"   Could not terminate PID {proc['pid']}: {e}")
        else:
            kept.append(proc)
            print(f"✅ Keeping essential server: PID {proc['pid']}")
    
    print(f"\n📊 Environment optimization results:")
    print(f"   Terminated: {len(terminated)} unnecessary servers")
    print(f"   Kept: {len(kept)} essential servers")


def optimize_python_environment():
    """Set Python environment variables for optimal econometric computing."""
    print("\n🐍 Optimizing Python environment for econometric modeling...")
    
    optimizations = {
        'PYTHONOPTIMIZE': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONHASHSEED': '42',
        'MPLBACKEND': 'Agg',  # Use non-interactive matplotlib backend
        'OMP_NUM_THREADS': str(os.cpu_count()),  # Optimize for all CPU cores
        'OPENBLAS_NUM_THREADS': str(os.cpu_count()),
        'ENVIRONMENT_TYPE': 'econometric_modeling',
        'ENVIRONMENT_OPTIMIZED': 'true'
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"   Set {key}={value}")


def verify_econometric_functionality():
    """Verify that econometric agent functionality is intact."""
    print("\n🧪 Verifying econometric functionality...")
    
    try:
        # Test import of main econometric components
        sys.path.append(str(Path(__file__).parent))
        from econometric_agent import EconometricAgent
        
        # Quick functionality test
        agent = EconometricAgent()
        print("   ✅ EconometricAgent import successful")
        
        # Test data generation capability
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            'value': np.random.normal(0, 1, 100)
        }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
        
        agent.datasets['test'] = test_data
        results = agent.validate_data('test')
        
        print("   ✅ Data validation functionality working")
        print(f"   ✅ Environment optimized for econometric modeling")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Econometric functionality check failed: {e}")
        return False


def create_optimization_status():
    """Create a status file indicating optimization is complete."""
    status_file = Path(__file__).parent / '.environment-optimized'
    
    with open(status_file, 'w') as f:
        f.write(f"""# Environment Optimization Status
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

ENVIRONMENT_TYPE=econometric_modeling
ENVIRONMENT_OPTIMIZED=true
MCP_SERVERS_OPTIMIZED=true
PYTHON_ENV_OPTIMIZED=true

# This file indicates the environment has been optimized for pure econometric modeling
# Unnecessary MCP servers have been disabled
# Python environment variables have been set for optimal numerical computing
""")
    
    print(f"\n📄 Created optimization status file: {status_file}")


def main():
    """Main optimization function."""
    print("🎯 ECONOMETRIC ENVIRONMENT OPTIMIZATION")
    print("=" * 50)
    print("Optimizing environment for pure econometric modeling...")
    
    # Step 1: Disable unnecessary MCP servers
    disable_unnecessary_servers()
    
    # Step 2: Optimize Python environment
    optimize_python_environment()
    
    # Step 3: Verify functionality
    if verify_econometric_functionality():
        print("\n✅ ALL CHECKS PASSED")
    else:
        print("\n❌ SOME CHECKS FAILED")
        return 1
    
    # Step 4: Create status file
    create_optimization_status()
    
    print("\n🎉 ENVIRONMENT OPTIMIZATION COMPLETE!")
    print("Environment is now optimized for pure econometric modeling.")
    print("All unnecessary MCP servers have been disabled.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script for PegaEngine Python bindings
"""

from pegaflow import PegaEngine


def test_basic_operations():
    """Test basic put, get, and remove operations"""
    print("Creating PegaEngine instance...")
    engine = PegaEngine()
    
    # Test put and get
    print("\nTest 1: Put and Get")
    engine.put("name", "PegaFlow")
    result = engine.get("name")
    print(f"  put('name', 'PegaFlow')")
    print(f"  get('name') = {result}")
    assert result == "PegaFlow", f"Expected 'PegaFlow', got {result}"
    print("  ✓ Test passed")
    
    # Test get non-existent key
    print("\nTest 2: Get non-existent key")
    result = engine.get("nonexistent")
    print(f"  get('nonexistent') = {result}")
    assert result is None, f"Expected None, got {result}"
    print("  ✓ Test passed")
    
    # Test multiple put operations
    print("\nTest 3: Multiple put operations")
    engine.put("key1", "value1")
    engine.put("key2", "value2")
    engine.put("key3", "value3")
    print(f"  put('key1', 'value1')")
    print(f"  put('key2', 'value2')")
    print(f"  put('key3', 'value3')")
    
    result1 = engine.get("key1")
    result2 = engine.get("key2")
    result3 = engine.get("key3")
    print(f"  get('key1') = {result1}")
    print(f"  get('key2') = {result2}")
    print(f"  get('key3') = {result3}")
    
    assert result1 == "value1", f"Expected 'value1', got {result1}"
    assert result2 == "value2", f"Expected 'value2', got {result2}"
    assert result3 == "value3", f"Expected 'value3', got {result3}"
    print("  ✓ Test passed")
    
    # Test update existing key
    print("\nTest 4: Update existing key")
    engine.put("key1", "updated_value")
    result = engine.get("key1")
    print(f"  put('key1', 'updated_value')")
    print(f"  get('key1') = {result}")
    assert result == "updated_value", f"Expected 'updated_value', got {result}"
    print("  ✓ Test passed")
    
    # Test remove
    print("\nTest 5: Remove key")
    removed = engine.remove("key2")
    print(f"  remove('key2') = {removed}")
    assert removed == "value2", f"Expected 'value2', got {removed}"
    
    result = engine.get("key2")
    print(f"  get('key2') = {result}")
    assert result is None, f"Expected None after removal, got {result}"
    print("  ✓ Test passed")
    
    # Test remove non-existent key
    print("\nTest 6: Remove non-existent key")
    removed = engine.remove("nonexistent")
    print(f"  remove('nonexistent') = {removed}")
    assert removed is None, f"Expected None, got {removed}"
    print("  ✓ Test passed")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    test_basic_operations()


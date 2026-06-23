import random
import numpy as np
from math_utils import dot_product, magnitude, cosine_similarity, mean, variance

def test_all():
    """Run 100 random tests and compare our functions against NumPy."""
    passed = 0
    total = 100
    
    for i in range(total):
        # Create two random vectors of length between 3 and 10
        length = random.randint(3, 10)
        a = [random.uniform(-10, 10) for _ in range(length)]
        b = [random.uniform(-10, 10) for _ in range(length)]
        
        # 1. Test dot product
        my_dot = dot_product(a, b)
        np_dot = np.dot(a, b)
        if not abs(my_dot - np_dot) < 1e-9:
            print(f"Test {i}: Dot product failed. My: {my_dot}, NumPy: {np_dot}")
            continue
        
        # 2. Test magnitude
        my_mag = magnitude(a)
        np_mag = np.linalg.norm(a)
        if not abs(my_mag - np_mag) < 1e-9:
            print(f"Test {i}: Magnitude failed. My: {my_mag}, NumPy: {np_mag}")
            continue
        
        # 3. Test cosine similarity
        my_cos = cosine_similarity(a, b)
        # NumPy cosine: dot / (norm(a)*norm(b))
        np_cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # Handle near-zero norm: np might return nan, but we return 0.0
        if np.isnan(np_cos):
            # If the NumPy result is nan, our function should also return 0.0
            if my_cos != 0.0:
                print(f"Test {i}: Cosine similarity failed (zero-vector case). My: {my_cos}, NumPy: {np_cos}")
                continue
        else:
            if not abs(my_cos - np_cos) < 1e-9:
                print(f"Test {i}: Cosine similarity failed. My: {my_cos}, NumPy: {np_cos}")
                continue
        
        # 4. Test mean
        my_mean = mean(a)
        np_mean = np.mean(a)
        if not abs(my_mean - np_mean) < 1e-9:
            print(f"Test {i}: Mean failed. My: {my_mean}, NumPy: {np_mean}")
            continue
        
        # 5. Test variance (population variance matches np.var default)
        my_var = variance(a)
        np_var = np.var(a)  # default is population variance
        if not abs(my_var - np_var) < 1e-9:
            print(f"Test {i}: Variance failed. My: {my_var}, NumPy: {np_var}")
            continue
        
        # If we reach here, all checks passed for this test
        passed += 1
    
    print(f"Passed {passed} out of {total} random tests.")
    if passed == total:
        print("✅ All tests passed within 1e-9 tolerance!")
    else:
        print("❌ Some tests failed. See error messages above.")

if __name__ == "__main__":
    test_all()
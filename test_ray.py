# test_ray.py
import ray
ray.init(ignore_reinit_error=True)
@ray.remote
def f(x): return x*x
print(ray.get([f.remote(i) for i in range(4)]))
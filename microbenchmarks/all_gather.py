import datetime
from functools import partial
import random
import string
import jax
import numpy as np
import jsonlines
import argparse

runtime_version = "unknown"
accelerator_type = "unknown"
zone = "unknown"
tpu_cores = -1

matrix_size_GB_to_bandwidth = {}

def simple_timeit(f, *args, tries=10, task=None):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!

  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())

  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"{task}: average time milliseconds: {average_time_ms:.2f}")
  return average_time_ms




def all_gather(MATRIX_SIZE):
  global matrix_size_GB_to_bandwidth
  dtype = jax.numpy.bfloat16
  A = jax.numpy.ones((MATRIX_SIZE, MATRIX_SIZE), dtype=dtype)

  selected_devices = jax.devices()
  mesh = jax.sharding.Mesh(selected_devices, "ouraxis")
  sharded_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("ouraxis")
  )
  unsharded_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(None)
  )

  A = jax.device_put(A, sharded_sharding)

  @partial(jax.jit, out_shardings=unsharded_sharding)
  def unshard_array(input):
    return input

  average_time_ms = simple_timeit(unshard_array, A, task="unshard_array")

  matrix_size_GB = A.size * dtype.dtype.itemsize / 1e9
  number_of_devices = len(jax.devices())
  sharded_matrix_size_GB = matrix_size_GB / number_of_devices
  # Send the data to all other (N-1) devices.
  achieved_bandwidth_GB_s = (
      sharded_matrix_size_GB * (number_of_devices - 1) / (average_time_ms / 1e3)
  )

  matrix_size_GB_to_bandwidth[matrix_size_GB] = achieved_bandwidth_GB_s
  print(
      f"Matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}, {dtype=}, {matrix_size_GB=},"
      f" {achieved_bandwidth_GB_s=}"
  )

def run_benchmark():
  test_start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  trace_name = (
      f"t_all_gather_"
      + "".join(
          random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
      )
      + "_"
      + test_start_time
  )
  trace_dir = f"/tmp/microbenchmark/outputs/{trace_name}"
  jax.profiler.start_trace(trace_dir)

  matrix_size = 1024
  while True:
    try:
      all_gather(matrix_size)
      matrix_size += 1024
      if matrix_size > 10000:
        break
    except MemoryError:
      print(
          "MemoryError: Failed to create or process matrix of size"
          f" {matrix_size} x {matrix_size}.\n"
      )
      break
    except Exception as e:
      print(f"Exception: {e} occurred at size {matrix_size} x {matrix_size}.\n")
      break
  jax.profiler.stop_trace()
  print(f"Trace saved to {trace_dir}")

  test_end_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  global matrix_size_GB_to_bandwidth
  max_achieved_bandwidth_GB_s = max(matrix_size_GB_to_bandwidth.values())
  median_achieved_bandwidth_GB_s = np.percentile(
      list(matrix_size_GB_to_bandwidth.values()), 50
  )
  p90_achieved_bandwidth_GB_s = np.percentile(
      list(matrix_size_GB_to_bandwidth.values()), 90
  )
  jsonl_name = f"all_gather_metrics_" + "".join(
          random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
      )+ "_" + test_start_time + ".jsonl"
  jsonl_path = f"/tmp/microbenchmark/outputs/{jsonl_name}"
  metrics = {
      "metrics": {
          "max_achieved_bandwidth_GB_s": max_achieved_bandwidth_GB_s,
          "median_achieved_bandwidth_GB_s": median_achieved_bandwidth_GB_s,
          "p90_achieved_bandwidth_GB_s": p90_achieved_bandwidth_GB_s,
      },
      "dimensions": {
          "testsuite": "microbenchmark",
          "test_name": "all_gather",
          "accelerator_type": accelerator_type,
          "tpu_cores": tpu_cores,
          "zone": zone,
          "runtime_version": runtime_version,
          "test_start_timestamp": test_start_time,
          "test_end_timestamp": test_end_time,
      },
  }
  with jsonlines.open(jsonl_path, mode="w") as writer:
    writer.write(metrics)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          f'A script to analyze the benchmark results and dump the result to a jsonl file.'
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )
  parser.add_argument(
      '--accelerator_type',
      type=str,
      help='Set the accelerator type, such as `--accelerator_type=v6e`',
  )
  parser.add_argument(
      '--zone',
      type=str,
      help='Set the zone, such as `--zone=us-central1-a`',
  )
  parser.add_argument(
      '--runtime_version',
      type=str,
      help='Set the runtime version, such as `--runtime_version=v2-alpha-tpuv6e`',
  )
  parser.add_argument('--tpu_cores', type=int, help='Set the number of TPU cores, such as `--tpu_cores=4`')
  

  args = parser.parse_args()
  flag_options = vars(args)

  global accelerator_type, zone, runtime_version, tpu_cores
  if flag_options['accelerator_type']:
    accelerator_type = args.accelerator_type
  if flag_options['zone']:
    zone = args.zone
  if flag_options['runtime_version']:
    runtime_version = args.runtime_version
  if flag_options['tpu_cores']:
    tpu_cores = args.tpu_cores
    
  run_benchmark()


if __name__ == '__main__':
  main()
  


# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import sys
import re
from bin import cubff


INIT_SEED = 0
THRESHOLD_ENTROPY = 3

NUM_PROGRAMS = 1024 # * 1024
MAX_EPOCHS = 32*1024*1024*1024 / NUM_PROGRAMS


def StandardParams(f):
  params = cubff.SimulationParams()
  params.num_programs = NUM_PROGRAMS
  params.seed = INIT_SEED
  params.load_from = f
  params.mutation_prob = int(round(0.0001*(1<<30)))
  return params

def find_threshold_epoch(params):
  initial_epoch = None
  epochs = 0
  ok = False
  
  def callback(state):
    nonlocal initial_epoch
    if not initial_epoch:
      initial_epoch = state.epoch
    nonlocal epochs
    epochs = state.epoch
    nonlocal ok
    ok = state.higher_entropy > THRESHOLD_ENTROPY
    return ok or state.epoch > MAX_EPOCHS + initial_epoch
  
  cubff.RunSimulation("bff", params, None, callback)

  if ok:
    return epochs
  return None

files = os.listdir(sys.argv[1])
print('seed,start_epoch,stop_epoch')
while True:
  f = random.choice(files)
  params = StandardParams(os.path.join(sys.argv[1], f))
  start_epoch = int(re.sub('([0-9]*).dat', '\\1', os.path.basename(f)))
  params.seed = random.getrandbits(32)
  print(params.seed, ',', start_epoch, ',', find_threshold_epoch(params))

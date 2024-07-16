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

import sys
from bin import cubff


INIT_SEED = 0
THRESHOLD_ENTROPY = 3


def StandardParams():
  params = cubff.SimulationParams()
  params.num_programs = 131072
  params.seed = INIT_SEED
  params.load_from = sys.argv[1]
  params.mutation_prob = int(round(0.001*(1<<30)))
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
    return ok or state.epoch > 32*1024 + initial_epoch
  
  cubff.RunSimulation("bff_noheads", params, None, callback)

  if ok:
    return epochs
  return None

params = StandardParams()
for s in range(100):
  params.seed = INIT_SEED + s
  print(s, find_threshold_epoch(params))

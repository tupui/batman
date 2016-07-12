# read x and y
for l in open('jpod-data/header'): exec l

# create a snapshot and write it to disk
import snapshot_io
import functions
from pod import snapshot
from space import Point

p = Point([x,y])

snapshot.Snapshot.initialize(snapshot_io.io)
snapshot.Snapshot(p, functions.f1(5, p)).write('output')

# wait for a random duration
import time
import random
time.sleep(5 * random.random())

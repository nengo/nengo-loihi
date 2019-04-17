import os

import matplotlib.pyplot as plt
import nengo
from nengo_loihi.inputs import DVSFileChipNode

datafile = os.path.expanduser(
    '~/workspace/davis_tracking/dataset/retinaTest95.events')
t_start = 2.1

with nengo.Network() as net:
    u = DVSFileChipNode(filename=datafile, t_start=t_start,
                        pool=(10, 10), channels_last=True)
    o = nengo.Node(size_in=18*24)
    nengo.Connection(u[0::2], o, transform=1, synapse=0.01)
    nengo.Connection(u[1::2], o, transform=-1, synapse=0.01)

    p = nengo.Probe(o)

with nengo.Simulator(net, progress_bar=False) as sim:
    sim.run(0.1)

plt.imshow(sim.data[p][-1].reshape(18, 24), vmin=-1, vmax=1)
plt.show()

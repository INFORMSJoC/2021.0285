#  ___________________________________________________________________________
#
#  Parapint
#  Copyright (c) 2020
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import plotly.graph_objects as go
import csv
import math


f = open('schur_results.csv', 'r')
reader = csv.DictReader(f)
nblocks_list = list()
time_list = list()
for row in reader:
    method = row['method']
    size = int(row['# processes'])
    nblocks = int(row['# blocks'])
    n_q_per_block = int(row['n_q_per_block'])
    n_y_multiplier = int(row['n_y_multiplier'])
    n_theta = int(row['n_theta'])
    a_nnz_per_row = int(row['A NNZ per row'])
    time = float(row['Total Time (s)'])

    assert nblocks == size
    assert method == 'Parallel Schur-Complement'
    assert n_q_per_block == 5000
    assert n_y_multiplier == 120
    assert n_theta == 10
    assert a_nnz_per_row == 3

    nblocks_list.append(nblocks)
    time_list.append(time)
f.close()

zipped = list(zip(nblocks_list, time_list))
zipped.sort(key=lambda x: x[0])

nblocks_list = [i[0] for i in zipped]
time_list = [i[1] for i in zipped]
first_time = time_list[0]
time_list = [i/first_time for i in time_list]

marker = go.scatter.Marker(
    symbol='cross',
    color='black',
    size=15,
)

line = go.scatter.Line(
    color='black',
    dash='dot',
    width=5,
)

fig_data = [go.Scatter(
    x=nblocks_list,
    y=time_list,
    mode='markers',
    marker=marker,
)]
fig_data.append(go.Scatter(
    x=nblocks_list,
    y=[1]*len(nblocks_list),
    mode='lines',
    line=line,
))
fig = go.Figure(
    data=fig_data,
)
fig.update_layout(
    xaxis_title="Number of Blocks/Processors",
    yaxis_title=f"Schur-Complement Time<br>({nblocks_list[0]} cores as base)",
    font=dict(size=18),
    showlegend=False,
    yaxis_range=[0,math.ceil(max(time_list))],
)
fig.update_xaxes(type='log', dtick=math.log10(2))
fig.update_yaxes()
fig.show()

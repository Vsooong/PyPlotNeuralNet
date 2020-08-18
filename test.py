
def to_add(name, to, offset="(0,0,0)"):
    return r"""
\pic[shift={ """ + offset + """ }] at """ + "({}-east) ".format(to) + """ 
    {Ball={
        name=""" + name + """,
        fill=\SumColor,
        opacity=0.6
        radius=2.5,
        logo=$+$
        }
    };
"""
text=to_add(name='add1',to='fc1',offset=(1.5,0.,0.)),

print(to_add())
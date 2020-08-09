import sys

sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

# defined your arch
# arch = [
#     to_head( '..' ),
#     to_cor(),
#     to_begin(),
#     to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
#     # block_2ConvPool(name='block1', botton='conv1', top='pool1'),
#     to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
#     to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
#     to_connection( "pool1", "conv2"),
#     to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
#     to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
#     to_connection("pool2", "soft1"),
#     to_end()
#     ]

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # input
    to_input('../examples/fcn8s/cats.jpg'),

    # block-001
    to_ConvConvRelu(name='ccr_b1', s_filer=500, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2, 2),
                    height=40, depth=40),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5),

    *block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2', s_filer=256, n_filer=128, offset="(1,0,0)",
                     size=(32, 32, 3.5), opacity=0.5),
    *block_2ConvPool(name='b3', botton='pool_b2', top='pool_b3', s_filer=128, n_filer=256, offset="(1,0,0)",
                     size=(25, 25, 4.5), opacity=0.5),
    *block_2ConvPool(name='b4', botton='pool_b3', top='pool_b4', s_filer=64, n_filer=512, offset="(1,0,0)", size=(16, 16, 5.5), opacity=0.5),

    # Bottleneck
    # block-005
    # to_ConvConvRelu(name='ccr_b5', s_filer=32, n_filer=(1024, 1024), offset="(2,0,0)", to="(pool_b4-east)",
    #                 width=(8, 8), height=8, depth=8, caption="Bottleneck"),
    # to_SoftMax("soft1", 10, "(3,0,0)", "(pool1-east)", caption="SOFT"),
    to_FullyConnected(name="soft1", s_filer=4096, offset="(1.25,0,0)", to="(pool_b4-east)", width=1, height=1, depth=20,
                      caption="fl"),
    to_connection("pool_b4", "soft1"),

    # Decoder
    *block_Unconv(name="b6", botton="soft1", top='end_b6', s_filer=64, n_filer=512, offset="(2.1,0,0)",
                  size=(16, 16, 5.0), opacity=0.5),
    to_skip(of='ccr_b4', to='ccr_res_b6', pos=1.25),
    *block_Unconv(name="b7", botton="end_b6", top='end_b7', s_filer=128, n_filer=256, offset="(2.1,0,0)",
                  size=(25, 25, 4.5), opacity=0.5),
    to_skip(of='ccr_b3', to='ccr_res_b7', pos=1.25),
    *block_Unconv(name="b8", botton="end_b7", top='end_b8', s_filer=256, n_filer=128, offset="(2.1,0,0)",
                  size=(32, 32, 3.5), opacity=0.5),
    to_skip(of='ccr_b2', to='ccr_res_b8', pos=1.25),

    *block_Unconv(name="b9", botton="end_b8", top='end_b9', s_filer=512, n_filer=64, offset="(2.1,0,0)",
                  size=(40, 40, 2.5), opacity=0.5),
    to_skip(of='ccr_b1', to='ccr_res_b9', pos=1.25),

    to_ConvSoftMax(name="soft1", s_filer=512, offset="(0.75,0,0)", to="(end_b9-east)", width=1, height=40, depth=40,
                   caption="SOFT"),
    to_connection("end_b9", "soft1"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
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
opacity=0.6
arch1 = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # input
    to_input('../examples/fcn8s/cats.jpg'),

    # block-001
    to_ConvRelu(name='ccr_b1', s_filer=500, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=2,
                    height=40, depth=40,fill_color='LightSkyBlue'),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=opacity),

    *block_1ConvPool(name='b2', botton='pool_b1', top='pool_b2', s_filer=256, n_filer=128, offset="(2,0,0)",
                     size=(32, 32, 3.5), opacity=0.5),
    *block_1ConvPool(name='b3', botton='pool_b2', top='pool_b3', s_filer=128, n_filer=256, offset="(2,0,0)",
                     size=(25, 25, 4.5), opacity=0.5),
    *block_1ConvPool(name='b4', botton='pool_b3', top='pool_b4', s_filer=64, n_filer=512, offset="(1.5,0,0)", size=(16, 16, 5.5), opacity=0.5),

    # Bottleneck
    # to_SoftMax("soft1", 10, "(3,0,0)", "(pool1-east)", caption="SOFT"),
    to_FcRelu(name="fc1", s_filer=4096, offset="(3,0,0)", to="(pool_b4-east)", width=2, height=2, depth=66,
                      caption="fl",opacity=0.5),
    to_connection("pool_b4", "fc1"),
    to_add(name='add1', to='fc1', offset="(1.5,0,0)"),
    to_connection("fc1", "add1"),
    to_FullyConnected(name="Common Repr", s_filer=4096, offset="(1.5,0,0)", to="(add1-east)", width=2, height=2, depth=66,
                      caption="Common Repr",fill_color='LimeGreen',opacity=0.6),
    to_connection("add1", "Common Repr"),
    to_FullyConnected(name="fc2", s_filer=4096, offset="(3,0,0)", to="(Common Repr-east)", width=2, height=2, depth=66,
                      caption="f2",opacity=0.5),
    to_connection("Common Repr", "fc2"),
    # Decoder
    *block_1Unconv(name="b6", botton="fc2", top='end_b6', s_filer=64, n_filer=512, offset="(3,0,0)",
                  size=(16, 16, 5.0), opacity=opacity),
    to_skip(of='ccr_b4', to='end_b6', pos=1.25),
    *block_1Unconv(name="b7", botton="end_b6", top='end_b7', s_filer=128, n_filer=256, offset="(2,0,0)",
                  size=(25, 25, 4.5), opacity=opacity),
    to_skip(of='ccr_b3', to='end_b7', pos=1.25),
    *block_1Unconv(name="b8", botton="end_b7", top='end_b8', s_filer=256, n_filer=128, offset="(2,0,0)",
                  size=(32, 32, 3.5), opacity=opacity),
    to_skip(of='ccr_b2', to='end_b8', pos=1.25),

    *block_1Unconv(name="b9", botton="end_b8", top='end_b9', s_filer=512, n_filer=64, offset="(2.5,0,0)",
                  size=(40, 40, 2.5), opacity=opacity,fill_color='LightSkyBlue'),
    # to_skip(of='ccr_b1', to='ccr_res_b9', pos=1.25),

    to_ConvSoftMax(name="soft1", s_filer=512, offset="(2,0,0)", to="(end_b9-east)", width=1, height=40, depth=40,
                   caption="Decoded Imgs"),
    to_connection("end_b9", "soft1"),

    to_end()
]

arch2=[
    to_head('..'),
    to_cor(),
    to_begin(),

    to_FullyConnected(name="fc1", s_filer=4096, offset="(3,0,0)", width=2, height=2, depth=36,
                      caption="fl",opacity=0.5),
    to_FullyConnected(name="fc2", s_filer=4096, offset="(3,0,0)", to="(fc1-east)", width=2, height=2, depth=66,
                      caption="fl",opacity=0.5),
    to_connection("fc1", "fc2"),
    to_FcRelu(name="cr1", s_filer=4096, offset="(3,0,0)", to="(fc2-east)", width=2, height=2, depth=66,
                      caption="cr1",opacity=0.6),
    to_connection("fc2", "cr1"),

    to_FullyConnected(name="cr2", s_filer=4096, offset="(6,0,0)", to="(cr1-east)", width=2, height=2, depth=66,
                      caption="cr2",opacity=0.6),
    # to_connection("cr1", "cr2"),
    to_FullyConnected(name="fc3", s_filer=4096, offset="(3,0,0)", to="(cr2-east)", width=2, height=2, depth=66,
              caption="fl", opacity=0.5),
    to_connection("cr2", "fc3"),
    to_FullyConnected(name="fc4", s_filer=4096, offset="(3,0,0)", to="(fc3-east)", width=2, height=2, depth=36,
                      caption="fl",opacity=0.5),
    to_connection("fc3", "fc4"),
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch1, namefile + '.tex')


if __name__ == '__main__':
    main()

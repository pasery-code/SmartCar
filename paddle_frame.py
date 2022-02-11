import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Pool2D, Conv2D
from paddle.nn.functional import interpolate

import numpy as np


class BasicModel(fluid.dygraph.Layer):
    def __init__(self, num_classes=4):
        super(BasicModel, self).__init__()
        self.pool = Pool2D(pool_size=2, pool_stride=2, pool_type='max')  # 池化层尺寸、步长及池化函数
        self.conv = Conv2D(num_channels=3, num_filters=1, filter_size=1, padding=0, act='relu')
        ''' 
        num_channel:输入数据通道数
        num_filter:每通道上使用的滤波器数量，也即输出数据的通道数
        filter_size:每个滤波器的尺寸
        padding:零填充
        act:激活函数
        '''

    def forward(self, inputs):
        x = self.pool(inputs)
        x = interpolate(x, size=inputs.shape[2::])
        x = self.conv(x)
        return x


def main():
    place = paddle.fluid.CPUPlace()
    # place = paddle.fluid.CUDAPlace(0)     # GPU
    with fluid.dygraph.guard(place):
        model = BasicModel(num_classes=4)
        model.eval()
        input_data = np.random.rand(10, 3, 8, 8).astype(np.float32)       # (batch_size, channels, length, width)
        print("input data shape:", input_data.shape)
        input_data = to_variable(input_data)
        output_data = model(input_data)
        output_data = output_data.numpy()
        print("output data shape:", output_data.shape)


if __name__ == "__main__":
    main()

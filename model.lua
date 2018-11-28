require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

-- cnn: VGG-16 with batch norm
local function create_conv(c1, c2, c3)
    local conv = nn.Sequential()
    conv:add(cudnn.SpatialConvolution(c1, c2, 3, 3, 1, 1, 1, 1, 1))
    conv:add(cudnn.SpatialBatchNormalization(c2, nil, nil, nil))
    conv:add(cudnn.ReLU(true))
    conv:add(cudnn.SpatialConvolution(c2, c3, 3, 3, 1, 1, 1, 1, 1))
    conv:add(cudnn.SpatialBatchNormalization(c3, nil, nil, nil))
    conv:add(cudnn.ReLU(true))
    return conv
end

local function create_deconv(c1, c2, c3)
    local conv = nn.Sequential()
    conv:add(nn.SpatialFullConvolution(c1, c2, 3, 3, 1, 1, 1, 1))
    conv:add(cudnn.SpatialBatchNormalization(c2, nil, nil, nil))
    conv:add(cudnn.ReLU(true))
    conv:add(nn.SpatialFullConvolution(c2, c3, 3, 3, 1, 1, 1, 1))
    conv:add(cudnn.SpatialBatchNormalization(c3, nil, nil, nil))
    conv:add(cudnn.ReLU(true))
    return conv
end

local function create_model(nb_class)
    local conv1 = create_conv(3, 64, 128)
    local conv2 = create_conv(128, 128, 128)
    local conv3 = create_conv(128, 128, 128)
    local conv4 = create_conv(128, 128, 128)

    local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)
    local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)
    local pool3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)

    local deconv4 = create_deconv(128, 128, 128)
    local deconv3 = create_deconv(128+128, 128, 128)
    local deconv2 = create_deconv(128+128, 128, 128)
    local deconv1 = create_deconv(128+128, 64, nb_class)
    deconv1:remove(6)

    -- nngraph
    local input = nn.Identity()()
    local features1 = conv1(input)
    local features2 = conv2(pool1(features1))
    local features3 = conv3(pool2(features2))
    local features4 = conv4(pool3(features3))

    local de_features4
    de_features4 = nn.SpatialMaxUnpooling(pool3)(features4)
    de_features4 = deconv4(de_features4)

    -- theoritically we can fisrt do JoinTable then Unpool, however current implementation of Unpool requires same tensor size as corresponding max-pool
    local de_features3, de_features3_1, de_features3_2
    de_features3_1 = nn.SpatialMaxUnpooling(pool2)(features3)
    de_features3_2 = nn.SpatialMaxUnpooling(pool2)(de_features4)
    de_features3 = nn.JoinTable(1, 3)({de_features3_1, de_features3_2})
    de_features3 = deconv3(de_features3)

    local de_features2, de_features2_2, de_features2_1
    de_features2_1 = nn.SpatialMaxUnpooling(pool1)(features2)
    de_features2_2 = nn.SpatialMaxUnpooling(pool1)(de_features3)
    de_features2 = nn.JoinTable(1, 3)({de_features2_1, de_features2_2})
    de_features2 = deconv2(de_features2)

    local de_features1 = nn.JoinTable(1, 3)({features1, de_features2})
    de_features1 = deconv1(de_features1)

    local model = nn.gModule({input}, {de_features1})
    return model
end

return create_model

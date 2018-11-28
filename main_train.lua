require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'

require 'BatchIterator'
require 'utils'
require 'Rect'

-- config
local config = dofile('config.lua')
config = config.parse(arg)
print(config)
cutorch.setDevice(config.gpuid)

-- dataset
local dataset = load_obj(config.data_file)
local batch_iterator = BatchIterator(config, dataset.ground_truth)

-- model
local model = dofile('model.lua')(#config.classes)
model:cuda()
parameters, gradParameters = model:getParameters()

-- warm-up, otherwise cutorch illegal memory access. wtf
-- local ipt = torch.Tensor(1, 3, 512, 361):cuda()
-- local opt = model:forward(ipt)

-- resume training
if config.resume_training then
    print('loading saved model weight...')
    parameters:copy(torch.load(config.saved_model_weights))
    -- config.optim_state = torch.load(config.saved_optim_state)
end

-- criterion
local class_weights
if config.have_class_weights then
    class_weights = torch.Tensor(config.class_weights)
    class_weights:div(torch.sum(class_weights)):mul(#config.classes)
end
print('class weights for loss')
print(class_weights)
local criterion = nn.CrossEntropyCriterion(class_weights):cuda()

-- logger
local logger = optim.Logger(config.log_path .. 'log', true)
logger.showPlot = false

-- confusion matrix
local confusion = optim.ConfusionMatrix(config.classes)

-- main training
for it_batch = 1, math.floor(config.nb_epoch * batch_iterator.train.data:size(1) / config.batch_size) do
    local batch = batch_iterator:nextBatch('train')

    -- inputs and targets
    local inputs = batch.inputs
    inputs = inputs:cuda()
    local targets = batch.targets
    targets = targets:cuda()
    targets = targets:view(-1, 1)
    
    local feval = function(x)
        -- prepare
        collectgarbage()
        if x ~= parameters then
            parameters:copy(x)
        end
        
        -- output
        local outputs = model:forward(inputs)
        local ch, h, w = outputs:size(2), outputs:size(3), outputs:size(4)

        outputs = outputs:permute(1, 3, 4, 2):contiguous()
        outputs= outputs:view(-1, ch)
        confusion:batchAdd(outputs, targets)

        -- criterion
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        df_do = df_do:view(-1, h, w, ch)
        df_do = df_do:permute(1, 4, 2, 3):contiguous()

        -- bg
        gradParameters:zero()
        model:backward(inputs, df_do)
        
        -- print
        if it_batch % config.print_iters == 0 then
            print(it_batch, f)
        end

        -- log
        if it_batch % config.log_iters == 0 then
            logger:add{['training_loss'] = f}
        end

        -- return
        return f,gradParameters
    end

    -- optimizer
    optim.rmsprop(feval, parameters, config.optim_state)

    -- save
    if it_batch % config.snapshot_iters == 0 then
        print('saving model weight...')
        local filename
        filename = config.model_path .. config.prefix .. 'iter_' .. it_batch .. os.date('_%m.%d_%H.%M') .. '.t7'
        torch.save(filename, parameters)
        filename = config.model_path .. config.prefix .. 'iter_' .. it_batch .. os.date('_%m.%d_%H.%M') .. '_state.t7'
        torch.save(filename, config.optim_state)
    end

    -- confusion
    if it_batch % config.confusion_iters == 0 then
        confusion:updateValids()
        print(config.classes)
        print(string.format('training confusion matrix: %s',tostring(confusion)))
        confusion:zero()
    end

    -- lr
    if it_batch % config.lr_decay == 0 then
        config.optim_state.learningRate = config.optim_state.learningRate / config.lr_decay_t
        config.optim_state.learningRate = math.max(config.optim_state.learningRate, config.optim_state.learningRateMin)
        print('decresing lr... new lr:', config.optim_state.learningRate)
    end
end

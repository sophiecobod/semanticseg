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

-- confusion matrix
local confusion = optim.ConfusionMatrix(config.classes)

-- main training
for it_batch = 1, math.floor(config.nb_epoch * batch_iterator.train.data:size(1) / config.batch_size) do
    local batch = batch_iterator:nextBatch('train')

    -- inputs and targets
    local inputs = batch.inputs
    local targets = batch.targets
    targets = targets:view(-1, 1)

    -- outputs
    local outputs = torch.Tensor(targets:size(1), #config.classes):fill(1)
    confusion:batchAdd(outputs, targets)

    -- confusion
    if it_batch % config.confusion_iters == 0 then
        confusion:updateValids()
        print(config.classes)
        print(string.format('training confusion matrix: %s',tostring(confusion)))
    end
end

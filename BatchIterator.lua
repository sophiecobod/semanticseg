require 'image'
require 'utils'
require 'csvigo'
local json = require 'json'

local BatchIterator = torch.class('BatchIterator')

function BatchIterator:__init(config, dataset)
    -- dataset (before): {image_file_name: , rois: {rect: {minX, minY, maxX, maxY}, class_name: , class_index: }}

    self.batch_size = config.batch_size or 128
    self.pixel_means = config.pixel_means or {0, 0, 0}
    self.max_height = config.max_height

    self.classes = config.classes

    self.data = {}
    for k, v in pairs(dataset) do
        table.insert(self.data, {k, v})
    end
    -- dataset (after): {k: {image_file_name: , rois: {rect: {minX, minY, maxX, maxY}, class_name: , class_index: }}}

    -- self dataset: {train, val, test}
    self.train = {}
    self.val = {}
    self.test = {}

    local order = torch.randperm(#self.data)
    local length = #self.data
    self.train.data = order[{{1, math.floor(length * 0.7)}}]
    self.val.data = order[{{math.floor(length * 0.7), math.floor(length * 0.9)}}]
    self.test.data = order[{{math.floor(length * 0.9), length}}]

    -- order
    self.train.order = torch.randperm(self.train.data:size(1))
    self.val.order = torch.randperm(self.val.data:size(1))
    self.test.order = torch.randperm(self.test.data:size(1))
end

function BatchIterator:setBatchSize(batch_size)
    self.batch_size = batch_size or 128
end

function BatchIterator:nextEntry(set)
    local i = self[set].i or 1
    self[set].i = i
    if i > self[set].data:size(1) then
        self[set].order = torch.randperm(self[set].data:size(1))
        i = 1
    end

    local index = self[set].order[i]
    self[set].i = self[set].i + 1
    return self.data[self[set].data[index]]
end

function BatchIterator:nextBatch(set)
    local batch = {}
    batch.inputs = {}
    batch.targets = {}
    batch.images = {}

    for i = 1, self.batch_size do
        -- get entry
        local entry = self:nextEntry(set)
        local img_name = entry[1]

        -- img
        local img = image.load(img_name, 3)
        local H, W = img:size(2), img:size(3)
        table.insert(batch.images, img)
        img = image.scale(img, self.max_height)

        local h = math.floor(img:size(2) / 8) * 8
        local w = math.floor(img:size(3) / 8) * 8
        img = image.scale(img, w, h)

        -- FIXME: 100*100 warmup, 640*448 or 448*640 okay
        -- local h, w
        -- if img:size(2) > img:size(3) then
        --     h, w = 640, 448
        --     img = image.scale(img, w, h)
        -- else
        --     h, w = 448, 640
        --     img = image.scale(img, w, h)
        -- end

        local scale_h = h / H
        local scale_w = w / W

        -- subtract mean
        for ch = 1, 3 do
            if math.max(unpack(self.pixel_means)) < 1 then
                img[{ch, {}, {}}]:add(-self.pixel_means[ch])
            else
                img[{ch, {}, {}}]:add(-self.pixel_means[ch] / 255)
            end
        end
        table.insert(batch.inputs, img)

        -- target
        local target = torch.Tensor(img:size(2), img:size(3)):fill(1) -- 1 for background
        local ht, wt = target:size(1), target:size(2)
        local rois = entry[2].rois
        for i, roi in ipairs(rois) do
            local minX, minY, maxX, maxY = roi.rect.minX, roi.rect.minY, roi.rect.maxX, roi.rect.maxY
            minX = math.floor(minX * scale_w)
            minX = math.min(wt, (math.max(minX, 1)))

            minY = math.floor(minY * scale_h)
            minY = math.min(ht, (math.max(minY, 1)))

            maxX = math.floor(maxX * scale_w)
            maxX = math.min(wt, (math.max(maxX, 1)))

            maxY = math.floor(maxY * scale_h)
            maxY = math.min(ht, (math.max(maxY, 1)))
            local class_index = roi.class_index
            target[{{minY, maxY}, {minX, maxX}}]:fill(class_index)
        end
        table.insert(batch.targets, target)
    end

    -- format inputs
    local ch, h, w
    ch, h, w= batch.inputs[1]:size(1), batch.inputs[1]:size(2), batch.inputs[1]:size(3)
    batch.inputs= torch.cat(batch.inputs):view(self.batch_size, ch, h, w)

    -- format targets
    ch, h, w = 1, batch.targets[1]:size(1), batch.targets[1]:size(2)
    batch.targets = torch.cat(batch.targets):view(self.batch_size, ch, h, w)

    -- return
    return batch
end

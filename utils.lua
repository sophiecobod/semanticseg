function load_obj(file_name)
  local f = torch.DiskFile(file_name, 'r')
  local obj = f:readObject()
  f:close()
  return obj
end

function convert2img(outputs, colors)
    -- outputs: bz * ch * h * w
    local h, w = outputs:size(3), outputs:size(4)
    local img = torch.Tensor(3, h, w):zero()

    local values, classes = torch.max(outputs, 2)

    for y = 1, h do
        for x = 1, w do
            local class = classes[{1, 1, y, x}]
            local prob = values[{1, 1, y, x}]
            for ch = 1, 3 do
                img[{ch, y, x}] = colors[class][ch] * prob
            end
        end
    end
    return img
end
function convert4crf(outputs, colors)
    -- outputs: bz * ch * h * w
    local h, w = outputs:size(3), outputs:size(4)
    local img = torch.Tensor(3, h, w):zero()
    local map = outputs:clone():permute(1, 2, 4, 3):contiguous():log()

    local values, classes = torch.max(outputs, 2)
    local confidence = values[1][1]:clone():mul(-1):add(1)

    for y = 1, h do
        for x = 1, w do
            local class = classes[{1, 1, y, x}]
            local prob = values[{1, 1, y, x}]
            for ch = 1, 3 do
                img[{ch, y, x}] = colors[class][ch] * prob
            end
        end
    end
    return {img, map[1], confidence}
end

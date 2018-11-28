--- All parameters goes here
local config = config or {}

function config.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Adaptive classification')
	cmd:text()
	-- Parameters
    -- data loader
    cmd:option('-max_height', 512, 'img max height')
    cmd:option('-data_file', '../faster-rcnn.torch/pdf_synthetic_caption.t7', 'data path')
    cmd:option('-cache_path', './cache/', 'cache path')
	cmd:option('-pixel_means', {242.91027188, 241.89195856, 242.5565953}, 'Pixel mean values (RGB order)')
    cmd:option('-classes', {'__background__', 'section', 'figure', 'table', 'text', 'caption', 'list'}, 'classes. (section represents all types of sections)')
    cmd:option('-prefix', 'synthetic_caption_', 'prefix: dataset name')

    -- training and testing
    cmd:option('-gpuid', 1, 'gpu id')
    cmd:option('-optim_state', {rho=0.95, eps=1e-6, learningRate=1e-3, learningRateMin=1e-6, momentum=0.9}, 'optim state')
    cmd:option('-lr_decay', 10000, 'iterations between lr decreses')
    cmd:option('-lr_decay_t', 5, 'lr decay times')
    cmd:option('-nb_epoch', 50, 'number of epoches')
    cmd:option('-batch_size', 1, 'batch size')
    cmd:option('-test_img', 'test/1.jpg', 'test image path')
    cmd:option('-have_class_weights', false, 'whether having class_weights or not')

    -- resume
    cmd:option('-resume_training', false, 'whether resume training')
    cmd:option('-saved_model_weights', './models/synthetic_deconv2_iter_130000_07.25_16.58.t7', 'path to saved model weights')
    cmd:option('-saved_optim_state', './models/synthetic_deconv2_iter_130000_07.25_16.58.t7', 'path to saved model weights')

    -- save/print/log
	cmd:option('-snapshot_iters', 5000, 'Iterations between snapshots (used for saving the network)')
	cmd:option('-print_iters', 20, 'Iterations between print')
	cmd:option('-log_iters', 20, 'Iterations between log')
	cmd:option('-vis_iters', 200, 'Iterations between visualization')
	cmd:option('-confusion_iters', 200, 'Iterations between confusion matrix')
	cmd:option('-model_path','./models/','Path to be used for saving the trained models')
	cmd:option('-log_path','./logs/','Path to be used for logging')

	-- Parsing the command line 
	config = cmd:parse(arg or {})
    config.colors = {{0, 0, 0}, -- black 
                     {1, 0, 0}, -- red
                     {0, 1, 0}, -- green
                     {0, 0, 1}, -- blue
                     {1, 1, 0}, -- yellow
                     {1, 0, 1}, -- magenta
                     {0, 1, 1}, -- cyan
                     {1, 1, 1}  -- white
                    }
    config.class_weights = {10000/2071109637,
                            10000/81518384,
                            10000/177951017,
                            10000/156450436,
                            10000/375301594,
                            10000/44197636,
                            10000/420447296}
    config.class_names_create = {'__background__', 'section', 'figure', 'table', 'text', 'caption', 'list'} -- used when creating train.t7 file
	return config
end

return config
